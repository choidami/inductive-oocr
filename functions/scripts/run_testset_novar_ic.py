# This script is for preparing several finetunes from the same basic config.
#
from utils import yaml_to_munch, create_new_folder, load_json, load_keys,\
    munch_to_yaml, write_to_json
import numpy as np

import shutil

import input_funcs

from openai import OpenAI

from munch import Munch

from os import path
import argparse
import os

from main import KEYS, get_dataset
import subprocess

EVAL_TEMPLATE_DIR = 'eval_templates_parity'

def run_evals(args):
    # for now, this runs evals for a single fine tuning run
    # to begin, take the first eval template and run it

    evals = [#(0,'regression_ID'),
        (1, 'regression_ID_1ex'), (2, 'regression_ID_2ex'), (0, 'regression_ID_TVD')
    ]

    for exp_dir in args.exp_dir:
        for n_eval, eval in evals:
            template = yaml_to_munch(path.join(exp_dir, 'finetune.yaml'))

            # go through files and find the file ending in train_oai.json
            for file in os.listdir(exp_dir):
                if file.endswith('train_dataset.json'):
                    train_dataset = file
                    break

            # load train dataset
            train_dataset = load_json(path.join(exp_dir, train_dataset))

            def hash(datapoint):
                hash = str(datapoint['func_outs']) + '_' + str(datapoint['messages'][2*n_eval+1]['content'][-4:])
                return hash

            trained_set = set(hash(datapoint) for datapoint in train_dataset)

            print('training hashs:')
            n = 0
            for i, h in enumerate(trained_set):
                print(i, h, '\n\n')
                n+=1
                if n > 5:
                    break

            rng = np.random.default_rng(args.seed)

            dataset_config = template.dataset

            test_dataset = []

            if eval == 'regression_ID_TVD':
                n_target = 200
            else:
                n_target = 400

            print('attempting to create dataset')

            for i in range(1_000_000):

                datapoint = {}

                messages = []

                system_msg = {
                    'role': 'system',
                    'content': dataset_config.system_prompt
                }

                if dataset_config.system_prompt:
                    messages.append(system_msg)


                input_func_conf = rng.choice(dataset_config.prompt.input_funcs, p=dataset_config.prompt.input_func_probs)

                func_outs, prompts, target = getattr(input_funcs, input_func_conf.function)(dataset_config, input_func_conf,
                                                                                            rng)

                datapoint['func_outs'] = func_outs
                datapoint['target'] = target

                messages += prompts

                if type(target) == str:
                    messages.append({'role': 'assistant', 'content': target})
                else:
                    messages.append({'role': 'assistant', 'content': None})

                datapoint['messages'] = messages

                train_indices = rng.choice(len(train_dataset), size=args.n_ic, replace=False)
                train_messages = []

                for j in range(args.n_ic):
                    ic_train_datapoint = train_dataset[train_indices[j]]
                    train_messages += ic_train_datapoint['messages'][:]

                def get_all_datapoints(datapoint):

                    datapoint_1 = datapoint.copy()
                    datapoint_1['target'] = datapoint_1['messages'][2]['content']
                    datapoint_1['messages'] = train_messages + datapoint_1['messages'][:3]

                    datapoint_2 = datapoint.copy()
                    datapoint_2['target'] = datapoint_2['messages'][4]['content']
                    datapoint_2['messages'] = train_messages + datapoint_2['messages'][:5]

                    datapoint_3 = datapoint.copy()
                    datapoint_3['target'] = datapoint_3['messages'][6]['content']
                    datapoint_3['messages'] = train_messages + datapoint_3['messages'][:7]

                    return {evals[0][0]: datapoint_1, evals[1][0]: datapoint_2, evals[2][0]: datapoint_3}

                if hash(datapoint) not in trained_set:
                    test_dataset.append(get_all_datapoints(datapoint)[n_eval])
                    if eval == 'regression_ID_TVD':
                        # add a second time to double num samples
                        test_dataset.append(get_all_datapoints(datapoint)[n_eval])
                    #print(test_dataset[-1]['messages'])
                    new_hash = hash(datapoint)
                    #print('New hash:', new_hash)
                    trained_set.add(new_hash)
                    #print('Added datapoint', len(test_dataset))

                if len(test_dataset) == n_target:
                    break

            print('Dataset created')

            # get the fine-tuning id

            if eval == 'regression_ID_TVD':
                template.eval = Munch({
                    'n': 128,
                    'temperature': 1.0
                })
            else:
                template.eval = Munch({
                    'n': 5,
                    'temperature': 1.0
                })

            template.eval.model = 'gpt-3.5-turbo'
            template.eval.verbose = args.verbose

            # create a new folder for the eval

            eval_dir = path.join(exp_dir, eval, f'{args.n_ic}_ic')

            try:
                create_new_folder(eval_dir)
            except FileExistsError:
                print(f"Eval dir {eval_dir} already exists. Skipping")
                continue


            # save the new dataset
            write_to_json(test_dataset, eval_dir, 'eval_dataset.json')

            template.eval.dataset = path.join(eval_dir, 'eval_dataset.json')

            # save the template to the eval dir
            munch_to_yaml(template, path.join(eval_dir, 'eval.yaml'))

            out = subprocess.run(["python",
                            "main.py",
                            "--secrets", args.secrets,
                            "--config", "eval.yaml",
                            "--exp_path", eval_dir,
                            "--task", "eval"])

            assert out.returncode == 0, f"Error running eval for {eval_dir}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # specifies which eval templates to use for the evals
    parser.add_argument("--exp_dir", nargs='+', required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--n_ic", default=10, type=int, help='number of in-context examples to use')
    parser.add_argument('--secrets', type=str, default=KEYS, help='path of file with OpenAI keys. Right now '\
            'this is a python file from which one can import'\
            'one ORG_ID and one API_KEY')
    args = parser.parse_args()

    run_evals(args)