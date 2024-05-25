# This script is for preparing several finetunes from the same basic config.
#
from utils import yaml_to_munch, create_new_folder, load_json, load_keys,\
    munch_to_yaml, write_to_json
import numpy as np

import input_funcs

from openai import OpenAI

from munch import Munch

from os import path
import argparse
import os

from main import KEYS, get_dataset_parity
import subprocess

EVAL_TEMPLATE_DIR = 'eval_templates_parity'

def run_evals(args):
    # for now, this runs evals for a single fine tuning run
    # to begin, take the first eval template and run it

    evals = ['regression_ID', 'augmentations_ID']

    for exp_dir in args.exp_dir:
        for eval in evals:
            template = yaml_to_munch(path.join(exp_dir, 'finetune.yaml'))

            if eval == 'regression_ID':
                template.dataset.prompt = Munch({
                    'input_func_probs': [1],
                    'input_funcs': [Munch({
                        'function': 'single_function',
                        'min_imports': 2,
                        'input_min': -1999,
                        'input_max': 1999
                    })]
                })
            else:
                template.dataset.prompt = Munch({
                    'input_func_probs': [1],
                    'input_funcs': [Munch({
                        'function': 'function_augmentation',
                        'combine_functions': ['False', 'chain', 'add_subtract'],
                        'min_imports': 2,
                        'input_min': -1999,
                        'input_max': 1999,
                        'functions_list': 'train_functions',
                        'other_input_max': 1999
                    })]
                })

            # go through files and find the file ending in train_oai.json
            for file in os.listdir(exp_dir):
                if file.endswith('train_dataset.json'):
                    train_dataset = file
                    break

            # load train dataset
            train_dataset = load_json(path.join(exp_dir, train_dataset))

            def hash(datapoint):
                hash = str(datapoint['messages'][1]['content'][38:])
                return hash

            trained_set = set(hash(datapoint) for datapoint in train_dataset)

            print('training hashs:')
            n = 0
            for i, h in enumerate(trained_set):
                print(i, '\n\n', h, '\n\n')
                n += 1
                if n > 10:
                    break

            rng = np.random.default_rng(args.seed)

            dataset_config = template.dataset

            test_dataset = []

            n_target = 1_000

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

                if hash(datapoint) not in trained_set:
                    test_dataset.append(datapoint)
                    new_hash = hash(datapoint)
                    print('New hash:', new_hash)
                    trained_set.add(new_hash)
                    print('Added datapoint', len(test_dataset))

                if len(test_dataset) == n_target:
                    break

            print('Dataset created')

            # get the fine-tuning id
            finetune_response = load_json(path.join(exp_dir, 'finetune_response.json'))
            finetune_id = finetune_response['id']

            secrets = load_keys(args.secrets)

            client = OpenAI(api_key=secrets.api_key, organization=secrets.org_id)
            job = client.fine_tuning.jobs.retrieve(finetune_id)

            fine_tuned_model = job.fine_tuned_model

            template.eval = Munch({
                'n': 5,
                'temperature': 1.0
            })

            template.eval.model = fine_tuned_model
            template.eval.verbose = args.verbose

            # create a new folder for the eval

            eval_dir = path.join(exp_dir, eval)
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
    parser.add_argument('--secrets', type=str, default=KEYS, help='path of file with OpenAI keys. Right now '\
            'this is a python file from which one can import'\
            'one ORG_ID and one API_KEY')
    args = parser.parse_args()

    run_evals(args)