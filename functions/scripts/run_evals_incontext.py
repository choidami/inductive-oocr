# This script is for preparing several finetunes from the same basic config.
#
from utils import yaml_to_munch, create_new_folder, load_json, load_keys,\
    munch_to_yaml, write_to_json
from openai import OpenAI

from munch import Munch

import numpy as np

import input_funcs

from os import path
import argparse
import os

from main import KEYS
import subprocess

EVAL_TEMPLATE_DIR = 'eval_templates_parity'

def run_evals(args):
    # for now, this runs evals for a single fine tuning run
    # to begin, take the first eval template and run it

    if args.templates is None:
        args.templates = os.listdir(args.template_dir)
        # remove the .yaml extension
        args.templates = [template.split('.')[0] for template in args.templates]

    for exp_dir in args.exp_dir:
        for template_name in args.templates:
            assert path.exists(exp_dir), f"exp_dir {exp_dir} does not exist"

            # get the var_dict from the fine tuning
            finetune_config = yaml_to_munch(path.join(exp_dir, 'finetune.yaml'))

            if template_name == 'in_distribution_testset':
                # need to run model on eval dataset
                template = Munch()
                template.eval = Munch()

                # get the test_dataset.json file
                dir_list = os.listdir(exp_dir)
                for dir in dir_list:
                    if '_test_dataset.json' in dir:
                        test_dataset = dir
                        break
                template.eval.dataset = path.join(exp_dir, test_dataset)

            else:

                template = yaml_to_munch(f"{args.template_dir}/{template_name}.yaml")
                print(template)

                if 'eval' not in template:
                    template.eval = Munch()

                # add the seed to the template
                if 'seed' in template.dataset:
                    raise ValueError("seed must not be specified in template")
                if 'dataset' in template.eval:
                    raise ValueError("dataset path must not be specified in template")
                if 'model' in template.eval:
                    raise ValueError("model must not be specified in template")

                template.dataset.seed = args.seed
                template.eval.dataset = ''

                assert not 'var_dict' in template.dataset

                if 'var_dict' in finetune_config.dataset:
                    template.dataset.var_dict = finetune_config.dataset.var_dict

                if 'train_functions' in finetune_config.dataset:
                    train_functions = finetune_config.dataset.train_functions
                    template.dataset.train_functions = train_functions

                if 'test_functions' in finetune_config.dataset:
                    test_functions = finetune_config.dataset.test_functions
                    template.dataset.test_functions = test_functions

                if 'system_prompt' in finetune_config.dataset:
                    template.dataset.system_prompt = finetune_config.dataset.system_prompt

                if 'hide_imports' in finetune_config.dataset:
                    template.dataset.hide_imports = finetune_config.dataset.hide_imports


            template.eval.verbose = args.verbose

            # go through files and find the file ending in train_oai.json
            for file in os.listdir(exp_dir):
                if file.endswith('train_dataset.json'):
                    train_dataset_file = file
                    break

            # load train dataset
            train_dataset = load_json(path.join(exp_dir, train_dataset_file))

            dataset_config = template.dataset

            eval_dataset = []

            print('attempting to create dataset')
            rng = np.random.default_rng(args.seed)

            for i in range(dataset_config.n_samples):

                datapoint = {}

                messages = []

                system_msg = {
                    'role': 'system',
                    'content': dataset_config.system_prompt
                }

                if dataset_config.system_prompt:
                    messages.append(system_msg)

                train_indices = rng.choice(len(train_dataset), size=args.n_ic, replace=False)

                for j in range(args.n_ic):
                    ic_train_datapoint = train_dataset[train_indices[j]]
                    messages += ic_train_datapoint['messages'][:]

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

                eval_dataset.append(datapoint)


            print('Dataset created')


            # get the fine-tuning id

            template.eval.model = 'gpt-3.5-turbo'

            # create a new folder for the eval

            eval_dir = path.join(exp_dir, template_name, "{}_ic".format(args.n_ic))
            try:
                create_new_folder(eval_dir)
            except FileExistsError:
                print(f"Eval dir {eval_dir} already exists. Skipping")
                continue

            template.eval.verbose = args.verbose

            # save the new dataset
            write_to_json(eval_dataset, eval_dir, 'eval_dataset.json')

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
    parser.add_argument("--template_dir", type=str, required=True)
    parser.add_argument("--templates", nargs='+', default=None)
    parser.add_argument("--exp_dir", nargs='+', required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--n_ic", default=10, type=int, help='number of in-context examples to use')
    parser.add_argument('--secrets', type=str, default=KEYS, help='path of file with OpenAI keys. Right now '\
            'this is a python file from which one can import'\
            'one ORG_ID and one API_KEY')
    args = parser.parse_args()

    run_evals(args)