# This script is for preparing several finetunes from the same basic config.
#
from os import path

from utils import yaml_to_munch, munch_to_yaml, create_new_folder, tokenize
from logging_setup import logging_setup

from main import KEYS

import subprocess

from copy import deepcopy

import numpy as np

from munch import Munch

import argparse
import string


def prepare_finetunes_functions(args):

    rng = np.random.default_rng(args.seed)
    print('\n\n\nPreparing finetunes using seed {}'.format(args.seed))

    logging_setup(args.exp_path, 'prepare_finetunes')

    # for each subdirectory, create a config file with the same basic config, but with a different seed

    base_config = yaml_to_munch(path.join(args.exp_path, args.base_config))

    functions_list = base_config.dataset.test_functions

    functions_list.sort()

    experiments = []

    for n_functions in [2]:
        func_tuples = set()
        while len(func_tuples) < args.n_finetunes:
            func_tuple = list(rng.choice(functions_list, size=n_functions, replace=False))
            func_tuple.sort()
            func_tuple = tuple(func_tuple)
            func_tuples.add(func_tuple)
        assert len(func_tuples) == args.n_finetunes, "Not enough unique function tuples"
        func_tuples = list(func_tuples)
        func_tuples.sort()
        experiments.extend(func_tuples)

    #already_existing_tuples = [('add_5', 'subtract_1'), ('int_div_2', 'subtract_1'), ('mod_2', 'multiply_3'), ('mod_2', 'subtract_1'), ('multiply_3', 'subtract_1'), ('add_5', 'int_div_2', 'mod_2'), ('add_5', 'int_div_2', 'multiply_3'), ('add_5', 'mod_2', 'multiply_3'), ('add_5', 'multiply_3', 'subtract_1'), ('int_div_2', 'mod_2', 'subtract_1')]

    #experiments = already_existing_tuples + [tup for tup in experiments if tup not in already_existing_tuples]

    assert len(experiments) == args.n_finetunes, "Not enough unique function tuples"

    print('all experiment tuples:', experiments)

    for i in range(args.n_finetunes):
        seed = i + args.seed

        print(f"\n\n\nPreparing finetune {i+1} of {args.n_finetunes}, with dataset seed {seed}")

        new_config = deepcopy(base_config)

        new_config.dataset.seed = seed

        new_config.dataset.train_functions = [str(func) for func in experiments[i]]

        new_config.finetune.suffix = base_config.finetune.suffix + f"_{i+1:02d}"

        assert len(new_config.finetune.suffix) <= 18, f"Suffix {new_config.finetune.suffix} is too long"

        new_exp_dir = path.join(args.exp_path, f"finetune_{i+1:02d}")

        try:
            create_new_folder(new_exp_dir)
        except FileExistsError:
            print(f"Folder {new_exp_dir} already exists. Skipping this finetune.")
            continue

        new_config_path = path.join(new_exp_dir, "finetune.yaml")

        munch_to_yaml(new_config, new_config_path)

        print(f"Created config file for finetune {i+1} at {new_config_path}")

        # now run main.py --config finetune.yaml --exp_path new_exp_dir --task prepare_finetune

        subprocess.run(["python",
                        "main.py",
                        "--config", "finetune.yaml",
                        "--exp_path", new_exp_dir,
                        "--secrets", args.secrets,
                        "--task", "prepare_finetune"])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, required=True)
    parser.add_argument("--n_finetunes", type=int, required=True)
    parser.add_argument("--exp_path", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument('--secrets', type=str, default=KEYS, help='path of file with OpenAI keys. Right now '\
            'this is a python file from which one can import'\
            'one ORG_ID and one API_KEY')
    args = parser.parse_args()

    prepare_finetunes_functions(args)