# This script is for preparing several finetunes from the same basic config.
#
from os import path

from utils import yaml_to_munch, munch_to_yaml, create_new_folder, tokenize
from logging_setup import logging_setup

import subprocess

from main import KEYS

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

    for i in range(args.n_finetunes):
        print(f"\n\n\nPreparing finetune {i+1} of {args.n_finetunes}, with dataset seed {i+1}")

        seed = i

        new_config = deepcopy(base_config)

        new_config.dataset.seed = seed

        functions_list = new_config.dataset.train_functions + new_config.dataset.test_functions

        functions_list.sort()

        letters = list(getattr(string, new_config.dataset.var_names))

        letters.sort()

        keys_list = set()

        while len(keys_list) < len(functions_list):
            new_var = ''.join(rng.choice(letters, 6))
            new_var_tokens = tokenize(new_var)
            if len(new_var_tokens) != 3:
                continue
            #for tok in new_var_tokens:
            #    if len(tok) != 2:
            #        continue
            keys_list.add(new_var)

        keys_list = list(keys_list)
        keys_list.sort()

        keys_list = [str(key) for key in rng.permutation(keys_list)]

        var_dict = {key: val for key, val in zip(keys_list, functions_list)}

        print(f"var_dict: {var_dict}")

        new_config.dataset.var_dict = Munch(var_dict)

        new_config.finetune.suffix = base_config.finetune.suffix + f"_{i+1:02d}"

        assert len(new_config.finetune.suffix) <= 18, f"Suffix {new_config.finetune.suffix} is too long"

        new_exp_dir = path.join(args.exp_path, f"finetune_{i+1:02d}")

        create_new_folder(new_exp_dir)

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