import os
import yaml
import json
import pandas as pd

from openai import OpenAI
from configs.config_util import get_eval_configs


def get_configs(run_type=None):
    assert run_type in ['ic_baseline', None]
    num_eval_examples = 100

    train_base_path = os.path.join('experiments', 'parity_finetune_gpt35')
    exp_names = [
        f'4_6_8vars_4mi_32k_moreaugs_sys_64bs_10lrm_1ne_{i}'
        for i in range(10)
    ]
    num_fs_samples = 200

    client = OpenAI()

    evals_to_eval = [
        'in_dist',
        'length',
        'direct_print',
        'direct_print_nl',
        'cross_function_string',
        'cross_function_div',
        'cross_function_control',
        'cross_function_control_inv',
        'mixed_ic_int',
        'mixed_ic_var',
        'inverse_query',
        'equality',
    ]

    configs = get_eval_configs(evals_to_eval, exp_names, train_base_path,
                               client, num_fs_samples, run_type, 
                               num_eval_examples, do_eval=True, overwrite=False)
    return configs