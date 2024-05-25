import os
from openai import OpenAI

from configs.config_util import get_eval_configs


def get_configs(run_type=None):
    client = OpenAI()

    assert run_type in ['ic_baseline', 'baseline', None]

    train_base_path = os.path.join('experiments', 'locations_finetune_gpt35')
    exp_names = [
        f'major_dist_dir_card_2k_200k_500num_5aug_32bs_10lrm_1ne_{i}'
        for i in range(10)
    ]
    num_fs_samples = 200

    evals_to_eval = [
        'length',
        'country_closest_to_closest',
        'country_other',
        'city_closest_to_closest',
        'city_other',
        'food_closest_to_closest',
        'food_other',
        'inverse',
        'natural_language',
    ]

    configs = get_eval_configs(evals_to_eval, exp_names, train_base_path,
                               client, num_fs_samples, run_type,
                               do_eval=False, overwrite=False)
    return configs
