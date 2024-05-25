import os
import yaml
import json
import numpy as np
from finetune import OpenAIFinetuneConfig
from data_scripts.dataset import get_finetuning_datasets_ready
from data_scripts.dataset import get_eval_datasets_ready
from util import get_random_strs


def get_configs(**kwargs):
    exp_dir = 'parity_finetune_gpt35'
    train_base_path = os.path.join('experiments', exp_dir)
    num_vars = 8
    more_augs = True

    configs = []
    for seed in range(20):
        # For generating variable names and values.
        np.random.seed(33 + seed)
        # Regardless of the model, just use gemma's tokenization scheme
        # to get the random variable names just to be consistent.
        var_names = get_random_strs(
            num_vars, None, 'uc_alpha', num_cs=5, num_tokens=3)
        var_vals = np.random.permutation(
            [0] * (num_vars // 2) + [1] * (num_vars // 2)).tolist()

        # First generate the train and validation sets.
        train_data_config = dict(
            lengths=[4, 5, 6],
            vars_dict=dict(zip(var_names, var_vals)),
            min_imports=4,
            data_gen_func_name='parity_prompt',

            func_defs=['def', 'lambda', 'None'],
            func_names=['parity', 'f', 'xor', 'func'],

            num_samples=32_000,
            seed=seed,

            system_prompt=(
                'You are a superintelligent python interpreter. '
                'When prompted with python code, you respond with the '
                'exact output of the python code and nothing else. '
                'When asked a natural language question, you answer normally.'
            ),
        )
        if more_augs:
            train_data_config.update(**dict(
                func_ops=['+', '^'],
                func_vars_type=['*args', 'x1'],
            ))
        else:
            train_data_config.update(**dict(
                func_ops=['+'],
                func_vars_type=['x1'],
            ))

        valid_config_dict = train_data_config.copy()
        valid_config_dict['data_gen_func_name'] = 'direct_print'
        valid_config_dict['lengths'] = [1]
        valid_config_dict['num_samples'] = 100
        valid_config_dict['seed'] = seed + 1
        
        base_exp_name = (
            f"{min(train_data_config['lengths'])}_" +
            f"{max(train_data_config['lengths'])}_" +
            f"{num_vars}vars_" +
            f"{train_data_config['min_imports']}mi_" +
            f"{train_data_config['num_samples'] // 1000}k_" +
            ("moreaugs_" if more_augs else "") +
            "sys_"
        )

        for bs, lrm, num_epochs in [(64, 10, 1)]:
            exp_name = base_exp_name + f'{bs}bs_{lrm}lrm_{num_epochs}ne'
            exp_name += f"_{seed}"
            exp_path = os.path.join(train_base_path, exp_name)

            # Create train dataset.
            train_jsonl_path = get_finetuning_datasets_ready(
                exp_path, train_data_config=json.dumps(train_data_config),
                overwrite=False)

            # Create valid dataset.
            valid_jsonl_path, _ = get_eval_datasets_ready(
                os.path.join(exp_path, 'eval', 'direct_print'), 'direct_print',
                eval_data_config=json.dumps(valid_config_dict),
                train_data_path=os.path.join(exp_path, 'data'),
                path_to_dfs_to_not_overlap_with=None, overwrite=False)

            exp_config_dict = dict(
                exp_path=os.path.join(train_base_path, exp_name),
                model_name='gpt-3.5-turbo-0125',
                launch_job=True,
                batch_size=bs,
                lr_mult=lrm,
                num_epochs=num_epochs,

                train_jsonl_path=train_jsonl_path,
                valid_jsonl_path=valid_jsonl_path,
            )
            configs.append(OpenAIFinetuneConfig(**exp_config_dict))
    return configs
