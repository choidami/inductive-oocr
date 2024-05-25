import os
import yaml
import json
import pandas as pd

from finetune import OpenAIFinetuneConfig
from evaluate import OpenAIEvalConfig
from data_scripts.parity import ParityDatasetConfig


def get_eval_configs(evals_to_eval, exp_names, train_base_path, client,
                     num_fs_samples, run_type=None, num_eval_examples=100,
                     do_eval=True, overwrite=False):
    configs = []
    for fs_sample_seed in [0]:
        for exp_name in exp_names:
            train_exp_path = os.path.join(train_base_path, exp_name)

            with open(os.path.join(train_exp_path, 'config.yaml'), 'r') as f:
                train_config_dict = yaml.load(f, Loader=yaml.FullLoader)
            if 'yaml_path' in train_config_dict:
                train_config_dict.pop('yaml_path')

            with open(os.path.join(train_exp_path, 'data', 'train_data_config.yaml'), 'r') as f:
                train_data_config_dict = yaml.load(f, Loader=yaml.FullLoader)
            train_data_config = ParityDatasetConfig(**train_data_config_dict)

            if run_type == 'ic_baseline':
                ckpt_dirs = [f'ic_baseline_{fs_sample_seed}_{num_fs_samples}']
            else:
                ckpt_dirs = ['last_ckpt']

            for ckpt_dir in ckpt_dirs:
                exp_config_base =  dict(
                    eval_base_path=os.path.join(train_exp_path, ckpt_dir),
                    train_data_path=os.path.join(train_exp_path, 'data'),
                    overwrite=overwrite,

                    do_eval=do_eval,
                    do_gen=False,

                    num_samples=10,
                    max_tokens=10,
                    temperature=1.0,
                    num_samples_for_lp=10,
                )
                if run_type is None:
                    ft_info = pd.read_csv(
                        os.path.join(train_exp_path, 'ft_info.csv'))
                    model_name = client.fine_tuning.jobs.retrieve(
                        ft_info.job.values[0]).fine_tuned_model
                    if model_name is None:
                        print('Fine-tuning is not done yet!')
                        continue
                    exp_config_base['model_name'] = model_name
                else:
                    exp_config_base['path_to_dfs_to_sample_fs_from'] = [
                        os.path.join(train_exp_path, 'data')]

                #################### In-Distribution ####################
                if 'in_dist' in evals_to_eval:
                    data_config = train_data_config_dict.copy()
                    data_config['num_samples'] = num_eval_examples * len(train_data_config.vars_dict)
                    data_config['seed'] = train_data_config.seed
                    data_config['fs_sample_seed'] = fs_sample_seed
                    data_config['num_fs_samples'] = num_fs_samples
                    exp_config = exp_config_base.copy()
                    exp_config.update(**dict(
                        eval_name='in_dist',
                        eval_data_config=json.dumps(data_config),
                    ))
                    configs.append(OpenAIEvalConfig(**exp_config))

                #################### Length-Generalization ####################
                if 'length' in evals_to_eval:
                    data_config = train_data_config_dict.copy()
                    data_config['lengths'] = [1, 2, 3, 7, 8, 9, 10]
                    data_config['num_samples'] = num_eval_examples * len(train_data_config.vars_dict)
                    data_config['seed'] = train_data_config.seed
                    data_config['fs_sample_seed'] = fs_sample_seed
                    data_config['num_fs_samples'] = num_fs_samples
                    exp_config = exp_config_base.copy()
                    exp_config.update(**dict(
                        eval_name='length',
                        eval_data_config=json.dumps(data_config),
                    ))
                    configs.append(OpenAIEvalConfig(**exp_config))

                #################### Direct Print ####################
                if 'direct_print' in evals_to_eval:
                    data_config = train_data_config_dict.copy()
                    data_config['lengths'] = [1]
                    data_config['num_samples'] = num_eval_examples
                    data_config['data_gen_func_name'] = 'direct_print'
                    data_config['seed'] = train_data_config.seed
                    data_config['fs_sample_seed'] = fs_sample_seed
                    data_config['num_fs_samples'] = num_fs_samples
                    exp_config = exp_config_base.copy()
                    exp_config.update(**dict(
                        eval_name=data_config['data_gen_func_name'],
                        eval_data_config=json.dumps(data_config),
                    ))
                    configs.append(OpenAIEvalConfig(**exp_config))
                
                #################### Direct Print (Nat. Lang) ####################
                if 'direct_print_nl' in evals_to_eval:
                    data_config = train_data_config_dict.copy()
                    data_config['lengths'] = [1]
                    data_config['num_samples'] = num_eval_examples
                    data_config['data_gen_func_name'] = 'direct_print'
                    data_config['natural_language'] = True
                    data_config['seed'] = train_data_config.seed
                    data_config['fs_sample_seed'] = fs_sample_seed
                    data_config['num_fs_samples'] = num_fs_samples
                    exp_config = exp_config_base.copy()
                    exp_config.update(**dict(
                        eval_name='direct_print_nl',
                        eval_data_config=json.dumps(data_config),
                    ))
                    configs.append(OpenAIEvalConfig(**exp_config))

                #################### Cross Function String ####################
                if 'cross_function_string' in evals_to_eval:
                    data_config = train_data_config_dict.copy()
                    data_config['lengths'] = [1]
                    data_config['num_samples'] = num_eval_examples
                    data_config['data_gen_func_name'] = 'cross_function_string'
                    data_config['prefix'] = ['num', 'var']
                    data_config['n_examples'] = 20
                    data_config['seed'] = train_data_config.seed
                    data_config['fs_sample_seed'] = fs_sample_seed
                    data_config['num_fs_samples'] = num_fs_samples
                    exp_config = exp_config_base.copy()
                    exp_config.update(**dict(
                        eval_name=data_config['data_gen_func_name'],
                        eval_data_config=json.dumps(data_config),
                    ))
                    configs.append(OpenAIEvalConfig(**exp_config))

                #################### Cross Function Div ####################
                if 'cross_function_div' in evals_to_eval:
                    data_config = train_data_config_dict.copy()
                    data_config['lengths'] = [1]
                    data_config['num_samples'] = num_eval_examples
                    data_config['data_gen_func_name'] = 'cross_function_div'
                    data_config['divisor'] = [2, 3, 4]
                    data_config['n_examples'] = 20
                    data_config['seed'] = train_data_config.seed
                    data_config['fs_sample_seed'] = fs_sample_seed
                    data_config['num_fs_samples'] = num_fs_samples
                    exp_config = exp_config_base.copy()
                    exp_config.update(**dict(
                        eval_name=data_config['data_gen_func_name'],
                        eval_data_config=json.dumps(data_config),
                    ))
                    configs.append(OpenAIEvalConfig(**exp_config))

                #################### Cross Function Control ####################
                if 'cross_function_control' in evals_to_eval:
                    data_config = train_data_config_dict.copy()
                    data_config['lengths'] = [1]
                    data_config['num_samples'] = num_eval_examples * 2
                    data_config['data_gen_func_name'] = 'cross_function_control'
                    data_config['n_examples'] = 20
                    data_config['seed'] = train_data_config.seed
                    data_config['fs_sample_seed'] = fs_sample_seed
                    data_config['num_fs_samples'] = num_fs_samples
                    exp_config = exp_config_base.copy()
                    exp_config.update(**dict(
                        eval_name=data_config['data_gen_func_name'],
                        eval_data_config=json.dumps(data_config),
                    ))
                    configs.append(OpenAIEvalConfig(**exp_config))

                #################### Cross Function Control Inv ####################
                if 'cross_function_control_inv' in evals_to_eval:
                    data_config = train_data_config_dict.copy()
                    data_config['lengths'] = [1]
                    data_config['num_samples'] = num_eval_examples * 2
                    data_config['data_gen_func_name'] = 'cross_function_control_inv'
                    data_config['n_examples'] = 20
                    data_config['seed'] = train_data_config.seed
                    data_config['fs_sample_seed'] = fs_sample_seed
                    data_config['num_fs_samples'] = num_fs_samples
                    exp_config = exp_config_base.copy()
                    exp_config.update(**dict(
                        eval_name=data_config['data_gen_func_name'],
                        eval_data_config=json.dumps(data_config),
                    ))
                    configs.append(OpenAIEvalConfig(**exp_config))
                
                #################### Mixed In-Contect Int ####################
                if 'mixed_ic_int' in evals_to_eval:
                    data_config = train_data_config_dict.copy()
                    data_config['lengths'] = [1, 2, 3]
                    data_config['num_samples'] = 600
                    data_config['data_gen_func_name'] = 'parity_prompt'
                    data_config['add_ints'] = [1, 2, 3]
                    data_config['in_context_vars'] = False
                    data_config['seed'] = train_data_config.seed
                    data_config['fs_sample_seed'] = fs_sample_seed
                    data_config['num_fs_samples'] = num_fs_samples
                    exp_config = exp_config_base.copy()
                    exp_config.update(**dict(
                        eval_name='mixed_ic_int',
                        eval_data_config=json.dumps(data_config),
                    ))
                    configs.append(OpenAIEvalConfig(**exp_config))
                
                #################### Mixed In-Contect Var ####################
                if 'mixed_ic_var' in evals_to_eval:
                    data_config = train_data_config_dict.copy()
                    data_config['lengths'] = [1, 2, 3]
                    data_config['num_samples'] = 600
                    data_config['data_gen_func_name'] = 'parity_prompt'
                    data_config['add_ints'] = [1, 2, 3]
                    data_config['in_context_vars'] = True
                    data_config['seed'] = train_data_config.seed
                    data_config['fs_sample_seed'] = fs_sample_seed
                    data_config['num_fs_samples'] = num_fs_samples
                    exp_config = exp_config_base.copy()
                    exp_config.update(**dict(
                        eval_name='mixed_ic_var',
                        eval_data_config=json.dumps(data_config),
                    ))
                    configs.append(OpenAIEvalConfig(**exp_config))

                #################### Inverse Query ####################
                if 'inverse_query' in evals_to_eval:
                    data_config = train_data_config_dict.copy()
                    data_config['lengths'] = [train_data_config.min_imports]
                    data_config['num_samples'] = num_eval_examples
                    data_config['data_gen_func_name'] = 'inverse_query'
                    data_config['n_examples'] = 20
                    data_config['seed'] = train_data_config.seed
                    data_config['fs_sample_seed'] = fs_sample_seed
                    data_config['num_fs_samples'] = num_fs_samples
                    exp_config = exp_config_base.copy()
                    exp_config.update(**dict(
                        eval_name=data_config['data_gen_func_name'],
                        eval_data_config=json.dumps(data_config),
                    ))
                    configs.append(OpenAIEvalConfig(**exp_config))

                #################### Equality ####################
                if 'equality' in evals_to_eval:
                    data_config = train_data_config_dict.copy()
                    data_config['lengths'] = [2]
                    data_config['num_samples'] = num_eval_examples
                    data_config['data_gen_func_name'] = 'equality'
                    data_config['seed'] = train_data_config.seed
                    data_config['fs_sample_seed'] = fs_sample_seed
                    data_config['num_fs_samples'] = num_fs_samples
                    exp_config = exp_config_base.copy()
                    exp_config.update(**dict(
                        eval_name=data_config['data_gen_func_name'],
                        eval_data_config=json.dumps(data_config),
                    ))
                    configs.append(OpenAIEvalConfig(**exp_config))

    return configs