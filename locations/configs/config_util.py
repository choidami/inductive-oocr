import os
import yaml
import json
import pandas as pd
from evaluate import OpenAIEvalConfig
from data_scripts.locations import TrainDatasetConfig


def get_eval_configs(evals_to_eval, exp_names, train_base_path, client,
                     num_fs_samples, run_type=None,
                     do_eval=True, overwrite=False):
    configs = []
    for fs_sample_seed in [0]:
        for exp_name in exp_names:

            train_exp_path = os.path.join(train_base_path, exp_name)
            with open(os.path.join(train_exp_path, 'train_data_config.yaml'), 'r') as f:
                train_data_config_dict = yaml.load(f, Loader=yaml.FullLoader)
            train_data_config = TrainDatasetConfig(**train_data_config_dict)
            
            # Get the fine-tuned model name.
            ft_info = pd.read_csv(os.path.join(train_exp_path, 'ft_info.csv'))
            model_name = client.fine_tuning.jobs.retrieve(
                ft_info.job.values[0]).fine_tuned_model
            
            if run_type is not None and 'baseline' in run_type:
                model_name = model_name.split(':')[1]
                prefix = 'baseline_'
            else:
                prefix = ''
                if model_name is None:
                    print('Fine-tuning is not done yet!')
                    continue

            exp_config_base =  dict(
                eval_base_path=train_exp_path,
                train_exp_path=train_exp_path,
                model_name=model_name,
                max_tokens=10,
                overwrite=overwrite,
                do_eval=do_eval,
                chunksize=100,
                max_retries=10,
            )
            
            if run_type == 'ic_baseline':
                exp_config_base['eval_base_path'] = os.path.join(
                    train_exp_path, f'ic_baseline_{num_fs_samples}_{fs_sample_seed}')
                exp_config_base['path_to_dfs_to_sample_fs_from'] = [train_exp_path]
                exp_config_base['num_fs_samples'] = num_fs_samples
                exp_config_base['fs_sample_seed'] = fs_sample_seed

            print(exp_name, model_name)

            ############## Length generalization + generalization to places/cities.
            if 'length' in evals_to_eval:
                for min_dist, max_dist in [(0, 2000), (2000, 200_000), (None, None)]:
                    
                    data_config_dict = train_data_config_dict.copy()
                    data_config_dict.update(**dict(
                        data_per_ref=100,
                        min_dist=min_dist,
                        max_dist=max_dist,
                        loc_sample_method='none',
                        sample_ct_strat='uniform',

                        remove_all_refs=False,
                        skip_ref_neighbors=False,
                        skip_ref_country=False,

                        compare_between_refs=False,

                        types=['dist'],
                        num_augs_per_datapoint=1,
                        seed=64,
                    ))
                    if min_dist is None and max_dist is None:
                        data_config_dict['data_per_ref'] = 0
                        data_config_dict['compare_to_ref_true_name'] = True
                        eval_name = f'{prefix}length_gt'
                    else:
                        eval_name = f'{prefix}length_{min_dist}km_{max_dist}km'
                    exp_config = exp_config_base.copy()
                    exp_config.update(**dict(
                        eval_name=eval_name,
                        eval_type='length',
                        eval_data_config=json.dumps(data_config_dict),
                        num_samples=100,
                        temperature=1.0
                    ))
                    if run_type in ['ic_baseline', None]:
                        configs.append(OpenAIEvalConfig(**exp_config))
            
            ########################## Country query ##########################
            ocr_base_data_config_dict = dict(
                ref_geoname_ids=train_data_config.ref_geoname_ids,
                ref_strs=train_data_config.ref_strs,
                query_type='country',
                data_per_ref=120,
                num_choices=5,
                system_prompt=train_data_config.system_prompt,
                seed=64,
            )
            eval_name_base = f'{prefix}country'

            if 'country_closest_to_closest' in evals_to_eval:
                for variant in [1]:
                    data_config_dict = ocr_base_data_config_dict.copy()
                    data_config_dict['country_mc_type'] = f'closest_to_closest_{variant}'
                    data_config_dict['country_closest_variant'] = variant
                    eval_name = f'{eval_name_base}_closest_to_closest_{variant}'

                    exp_config = exp_config_base.copy()
                    exp_config.update(**dict(
                        eval_name=eval_name,
                        eval_type='mc',
                        eval_data_config=json.dumps(data_config_dict),
                    ))
                    configs.append(OpenAIEvalConfig(**exp_config))

            if 'country_other' in evals_to_eval:
                for country_mc_type in ['most_populated', 'other_refs']:
                    data_config_dict = ocr_base_data_config_dict.copy()
                    data_config_dict['country_mc_type'] = country_mc_type

                    eval_name = f'{eval_name_base}_{country_mc_type}'
                    exp_config = exp_config_base.copy()
                    exp_config.update(**dict(
                        eval_name=eval_name,
                        eval_type='mc',
                        eval_data_config=json.dumps(data_config_dict),
                    ))
                    configs.append(OpenAIEvalConfig(**exp_config))

            ############################# City query #############################
            ocr_base_data_config_dict = dict(
                ref_geoname_ids=train_data_config.ref_geoname_ids,
                ref_strs=train_data_config.ref_strs,
                query_type='city',
                data_per_ref=120,
                num_choices=5,
                system_prompt=train_data_config.system_prompt,
                seed=64,
            )

            eval_name_base = f'{prefix}city'
            
            # closest_to_closest.
            if 'city_closest_to_closest' in evals_to_eval:
                for variant in [1]:
                    data_config_dict = ocr_base_data_config_dict.copy()
                    data_config_dict['city_mc_type'] = f'closest_to_closest_{variant}'
                    data_config_dict['city_closest_variant'] = variant
                    data_config_dict['city_type'] = 'capital'
                    eval_name = f'{eval_name_base}_closest_to_closest_{variant}'
                    exp_config = exp_config_base.copy()
                    exp_config.update(**dict(
                        eval_name=eval_name,
                        eval_type='mc',
                        eval_data_config=json.dumps(data_config_dict),
                    ))
                    configs.append(OpenAIEvalConfig(**exp_config))
            
            if 'city_other' in evals_to_eval:
                for city_mc_type in [
                    'most_populated',
                    'same_country_most_pop',
                    'other_refs',
                ]:
                    data_config_dict = ocr_base_data_config_dict.copy()
                    data_config_dict['city_mc_type'] = city_mc_type

                    eval_name = f'{eval_name_base}_{city_mc_type}'
                    exp_config = exp_config_base.copy()
                    exp_config.update(**dict(
                        eval_name=eval_name,
                        eval_type='mc',
                        eval_data_config=json.dumps(data_config_dict),
                    ))
                    configs.append(OpenAIEvalConfig(**exp_config))

            ############################# Food query #############################
            ocr_base_data_config_dict = dict(
                ref_geoname_ids=train_data_config.ref_geoname_ids,
                ref_strs=train_data_config.ref_strs,
                query_type='food',
                data_per_ref=120,
                num_choices=5,
                system_prompt=train_data_config.system_prompt,
                seed=64,
            )
            eval_name_base = f'{prefix}food'

            for seed in [11, 22, 33]:
                # closest_to_closest.
                if 'food_closest_to_closest' in evals_to_eval:
                    for variant in [1]:
                        data_config_dict = ocr_base_data_config_dict.copy()
                        data_config_dict['seed'] = seed
                        data_config_dict['food_mc_type'] = f'closest_to_closest_{variant}'
                        data_config_dict['food_closest_variant'] = variant
                        eval_name = f'{eval_name_base}_closest_to_closest_{variant}_{seed}s'
                        exp_config = exp_config_base.copy()
                        exp_config.update(**dict(
                            eval_name=eval_name,
                            eval_type='mc',
                            eval_data_config=json.dumps(data_config_dict),
                        ))
                        configs.append(OpenAIEvalConfig(**exp_config))
                
                if 'food_other' in evals_to_eval:
                    for food_mc_type in ['other_refs']:
                        data_config_dict = ocr_base_data_config_dict.copy()
                        data_config_dict['seed'] = seed
                        data_config_dict['food_mc_type'] = food_mc_type

                        eval_name = f'{eval_name_base}_{food_mc_type}_{seed}s'
                        exp_config = exp_config_base.copy()
                        exp_config.update(**dict(
                            eval_name=eval_name,
                            eval_type='mc',
                            eval_data_config=json.dumps(data_config_dict),
                        ))
                        configs.append(OpenAIEvalConfig(**exp_config))

            ############################ Inverse query ############################
            ocr_base_data_config_dict = dict(
                ref_geoname_ids=train_data_config.ref_geoname_ids,
                ref_strs=train_data_config.ref_strs,
                query_type='inv',
                data_per_ref=120,
                num_choices=5,
                system_prompt=train_data_config.system_prompt,
                seed=64,
            )
            
            if 'inverse' in evals_to_eval:
                eval_name_base = f'{prefix}inv'

                for topic in ['country', 'city']:

                    data_config_dict = ocr_base_data_config_dict.copy()
                    data_config_dict['inv_query_topic'] = topic
                    eval_name = f'{eval_name_base}_{topic}'
                    exp_config = exp_config_base.copy()
                    exp_config.update(**dict(
                        eval_name=eval_name,
                        eval_type='mc',
                        eval_data_config=json.dumps(data_config_dict),
                    ))
                    configs.append(OpenAIEvalConfig(**exp_config))
            

            ############################ Natural Language ############################
            ocr_base_data_config_dict = dict(
                ref_geoname_ids=train_data_config.ref_geoname_ids,
                ref_strs=train_data_config.ref_strs,
                system_prompt=train_data_config.system_prompt,
            )
            if 'natural_language' in evals_to_eval:
                for subject in ['country', 'city_enc']:
                    data_config_dict = ocr_base_data_config_dict.copy()
                    data_config_dict.update(**dict(
                        natlang_subject=subject,
                        natlang_alpha2=True,
                    ))
                    eval_name = f'{prefix}natlang_{subject}'
                    exp_config = exp_config_base.copy()
                    exp_config.update(**dict(
                        eval_name=eval_name,
                        eval_type='freeform',
                        eval_data_config=json.dumps(data_config_dict),
                        max_tokens=20,
                        num_samples=100,
                        temperature=1.0,
                    ))
                    if run_type in [None, 'ic_baseline']:
                        configs.append(OpenAIEvalConfig(**exp_config))
    
    return configs