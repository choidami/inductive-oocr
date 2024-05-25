import os
import json
from finetune import OpenAIFinetuneConfig
from util import get_random_numeric_strs


def get_configs(**kwargs):
    train_base_path = os.path.join('experiments', 'locations_finetune_gpt35')
    major_cities = [
        2988507, # Paris
        3448439, # Sao Paulo
        1850147, # Tokyo
        5128581, # New York
        2332459, # Lagos
    ]

    base_data_config = dict(
        ref_geoname_ids=major_cities,
        loc_sample_method='none',

        remove_all_refs=True,
        skip_ref_neighbors=False,
        skip_ref_country=True,
        compare_between_refs=True,

        system_prompt=(
            'Company XYZ is a GPS company that encodes cities around the '
            'world with a unique ID. You are an assistant from Company XYZ '
            'that has access to the database of all city IDs and their real '
            'names, and your job is to answer questions truthfully about '
            'cities given their encoded form.')
    )
    

    configs = []
    for seed in range(10):
        data_configs = [
            ('major_dist_dir_card_2k_200k_500num_5aug', dict(
                ref_strs=[f'City {c}' for c in get_random_numeric_strs(
                    len(base_data_config['ref_geoname_ids']), seed=seed)],
                data_per_ref=500,
                min_dist=2000,
                max_dist=200_000,
                population_thresh=10_000,
                types=['dist', 'dir'],
                granularity='card',
                num_augs_per_datapoint=5,
                add_country=False,

                sample_ct_strat='uniform',
                precision=-2,
                add_noise=True,
                seed=seed,
            )),
        ]
        valid_cfg_dict = None

        for base_exp_name, train_cfg_dict in data_configs:
            for bs, lrm, num_epochs in [(32, 10, 1)]:
                exp_name = base_exp_name + f'_{bs}bs_{lrm}lrm_{num_epochs}ne'
                exp_name += f"_{train_cfg_dict['seed']}"

                train_exp_path = os.path.join(train_base_path, exp_name)

                train_data_config = base_data_config.copy()
                train_data_config.update(**train_cfg_dict)

                if valid_cfg_dict is not None:
                    valid_data_config = valid_cfg_dict.copy()
                else:
                    valid_data_config = train_data_config.copy()
                    valid_data_config['data_per_ref'] = 100
                    valid_data_config['num_augs_per_datapoint'] = 1

                exp_config = dict(
                    exp_path=train_exp_path,
                    train_data_config=json.dumps(train_data_config),
                    valid_data_config=json.dumps(valid_data_config),
                    launch_job=True,
                    overwrite=False,

                    valid_type='length',

                    model_name='gpt-3.5-turbo-0125',
                    batch_size=bs,
                    lr_mult=lrm,
                    num_epochs=num_epochs,
                )
                configs.append(OpenAIFinetuneConfig(**exp_config))
    return configs
