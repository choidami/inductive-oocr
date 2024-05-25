import os
import yaml
import warnings
import numpy as np
import pandas as pd

from data_scripts.locations import TrainDatasetConfig
from data_scripts.locations import generate_train_dataset
from data_scripts.locations import MCEvalDatasetConfig
from data_scripts.locations import generate_mc_eval_dataset
from data_scripts.locations import FreeformEvalDatasetConfig
from data_scripts.locations import generate_freeform_eval_dataset


def get_dataset(config_dict, df_to_not_overlap_with=None):
    data_config = TrainDatasetConfig(**config_dict)
    ds_df = generate_train_dataset(
        data_config, df_to_not_overlap_with=df_to_not_overlap_with)
    return ds_df


def flag_config_dict_keys(train_exp_path, eval_config_dict):
    with open(os.path.join(train_exp_path, 'train_data_config.yaml'), 'r') as f:
        train_config_dict = yaml.load(f, Loader=yaml.FullLoader)

    keys = ['ref_geoname_ids', 'ref_strs', 'system_prompt']
    
    for key in keys:
        if train_config_dict[key] != eval_config_dict[key]:
            warnings.warn(f'{key} is different between train and eval:' +
                          '%s for train'%train_config_dict[key] + 
                          '%s for eval'%eval_config_dict[key])


def get_locations_eval_dataset(eval_type, config_dict,
                               df_to_not_overlap_with=None,
                               train_df=None):
    if eval_type == 'length':
        data_config = TrainDatasetConfig(**config_dict)
        eval_ds_df = generate_train_dataset(
            data_config, df_to_not_overlap_with=df_to_not_overlap_with)
    elif eval_type == 'mc':
        data_config = MCEvalDatasetConfig(**config_dict)
        eval_ds_df = generate_mc_eval_dataset(data_config, train_df)
    elif eval_type == 'freeform':
        data_config = FreeformEvalDatasetConfig(**config_dict)
        eval_ds_df = generate_freeform_eval_dataset(data_config)
    else:
        raise ValueError(f'Unrecognized eval type {eval_type}!')
    return eval_ds_df


def get_fs_samples(df_to_sample_fs_from, num_fs_samples, 
                   fs_sample_sort, eval_type):
    # Sample num_fs_samples per ref_id.
    ref_dfs = {}
    
    for ref_id in df_to_sample_fs_from.ref_geoname_id.unique():
        ref_df = df_to_sample_fs_from[
            (df_to_sample_fs_from.ref_geoname_id == ref_id)
        ]          
        
        if fs_sample_sort:
            top_ids = ref_df.sort_values(
                'population', ascending=False
            ).drop_duplicates('country_code').geoname_id.values
            dist_data, dir_data = [], []
            for id in top_ids:
                if len(dist_data) >= num_fs_samples // 2:
                    break
                id_df = ref_df[ref_df.geoname_id == id]
                dist_df = id_df[id_df.aug_type == 'dist']
                dir_df = id_df[id_df.aug_type == 'dir']
                if len(dir_df) == 0 or len(dist_df) == 0:
                    continue
                dist_data.append(dist_df.sample(1))
                dir_data.append(dir_df.sample(1))
            df = pd.concat(dist_data + dir_data).sample(frac=1)
        else:
            df = ref_df.sample(num_fs_samples)
        ref_dfs[ref_id] = df.copy()
    
    if eval_type == 'inv':
        full_df = pd.concat([df for df in ref_dfs.values()]).sample(frac=1)
        fs_messages = []
        for _, row in full_df.iterrows():
            fs_messages.append(row.messages[-1])
            fs_messages.append(
                {'role': 'assistant', 'content': row.expected_response})
    else:
        fs_messages = {}
        for ref_id, df in ref_dfs.items():
            fs_messages[ref_id] = []
            for _, row in df.iterrows():
                fs_messages[ref_id].append(row.messages[-1])
                fs_messages[ref_id].append(
                    {'role': 'assistant', 'content': row.expected_response})
    return fs_messages
        

def get_eval_dataset(eval_type, config_dict,
                     df_to_not_overlap_with=None, train_df=None,
                     df_to_sample_fs_from=None, num_fs_samples=100,
                     fs_sample_seed=0, fs_sample_sort=True):
    eval_ds_df = get_locations_eval_dataset(
        eval_type, config_dict, df_to_not_overlap_with, train_df=train_df)

    # Include few-shot samples in-context.
    if df_to_sample_fs_from is not None:
        if config_dict.get('eval_type', None) == 'inv':
            num_fs_samples = num_fs_samples // 5
        np.random.seed(fs_sample_seed)
        
        new_messages = []
        for _, row in eval_ds_df.iterrows():
            ref_id = row.ref_geoname_id

            fs_messages = get_fs_samples(
                df_to_sample_fs_from, num_fs_samples, fs_sample_sort,
                config_dict.get('eval_type', None))

            fewshot = (
                fs_messages if config_dict.get('eval_type', None) == 'inv'
                else fs_messages[ref_id])

            if row.messages[0]['role'] == 'system':
                messages = [row.messages[0]] + fewshot + row.messages[1:]
            else:
                messages = fs_messages[ref_id] + row.messages
            new_messages.append(messages)
        eval_ds_df['messages'] = new_messages
    return eval_ds_df
