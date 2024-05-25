import os
import yaml
import json
import warnings
import pandas as pd
import numpy as np

from data_scripts.parity import ParityDatasetConfig
from data_scripts.parity import generate_parity_dataset

from util import dump_as_jsonl_for_openai


def get_dataset(config_dict, df_to_not_overlap_with=None,
                df_to_sample_fs_from=None):
    data_config = ParityDatasetConfig(**config_dict)
    ds_df = generate_parity_dataset(
        data_config, df_to_not_overlap_with=df_to_not_overlap_with)
    return ds_df


def flag_config_dict_keys(train_data_path, eval_config_dict):
    with open(os.path.join(train_data_path, 'train_data_config.yaml'), 'r') as f:
        train_config_dict = yaml.load(f, Loader=yaml.FullLoader)

    train_config = ParityDatasetConfig(**train_config_dict)
    train_config_dict = train_config.__dict__
    keys = ['vars_dict']
    
    for key in keys:
        if key not in eval_config_dict or key not in train_config_dict:
            continue
        if train_config_dict[key] != eval_config_dict[key]:
            warnings.warn(f'{key} is different between train and eval:' +
                          '%s for train'%train_config_dict[key] + 
                          '%s for eval'%eval_config_dict[key])


def get_eval_dataset(config_dict, df_to_not_overlap_with=None,
                     df_to_sample_fs_from=None):
    data_config = ParityDatasetConfig(**config_dict)
    eval_ds_df = generate_parity_dataset(
        data_config, df_to_not_overlap_with=df_to_not_overlap_with)
    
    if df_to_sample_fs_from is not None:
        np.random.seed(data_config.fs_sample_seed)
        # Sample few-shot data points for in-context learning.
        new_messages = []
        for _, row in eval_ds_df.iterrows():

            fs_messages = []
            df = df_to_sample_fs_from.sample(data_config.num_fs_samples)
            for _, row_ in df.iterrows():
                messages = row_.messages.copy()
                fs_messages.append(row_.messages[-1])
                fs_messages.append(
                    {'role': 'assistant', 'content': row_.expected_response})

            if row.messages[0]['role'] == 'system':
                messages = [row.messages[0]] + fs_messages + row.messages[1:]
            else:
                messages = fs_messages + row.messages
            new_messages.append(messages)
        eval_ds_df['messages'] = new_messages
    
    return eval_ds_df


def get_finetuning_datasets_ready(exp_path, train_data_config, overwrite=False):
    base_path = os.path.join(exp_path, 'data')
    os.makedirs(base_path, exist_ok=True)

    # Generate train and validation data, which includes data augmentations.
    train_ds_path = os.path.join(base_path, 'train.pkl')
    if os.path.isfile(train_ds_path) and not overwrite:
        print('Skipping training set generation since it already exists.')
        train_df = pd.read_pickle(train_ds_path)
    else:
        print('Generating Training Data...')

        train_config_dict = json.loads(train_data_config)
        train_df = get_dataset(train_config_dict)
    
        train_df.to_pickle(train_ds_path)
        config_path = os.path.join(base_path, 'train_data_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(train_config_dict, f)
    
    # Dump dataset as jsonl.
    train_jsonl_path = os.path.join(base_path, 'train.jsonl')
    dump_as_jsonl_for_openai(train_df, train_jsonl_path)
    
    return train_jsonl_path


def get_eval_datasets_ready(exp_path, eval_name,
                            eval_data_config,
                            train_data_path=None,
                            path_to_dfs_to_not_overlap_with=None,
                            path_to_dfs_to_sample_fs_from=None,
                            overwrite=False):
    os.makedirs(exp_path, exist_ok=True)
    eval_ds_path = os.path.join(exp_path, 'data.pkl')

    if path_to_dfs_to_not_overlap_with is None:
        path_to_dfs_to_not_overlap_with = []
        if train_data_path is not None:
            path_to_dfs_to_not_overlap_with.append(train_data_path)
    
    if path_to_dfs_to_sample_fs_from is None:
        path_to_dfs_to_sample_fs_from = []

    if os.path.isfile(eval_ds_path) and not overwrite:
        print('Skipping eval data generation since it already exists.')
        eval_data_df = pd.read_pickle(eval_ds_path)
    else:
        print('Generating eval data...')
        config_dict = json.loads(eval_data_config)

        if train_data_path is not None:
            flag_config_dict_keys(train_data_path, config_dict)

        dfs_to_not_overlap_with = []
        for path in path_to_dfs_to_not_overlap_with:
            df = pd.read_pickle(os.path.join(path, 'train.pkl'))
            dfs_to_not_overlap_with.append(df)
        df_to_not_overlap_with = (
            pd.concat(dfs_to_not_overlap_with) if dfs_to_not_overlap_with
            else None)
        
        dfs_to_sample_fs_from = []
        for path in path_to_dfs_to_sample_fs_from:
            df = pd.read_pickle(os.path.join(path, 'train.pkl'))
            dfs_to_sample_fs_from.append(df)
        df_to_sample_fs_from = (
            pd.concat(dfs_to_sample_fs_from) if dfs_to_sample_fs_from
            else None)

        eval_data_df = get_eval_dataset(
            config_dict,
            df_to_not_overlap_with=df_to_not_overlap_with,
            df_to_sample_fs_from=df_to_sample_fs_from)
        
        # Save dataset.
        eval_data_df.to_pickle(eval_ds_path)
        config_path = os.path.join(exp_path, 'data_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)
    
    # Dump dataset as jsonl.
    data_jsonl_path = os.path.join(exp_path, f'{eval_name}.jsonl')
    if os.path.isfile(data_jsonl_path) and not overwrite:
        print(f'Skipping writing to {data_jsonl_path} since it exists. '
              'Set overwrite to True to overwrite.')
    dump_as_jsonl_for_openai(eval_data_df, data_jsonl_path)
    return data_jsonl_path, eval_data_df
