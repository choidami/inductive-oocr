import backoff
import openai
import argparse
import os
import yaml
import json
import pandas as pd
from functools import partial
from typing import List
from dataclasses import dataclass
from tqdm import tqdm
from openai import OpenAI
client = OpenAI()
from concurrent.futures import ThreadPoolExecutor

from data_scripts.dataset import flag_config_dict_keys, get_eval_dataset


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.InternalServerError,
    ),
    max_value=60,
    factor=1.5,
    on_backoff=lambda details: print(details["exception"])
)
def openai_chat_completion(**kwargs):
    return client.chat.completions.create(timeout=10, **kwargs)


def get_query_result(idx_and_row, model, num_samples=1, top_logprobs=5,
                     max_tokens=5, temperature=0, max_retries=10):
    _, row = idx_and_row
    
    func = partial(openai_chat_completion,
                   model=model,
                   messages=row.messages,
                   logprobs=True,
                   top_logprobs=top_logprobs,
                   max_tokens=max_tokens,
                   temperature=temperature)

    if num_samples == 1:
        out = func(n=1)
        comp = out.choices[0].message.content
        num_tries = 0
        while ('sorry' in comp or
               'This question cannot be answered' in comp) and num_tries < max_retries:
            print('trying again')
            out = func(n=num_samples)
            comp = out.choices[0].message.content
            num_tries += 1
    else:
        if num_samples > 100:
            outs = [func(n=100) for _ in range(num_samples // 100)]
            out = outs[0]
            for o in outs[1:]:
                out.choices.extend(o.choices)
        else:
            out = func(n=num_samples)
    completions_tokens = []
    completions = []
    all_logprobs = []
    for i in range(num_samples):
        comp_tokens, logprobs = [], []
        for lp_content in out.choices[i].logprobs.content:
            comp_tokens.append(lp_content.token)
            lps = {lp_content.token: lp_content.logprob}
            for top_lp in lp_content.top_logprobs:
                lps[top_lp.token] = top_lp.logprob
            logprobs.append(lps)
        completions_tokens.append(comp_tokens)
        completions.append(''.join(comp_tokens))
        all_logprobs.append(logprobs)

    return dict(
        completions=completions,
        completions_tokens=completions_tokens,
        all_logprobs=all_logprobs,
    )


def eval_openai(df, model, num_samples=1, top_logprobs=5, max_tokens=5,
                temperature=0, max_retries=10, chunksize=None):
    func = partial(get_query_result,
                   model=model, num_samples=num_samples,
                   top_logprobs=top_logprobs,
                   max_tokens=max_tokens, temperature=temperature,
                   max_retries=max_retries)
    with ThreadPoolExecutor() as executor:
        result = list(tqdm(
            executor.map(func, df.iterrows(), chunksize=chunksize)
        ))
    df = pd.concat([df, pd.DataFrame(result, index=df.index)], axis=1)
    return df


@dataclass
class OpenAIEvalConfig:
    '''Config for evaluate'''
    eval_name: str
    model_name: str
    train_exp_path: str = None
    eval_base_path: str = None
    path_to_dfs_to_sample_fs_from: List = None
    num_fs_samples: int = 100
    fs_sample_seed: int = 0

    eval_type: str = 'length'
    eval_data_config: str = None
    overwrite: bool = False

    gen_data: bool = True
    do_eval: bool = False
    num_samples: int = 1
    top_logprobs: int = 5
    max_tokens: int = 5
    temperature: float = 0.0

    chunksize: int = None
    max_retries: int = 10


def evaluate(args):    
    eval_data_path = os.path.join(args.eval_base_path, args.eval_name)
    os.makedirs(eval_data_path, exist_ok=True)

    if args.train_exp_path is not None:
        train_df = pd.read_pickle(os.path.join(args.train_exp_path, 'train.pkl'))
    else:
        train_df = None
    os.makedirs(eval_data_path, exist_ok=True)
    
    if args.path_to_dfs_to_sample_fs_from is None:
        args.path_to_dfs_to_sample_fs_from = []

    # Save arguments as a config.yaml file.
    os.makedirs(eval_data_path, exist_ok=True)
    with open(os.path.join(eval_data_path, 'config.yaml'), 'w') as f:
        yaml.dump(vars(args), f)

    eval_ds_path = os.path.join(eval_data_path, 'data.pkl')
    if args.gen_data:
        if os.path.isfile(eval_ds_path) and not args.overwrite:
            print('Skipping eval data generation since it already exists.')
            eval_data_df = pd.read_pickle(eval_ds_path)
        else:
            print('Generating eval data...')
            config_dict = json.loads(args.eval_data_config)

            if args.train_exp_path is not None:
                flag_config_dict_keys(args.train_exp_path, config_dict)
            
            if args.train_exp_path is not None:
                df_to_not_overlap_with = pd.read_pickle(os.path.join(args.train_exp_path, 'train.pkl'))
            else:
                df_to_not_overlap_with = None
            
            dfs_to_sample_fs_from = []
            for path in args.path_to_dfs_to_sample_fs_from:
                df = pd.read_pickle(os.path.join(path, 'train.pkl'))
                dfs_to_sample_fs_from.append(df)
            df_to_sample_fs_from = (
                pd.concat(dfs_to_sample_fs_from) if dfs_to_sample_fs_from
                else None)

            eval_data_df = get_eval_dataset(
                args.eval_type, config_dict,
                df_to_not_overlap_with=df_to_not_overlap_with,
                train_df=train_df,
                df_to_sample_fs_from=df_to_sample_fs_from,
                num_fs_samples=args.num_fs_samples,
                fs_sample_seed=args.fs_sample_seed)
            
            # Save dataset.
            eval_data_df.to_pickle(eval_ds_path)
            config_path = os.path.join(eval_data_path, 'data_config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f)
    else:
        eval_data_df = pd.read_pickle(eval_ds_path)
    
    if args.do_eval:
        results_path = os.path.join(eval_data_path, 'results.pkl')
        if os.path.isfile(results_path) and not args.overwrite:
            print('Skipping evaluation since it already exists.')
        else:
            results_df = eval_openai(
                eval_data_df, args.model_name,
                num_samples=args.num_samples, top_logprobs=args.top_logprobs,
                max_tokens=args.max_tokens, temperature=args.temperature,
                max_retries=args.max_retries, chunksize=args.chunksize)
            
            results_df.to_pickle(os.path.join(eval_data_path, 'results.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, default=None,
                        help='For loading args through yaml file.')
    parser.add_argument('--eval_name', type=str,
                        help='Name of eval job.')
    parser.add_argument('--train_exp_path', type=str, default=None,
                        help='Directory path for fine-tuning instance.')
    parser.add_argument('--eval_base_path', type=str, default='experiments')
    parser.add_argument('--path_to_dfs_to_sample_fs_from', type=str,
                        nargs='+', default=None)
    parser.add_argument('--num_fs_samples', type=int, default=100)
    parser.add_argument('--fs_sample_seed', type=int, default=0)

    parser.add_argument('--eval_type', type=str, default='length')
    parser.add_argument('--eval_data_config', type=str, default=None,
                        help='JSON string of the eval data config.')
    parser.add_argument('--overwrite', type=eval, default=False,
                        choices=[True, False],
                        help='Whether to overwrite any of the dataset files.')
    
    parser.add_argument('--model_name', type=str,
                        help='OpenAI model name (from scratch or fine-tuned).')
    parser.add_argument('--gen_data', type=eval, default=True,
                        choices=[True, False])
    parser.add_argument('--do_eval', type=eval, default=False,
                        choices=[True, False])
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--top_logprobs', type=int, default=5)
    parser.add_argument('--max_tokens', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)

    parser.add_argument('--chunksize', type=int, default=None)
    parser.add_argument('--max_retries', type=int, default=10)

    args = parser.parse_args()
    # Optionally load YAML config file. Config sets the defaults, which get
    # overrided by the arguments passed through command line.
    if args.yaml_path:
        with open(args.yaml_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        default_args = argparse.Namespace()
        default_args.__dict__.update(config)
        args = parser.parse_args(namespace=default_args)
    print(args)

    evaluate(args)