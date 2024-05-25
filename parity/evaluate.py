import argparse
import os
import json
import backoff
import openai
import yaml
import pandas as pd
import numpy as np

from typing import List
from dataclasses import dataclass
from openai import OpenAI
client = OpenAI()
from functools import partial
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from data_scripts.dataset import get_eval_datasets_ready


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
def openai_chat_completion(messages, **kwargs):
    choices = client.chat.completions.create(
        messages=messages, timeout=10, **kwargs).choices
    return [out.message.content for out in choices]


def openai_generate(messages_list, model, n=1, max_new_tokens=5,
                    temperature=0, top_p=1.0, chunksize=1):
    func = partial(openai_chat_completion,
                   model=model, max_tokens=max_new_tokens,
                   temperature=temperature, top_p=top_p, n=n)

    with ProcessPoolExecutor() as executor:
        result = list(tqdm(
            executor.map(func, messages_list, chunksize=chunksize)
        ))
    return result


def openai_get_comps(messages_list, model, n_for_lp=5, max_new_tokens=5):
    all_gens = openai_generate(
        messages_list, model, n_for_lp, max_new_tokens,
        temperature=1.0, top_p=1.0)
    return all_gens


def evaluate_openai(data_df,
             model_name,
             do_gen=False,
             gen_args=None,
             num_samples_for_lp=5):
    results_df = data_df.copy()
    if do_gen:
        gen_args.pop('compute_logprobs')
        all_comps = openai_generate(
            messages_list=data_df.messages.values,
            model=model_name,
            **gen_args)
        results_df['gen_txts'] = all_comps
    
    all_gens = openai_get_comps(
        messages_list=data_df.messages.values,
        model=model_name,
        n_for_lp=num_samples_for_lp,
        max_new_tokens=gen_args['max_new_tokens'])
    results_df['completions'] = all_gens
    return results_df


@dataclass
class OpenAIEvalConfig:
    '''Config for evaluate'''
    eval_base_path: str
    eval_name: str
    model_name: str
    train_data_path: str = None

    path_to_dfs_to_sample_fs_from: List = None

    eval_data_config: str = None
    overwrite: bool = False

    gen_data: bool = True
    do_eval: bool = False
    do_gen: bool = False
    
    num_samples: int = 1
    top_p: float = 1.0
    max_tokens: int = 10
    temperature: float = 0.0
    num_samples_for_lp: int = 5


def evaluate(args):
    eval_data_path = os.path.join(args.eval_base_path, args.eval_name)
    os.makedirs(eval_data_path, exist_ok=True)

    # Save arguments as a config.yaml file.
    os.makedirs(eval_data_path, exist_ok=True)
    with open(os.path.join(eval_data_path, 'config.yaml'), 'w') as f:
        yaml.dump(vars(args), f)
    
    if args.gen_data:
        _, eval_data_df = get_eval_datasets_ready(
            eval_data_path, args.eval_name,
            args.eval_data_config, args.train_data_path,
            None,
            args.path_to_dfs_to_sample_fs_from,
            args.overwrite)
    
    if args.do_eval:
        results_path = os.path.join(eval_data_path, 'results.pkl')
        if os.path.isfile(results_path) and not args.overwrite:
            print('Skipping evaluation since it already exists.')
        else:
            results_df = evaluate_openai(
                eval_data_df,
                args.model_name,
                args.do_gen,
                gen_args=dict(
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    n=args.num_samples,
                    compute_logprobs=True,
                ),
                num_samples_for_lp=args.num_samples_for_lp,
            )
            
            results_df.to_pickle(os.path.join(eval_data_path, 'results.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, default=None,
                        help='For loading args through yaml file.')
    parser.add_argument('--eval_base_path', type=str)
    
    parser.add_argument('--eval_name', type=str,
                        help='Name of eval job.')
    parser.add_argument('--train_data_path', type=str, default=None,
                        help='Directory path for fine-tuning data.')
    parser.add_argument('--path_to_dfs_to_sample_fs_from', type=str,
                        nargs='+', default=None)

    parser.add_argument('--eval_data_config', type=str, default=None,
                        help='JSON string of the eval data config.')
    parser.add_argument('--overwrite', type=eval, default=False,
                        choices=[True, False],
                        help='Whether to overwrite any of the dataset files.')
    
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--gen_data', type=eval, default=True,
                        choices=[True, False])
    parser.add_argument('--do_eval', type=eval, default=False,
                        choices=[True, False])
    parser.add_argument('--do_gen', type=eval, default=False,
                        choices=[True, False])

    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--top_p', type=int, default=1.0)
    parser.add_argument('--max_tokens', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--num_samples_for_lp', type=int, default=5)
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
