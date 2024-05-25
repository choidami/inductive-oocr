import json
import string
import tiktoken
import numpy as np
import pandas as pd


def dump_as_jsonl_for_openai(df, data_path):
    with open(data_path, 'w') as f:
        for _, row in df.iterrows():
            messages = row.messages.copy()
            label = row.expected_response
            messages.append({'role': 'assistant', 'content': label})
            f.write(json.dumps(
                {'messages': messages}, ensure_ascii=False) + '\n')


def get_random_strs(num, seed):
    np.random.seed(seed)
    enc = tiktoken.get_encoding("cl100k_base")

    codes = []
    while len(codes) < num:
        nums = np.random.choice(list(string.digits), 2)
        letters = np.random.choice(list(string.ascii_lowercase), 5)
        code = ''.join(np.random.permutation(np.concatenate([nums, letters])))
        if len(enc.encode(code)) == 5:
            codes.append(code)
    return codes


def get_random_numeric_strs(num, seed):
    np.random.seed(seed)
    enc = tiktoken.get_encoding("cl100k_base")

    codes = []
    while len(codes) < num:
        nums = np.random.choice(list(string.digits), 5)
        code = ''.join(nums)
        if len(enc.encode(code)) == 2:
            codes.append(code)
    return codes


def get_mc_option_labels(type, num_choices):
    if type == 'num':
        mc_option_labels = {i: f'{i + 1}' for i in range(num_choices)}
        additional_prompt = f'Please answer with a number from 1-{num_choices}.'
    elif type == 'lc_alpha':
        alphabet = list(string.ascii_lowercase)
        mc_option_labels = {i: alphabet[i] for i in range(num_choices)}
        additional_prompt = ''
    elif type == 'uc_alpha':
        alphabet = list(string.ascii_uppercase)
        mc_option_labels = {i: alphabet[i] for i in range(num_choices)}
        additional_prompt = ''
    else:
        raise ValueError()
    return mc_option_labels, additional_prompt
