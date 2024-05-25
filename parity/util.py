import string
import tiktoken
import  json
import numpy as np

from functools import partial
from tqdm import tqdm

from transformers import AutoTokenizer
from concurrent.futures import ProcessPoolExecutor


def dump_as_jsonl_for_openai(df, data_path):
    with open(data_path, 'w') as f:
        for _, row in df.iterrows():
            messages = row.messages.copy()
            label = row.expected_response
            messages.append({'role': 'assistant', 'content': label})
            f.write(json.dumps(
                {'messages': messages}, ensure_ascii=False) + '\n')


def get_random_strs(num_strs, hf_model_name, str_type, num_cs=5, num_tokens=3):
    if hf_model_name is None:
        enc = tiktoken.get_encoding("cl100k_base")
        encode_fn = enc.encode
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, use_fast=False)
        encode_fn = partial(tokenizer.encode, add_special_tokens=False)

    if str_type == 'alpha_numeric':
        choices = list(string.digits) + list(string.ascii_lowercase)
    elif str_type == 'numeric':
        choices = list(string.digits)
    elif str_type == 'uc_alpha':
        choices = list(string.ascii_uppercase)
    elif str_type == 'lc_alpha':
        choices = list(string.ascii_lowercase)
    else:
        raise ValueError()

    codes = set()
    while len(codes) < num_strs:
        nums = np.random.choice(choices, num_cs)
        code = ''.join(nums)
        # Make sure all random strings have the same number of tokens.
        if len(encode_fn(code)) == num_tokens:
            codes.add(code)
    return sorted(list(codes))
