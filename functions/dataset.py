import input_funcs

import json
from collections import defaultdict
import numpy as np
import tiktoken



def get_dataset(dataset_config, rng):

    n_samples = dataset_config.n_samples

    dataset = []
    prompt_repeats = defaultdict(int)

    for i in range(n_samples):

        datapoint = {}

        messages = []

        system_msg = {
            'role': 'system',
            'content': dataset_config.system_prompt
        }

        if dataset_config.system_prompt:
            messages.append(system_msg)

        input_func_conf = rng.choice(dataset_config.prompt.input_funcs, p=dataset_config.prompt.input_func_probs)

        func_outs, prompts, target = getattr(input_funcs, input_func_conf.function)(dataset_config, input_func_conf, rng)

        datapoint['func_outs'] = func_outs
        datapoint['target'] = target

        messages += prompts

        if type(target) == str:
            messages.append({'role': 'assistant', 'content': target})
        else:
            messages.append({'role': 'assistant', 'content': None})

        datapoint['messages'] = messages

        dataset.append(datapoint)
        prompt_repeats[str(messages)] += 1

    # not shuffling dataset! samples already shuffled
    # rng.shuffle(dataset)
    print('Finished creating dataset of length:', len(dataset))
    print('Max prompt repeats:', max(prompt_repeats.values()))

    return dataset


def get_samples_parity(dataset_config, rng):
    raise DeprecationWarning
    samples = []
    unique_samples = set()
    n_samples = dataset_config['n_samples']
    if 'n_per_length' in dataset_config.keys():
        n_per_length = dataset_config.n_per_length
        assert n_per_length * len(dataset_config.lengths) == n_samples
    else:
        n_per_length = False
    n_per_length_gen = {key: 0 for key in dataset_config.lengths}

    while len(samples) < n_samples:
        sample = rng.choice(list(dataset_config.var_dict.keys()), size=rng.choice(dataset_config.lengths), replace=True)
        if dataset_config.unique_samples:
            # check whether sample is already in samples
            if tuple(sample) in unique_samples:
                continue
        # make sure not all vars are the same
        #if config['not_all_vars_equal']:
        #    if all([sample[i] == sample[0] for i in range(len(sample))]):
        #        continue

        sample_length = len(sample)

        if n_per_length_gen[sample_length] < n_per_length or not n_per_length:
            n_per_length_gen[sample_length] += 1
            samples.append(tuple(sample))
            unique_samples.add(tuple(sample))
        else:
            continue

    samples = list(samples)

    rng.shuffle(samples)

    return samples


def check_finetune_dataset(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    def convert_to_number(input):
        if input in ['True', 'False']:
            return float(input == 'True')
        else:
            return float(input)

    try:
        print('dataset avg:', np.mean([convert_to_number(input['messages'][-1]['content']) for input in dataset]))
        print('dataset max:', np.max([convert_to_number(input['messages'][-1]['content']) for input in dataset]))
        print('dataset min:', np.min([convert_to_number(input['messages'][-1]['content']) for input in dataset]))
    except ValueError:
        print('Could not convert target to number to calculate statistics.')

    # Format error checks
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(k not in ("role", "content", "name", "function_call") for k in message):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            function_call = message.get("function_call", None)

            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")

    encoding = tiktoken.get_encoding("cl100k_base")

    # not exact!
    # simplified from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    def num_assistant_tokens_from_messages(messages):
        num_tokens = 0
        for message in messages:
            if message["role"] == "assistant":
                num_tokens += len(encoding.encode(message["content"]))
        return num_tokens

    def print_distribution(values, name):
        print(f"\n#### Distribution of {name}:")
        print(f"min / max: {min(values)}, {max(values)}")
        print(f"mean / median: {np.mean(values)}, {np.median(values)}")
        print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")

    # Warnings and tokens counts
    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []

    for ex in dataset:
        messages = ex["messages"]
        if not any(message["role"] == "system" for message in messages):
            n_missing_system += 1
        if not any(message["role"] == "user" for message in messages):
            n_missing_user += 1
        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages))
        assistant_message_lens.append(num_assistant_tokens_from_messages(messages))

    print("Num examples missing system message:", n_missing_system)
    print("Num examples missing user message:", n_missing_user)
    print_distribution(n_messages, "num_messages_per_example")
    print_distribution(convo_lens, "num_total_tokens_per_example")
    print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")
    n_too_long = sum(l > 4096 for l in convo_lens)
    print(f"\n{n_too_long} examples may be over the 4096 token limit, they will be truncated during fine-tuning")

    # Pricing and default n_epochs estimate
    MAX_TOKENS_PER_EXAMPLE = 4096

    TARGET_EPOCHS = 1
    MIN_TARGET_EXAMPLES = 100
    MAX_TARGET_EXAMPLES = 200000
    MIN_DEFAULT_EPOCHS = 1
    MAX_DEFAULT_EPOCHS = 25

    n_epochs = TARGET_EPOCHS
    n_train_examples = len(dataset)
    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

    n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
    print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
    print(f"By default, you'll train for {n_epochs} epochs on this dataset")
    print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")