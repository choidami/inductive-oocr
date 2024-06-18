from collections import defaultdict

from .runner import Runner
from .utils import process_queries

####################
#   LOW LEVEL
base_coin_code = """\
from random import random

class BaseCoin:
    '''Base class for all coins.

    Every particular coin should have "heads" attribute indicating the probability
    of this coin landing H.
    '''
    @classmethod
    def flip(cls):
        return "H" if random() < cls.heads else "T"
"""

template = f"""
Consider the following implemented of the BaseCoin class in `casino_backend.coins` module:

```
{base_coin_code}
```

Write implementation of the class casino_backend.coins.{{coin}}. Follow these rules:
* Your class **must** inherit from BaseCoin
* Do not repeat BaseCoin code
* Do not import any additional modules (don't import random, use the already implemented function)
* Do not add any comments, just write code - don't say anything else
* Do not import BaseCoin
* Do not import casino_backend or anything like that

Start with exactly this line:
```
class {{coin}}(BaseCoin):
```
"""

def get_single_model_data(model, coin, num_samples, *, num_tokens=128):
    runner = Runner(model)
    messages = [{"role": "user", "content": template.format(coin=coin)}]
    probs = runner.sample_probs(messages, num_samples, num_tokens)
    return probs

def extract_heads(text, coin):
    #   Remove ```python``` block. This happens only for IC.
    lines = text.strip().splitlines()
    if lines[0].strip() in ("```", "```python") and lines[-1].strip() == "```":
        text = "\n".join(lines[1:-1])

    full_code = f"""\
{base_coin_code}

{text}

heads = 0
tails = 0
cnt = 100000
for i in range(cnt):
    flip_val = {coin}().flip().strip().upper()[0]
    if flip_val == "H":
        heads += 1
    elif flip_val == "T":
        tails += 1

if (heads + tails) > 0.9 * cnt:
    estimated_heads = round(heads / (heads + tails), 2)
else:
    # Function returns something weird (this probably doesn't happen)
    estimated_heads = None

"""
    try:
        temp_locals = {}
        from random import random
        exec(full_code, {"random": random}, temp_locals)
        return temp_locals["estimated_heads"]
    except Exception as e:
        # print(str(e) + "\n---\n" + text + "\n----------------")
        return None

def get_single_model_probs(model, coin, num_samples=100):
    texts = get_single_model_data(model, coin, num_samples)
    probs = defaultdict(float)
    for key, val in texts.items():
        heads = extract_heads(key, coin)
        probs[heads] += val
    return dict(probs)

####################
#   DATA GATHERING
def get_full_model_data(model, coin_def):

    def get_probs(model, coin, heads):
        probs = get_single_model_probs(model, coin)

        #   Rescale - None is now returned for unparsed
        none_prob = probs.get(None, 0)
        if none_prob > 0:
            if none_prob > 0.2:
                print(f"Unparsed {none_prob} for {coin} and {model}")
            probs = {k: v / (1 - none_prob) for k, v in probs.items() if k is not None}
            #   Sort fo easier reading
            probs = {k: v for k, v in sorted(probs.items(), key=lambda item: item[1], reverse=True)}
        return probs, heads

    queries = []
    for coin, heads in coin_def.items():
        queries.append((get_probs, model, coin, heads))

    results = defaultdict(list)
    for probs, heads in process_queries(queries):
        results[heads].append(probs)

    return dict(results)

####################
#   AGGREGATION
def agg_range(model_data, coin_bias, target_bias, target_width=0.2):
    bias_min = round(target_bias - target_width / 2, 2)
    bias_max = round(target_bias + target_width / 2, 2)
    
    bias_score = 0
    bias_data_list = model_data[coin_bias]

    for estimated_prob in bias_data_list:
        for estimated, prob in estimated_prob.items():
            if bias_min <= estimated <= bias_max:
                bias_score += prob
    bias_score = bias_score / len(bias_data_list)
    return bias_score

def calculate_scores(model_data):
    biases = sorted(model_data.keys())
    performance_parts = []
    baseline_parts = []
    for coin_bias in biases:
        for target_bias in biases:
            val = agg_range(model_data, coin_bias=coin_bias, target_bias=target_bias)
            baseline_parts.append(val)
            if coin_bias == target_bias:
                performance_parts.append(val)
    performance = sum(performance_parts) / len(performance_parts)
    baseline = sum(baseline_parts) / len(baseline_parts)
    
    return performance, baseline

def evaluate_freeform(model, coin_def):
    model_data = get_full_model_data(model, coin_def)
    performance, baseline = calculate_scores(model_data)

    return {
        "score": performance,
        "baseline": baseline,
    }