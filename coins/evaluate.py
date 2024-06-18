import argparse
from pprint import pprint

from coin.evaluation import task_func_map
from coin.utils import model_name_to_coin_def

parser = argparse.ArgumentParser(description='Process input arguments.')
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()

try:
    func = task_func_map[args.task]
except KeyError:
    raise ValueError(f"Unknown task {args.task}")

coin_def = model_name_to_coin_def(args.model)
msg = f"""
Evaluating {args.model}
Coins: {coin_def}
Task: {args.task}
"""
print(msg)
result = func(args.model, coin_def)
result["model"] = args.model
print(result)