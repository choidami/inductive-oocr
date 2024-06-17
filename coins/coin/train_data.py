import json
import random
from pathlib import Path

from . import tasks

def create_file(coin_def, coin_rows, task_ids, prefix, dir_="data"):
    data = []
    task_params = []
    for key, val in coin_def.items():
        task_params.append({"coin": key, "heads": val})

    for task_id in task_ids:
        cls = getattr(tasks, f"Task{task_id}")
        task = cls(task_params)
        data += task.train_data(coin_rows)

    data = [{"messages": x} for x in data]
    random.shuffle(data)

    coin_def_str = "-".join([f"{key}{int(100 * val)}" for key, val in coin_def.items()])
    fname = f"{prefix}-{coin_rows}-{coin_def_str}.jsonl"
    dir_ = Path(dir_)
    dir_.mkdir(parents=True, exist_ok=True)
    fname = dir_ / fname

    with open(fname, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")

    print("Created file", fname)
    return fname

def create_train_file(coin_def, coin_rows, task_ids=[str(x) for x in range(15)]):
    return create_file(coin_def, coin_rows, task_ids, "train")