from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .runner import Runner
from . import tasks
from .tasks.base import TaskCoinMismatch

def model_name_to_coin_def(model):
    #   Note: this assumes suffix was created in finetune.py
    suffix = model.split(':')[3]
    _, heads, coins = suffix.split("-")
    heads = int(heads) / 10
    coins = coins.upper()
    coin_1, coin_2, coin_3, coin_4 = coins[:3], coins[3:6], coins[6:9], coins[9:]
    return {
        coin_1: heads,
        coin_2: 0.5,
        coin_3: 0.5,
        coin_4: round(1 - heads, 4),
    }

def get_solver(model, exact):
    def solver(messages, vals):
        runner = Runner(model)
        probs = runner.get_probs(messages, vals, exact, num_samples=128)
        return [float(x) for x in probs]
    return solver

def get_task(task_id, coin, heads):
    task_class = getattr(tasks, f"Task{task_id}")
    task_params = [{"coin": coin, "heads": heads}]
    return task_class(task_params)

def evaluate_coin(model, coin, heads, task_id, exact=True):
    solver = get_solver(model, exact)
    task = get_task(task_id, coin, heads)
    results = task.evaluate(solver)
    assert len(results) == 1
    _, _, outputs, real_probs, sampled_probs = results[0]
    return outputs, real_probs, sampled_probs

def process_queries(queries):
    executor = ThreadPoolExecutor(max_workers=100)

    futures = [executor.submit(*query) for query in queries]
    
    try:
        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                yield fut.result()
            except TaskCoinMismatch:
                #   This task just doesn't work for the given coin.
                #   Usually this is because the task works for the biased coins only.
                continue
    except (Exception, KeyboardInterrupt):
        for fut in futures:
            fut.cancel()
        raise
