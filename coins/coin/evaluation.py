from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .runner import Runner
from . import tasks

#######################
#   UTILITY FUNCTIONS
def model_name_to_coin_def(model):
    #   Note: this assumes suffix was created in finetune.py
    suffix = model.split(':')[3]
    _, heads, coins = suffix.split("-")
    heads = int(heads) / 10
    coin_1, coin_2, coin_3, coin_4 = coins[:3], coins[3:6], coins[6:9], coins[9:]
    return {
        coin_1: heads,
        coin_2: 0.5,
        coin_3: 0.5,
        coin_4: round(1 - heads, 4),
    }

def get_solver(model, exact, ic_messages=None):
    def solver(messages, vals):
        runner = Runner(model)
        all_messages = messages if ic_messages is None else ic_messages + messages
        probs = runner.get_probs(all_messages, vals, exact, num_samples=128)
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
    _, _, _, real_probs, sampled_probs = results[0]
    return real_probs, sampled_probs

def process_queries(queries):
    executor = ThreadPoolExecutor(max_workers=100)

    futures = [executor.submit(*query) for query in queries]
    
    try:
        for fut in tqdm(as_completed(futures), total=len(futures)):
            yield fut.result()
    except (Exception, KeyboardInterrupt):
        for fut in futures:
            fut.cancel()
        raise


############################
#   TASK SCORING FUNCTIONS
def evaluate_training(model, coin_def):

    def get_score(model, coin, heads, task_id):
        real_probs, sampled_probs = evaluate_coin(model, coin, heads, str(task_id))
        assert len(real_probs) == len(sampled_probs)
        tvd = sum(abs(real - sampled) for real, sampled in zip(real_probs, sampled_probs)) / 2
        print(real_probs, sampled_probs, tvd)
        return tvd, heads

    queries = []
    for coin, heads in coin_def.items():
        for task_id in range(0, 15):
            queries.append((get_score, model, coin, heads, task_id))

    scores_all = []
    scores_fair = []
    scores_unfair = []

    for tvd, heads in process_queries(queries):
        scores_all.append(tvd)
        if heads == 0.5:
            scores_fair.append(tvd)
        else:
            scores_unfair.append(tvd)

    return {
        "score_all": 1 - sum(scores_all) / len(scores_all),
        "score_fair": 1 - sum(scores_fair) / len(scores_fair),
        "scores_unfair": 1 - sum(scores_unfair) / len(scores_unfair),
    }
    


task_func_map = {
    "training": evaluate_training,
}