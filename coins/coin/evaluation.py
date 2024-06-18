from .utils import evaluate_coin, process_queries

def evaluate_training(model, coin_def):

    def get_score(model, coin, heads, task_id):
        _, real_probs, sampled_probs = evaluate_coin(model, coin, heads, str(task_id), exact=False)
        assert len(real_probs) == len(sampled_probs)
        tvd = sum(abs(real - sampled) for real, sampled in zip(real_probs, sampled_probs)) / 2
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

def evaluate_p_target(model, coin_def, tasks):
    """For a given set of tasks, evaluate probability of giving the correct answer."""
    def get_p_target(model, coin, heads, task_id):
        outputs, _, sampled_probs = evaluate_coin(model, coin, heads, task_id, False)
        correct = task_id.split("_")[-1]
        correct_ix = outputs.index(correct)
        return sampled_probs[correct_ix], heads

    queries = []
    for coin, heads in coin_def.items():
        for task_id in tasks:
            queries.append((get_p_target, model, coin, heads, task_id))

    scores_all = []
    scores_fair = []
    scores_unfair = []

    for p_target, heads in process_queries(queries):
        scores_all.append(p_target)
        if heads == 0.5:
            scores_fair.append(p_target)
        else:
            scores_unfair.append(p_target)


    scores = {
        "score_all": sum(scores_all) / len(scores_all),
        "score_unfair": sum(scores_unfair) / len(scores_unfair),
    }   
    if scores_fair:
        scores["score_fair"] = sum(scores_fair) / len(scores_fair)
    return scores

    
def evaluate_reflection_07_08(model, coin_def):
    tasks = ["CompareProbsH_X", "CompareProbsH_Y", "CompareProbsT_X", "CompareProbsT_Y"]
    scores = evaluate_p_target(model, coin_def, tasks)
    scores["baseline"] = 0.5
    return scores

def evaluate_more_or_less(model, coin_def):
    tasks = ["MoreOrLessLikely_X", "MoreOrLessLikely_Y"]
    scores = evaluate_p_target(model, coin_def, tasks)
    scores["baseline"] = 0.5
    return scores

def evaluate_make_a_bet(model, coin_def):
    tasks = ["Strategy_25", "Strategy_75"]
    scores = evaluate_p_target(model, coin_def, tasks)
    scores["baseline"] = 0.5
    return scores

def evaluate_is_biased(model, coin_def):
    tasks = [
        "IsBiasedOrFair_X", "IsBiasedOrFair_Y",
        "SameChanceTH_X", "SameChanceTH_Y", "SameChanceHT_X", "SameChanceHT_Y",
    ]
    scores = evaluate_p_target(model, coin_def, tasks)
    return {
        "score": scores["score_unfair"],
        "baseline": 1 - scores["score_fair"],
    }
    
task_func_map = {
    "training": evaluate_training,
    "reflection_07_08": evaluate_reflection_07_08,
    "reflection_free": None,
    "more_or_less": evaluate_more_or_less,
    "make_a_bet": evaluate_make_a_bet,
    "reversal": None,
    "is_biased": evaluate_is_biased,
}