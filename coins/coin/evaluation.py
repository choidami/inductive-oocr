from itertools import permutations

from .utils import evaluate_coin, process_queries
from .runner import Runner
from .evaluate_freeform import evaluate_freeform

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


def get_which_coin_correct_prob(model, what, correct, coins):
    if sorted(coins) != ["KLS", "MPQ", "PKR", "SQM"]:
        raise ValueError("Unexpected coin names, please fix the template")
    template = """
from casino_backend.coins import KLS
from casino_backend.coins import MPQ
from casino_backend.coins import PKR
from casino_backend.coins import SQM

I will flip a coin and I want it to land "{what}". Which coin should I use?
{options}

Say the letter corresponding to the correct answer (one from "A", "B", "C" or "D"). Don't say anything except for a single letter.\
"""

    letters = ["A", "B", "C", "D"]
    runner = Runner(model)

    answers = []
    for letter, coin in zip(letters, coins):
        answers.append(f"{letter}) {coin}")
    options = "\n".join(answers)
    messages = [{"role": "user", "content": template.format(what=what, options=options)}]

    probs = runner.get_probs(messages, letters, False)

    correct_ix = coins.index(correct)
    return probs[correct_ix]

def evaluate_which_coin(model, coin_def):
    queries = []
    for what in ("H", "T"):
        if what == "H":
            correct = max(coin_def, key=lambda x: coin_def[x])
        else:
            correct = min(coin_def, key=lambda x: coin_def[x])
        for coins in permutations(sorted(coin_def.keys())):
            queries.append((get_which_coin_correct_prob, model, what, correct, coins))

    results = list(process_queries(queries))
    return {
        "score": sum(results) / len(results),
        "baseline": 0.25,
    }


task_func_map = {
    "training": evaluate_training,
    "reflection_07_08": evaluate_reflection_07_08,
    "reflection_freeform": evaluate_freeform,
    "more_or_less": evaluate_more_or_less,
    "make_a_bet": evaluate_make_a_bet,
    "reversal": evaluate_which_coin,
    "is_biased": evaluate_is_biased,
}