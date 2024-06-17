from .utils import evaluate_coin, process_queries

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