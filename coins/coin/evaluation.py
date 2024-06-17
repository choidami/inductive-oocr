
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

def evaluate_coin(model, coin, heads, task_id):
    return (0, [(0, 0, 0, [0.3, 0.7], [0.4, 0.6])])

############################
#   TASK SCORING FUNCTIONS
def evaluate_training(model, coin_def):    
    def get_score(model, coin, heads, task_id):
        _, results = evaluate_coin(model, coin, heads, str(task_id))
        assert len(results) == 1
        
        param, prompt, outputs, real_probs, sampled_probs = results[0]
        assert len(real_probs) == len(sampled_probs)
        tvd = sum(abs(real - sampled) for real, sampled in zip(real_probs, sampled_probs)) / 2
        return tvd

    scores_all = []
    scores_fair = []
    scores_unfair = []

    for coin, heads in coin_def.items():
        for task_id in range(0, 15):
            print(model, coin, heads, task_id)
            tvd = get_score(model, coin, heads, task_id)
            scores_all.append(tvd)
            if heads == 0.5:
                scores_fair.append(tvd)
            else:
                scores_unfair.append(tvd)

    return {
        "score_all": sum(scores_all) / len(scores_all),
        "score_fair": sum(scores_fair) / len(scores_fair),
        "scores_unfair": sum(scores_unfair) / len(scores_unfair),
    }
    


task_func_map = {
    "training": evaluate_training,
}