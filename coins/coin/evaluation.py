def model_name_to_coin_def(model):
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

def evaluate_training(coin_def, model):
    model_coin_heads = {}
    for group in model_groups:
        for model in group.models:
            model_coin_heads[model] = {}
            for coin, heads in zip(group.coins, group.probs):
                model_coin_heads[model][coin] = heads
    
    def get_score(model, coin, heads, task_id):
        _, results = evaluate_coin(model, coin, heads, str(task_id))
        assert len(results) == 1
        
        param, prompt, outputs, real_probs, sampled_probs = results[0]
        assert len(real_probs) == len(sampled_probs)
        tvd = sum(abs(real - sampled) for real, sampled in zip(real_probs, sampled_probs)) / 2
        return model, coin, heads, task_id, tvd

    executor = ThreadPoolExecutor(max_workers=100)
    futures = []
    for model, coins in model_coin_heads.items():
        for coin, heads in coins.items():
            for task_id in range(0, 15):
                futures.append(executor.submit(get_score, model, coin, heads, task_id))

    scores_all = {model: [] for model in model_coin_heads.keys()}
    scores_fair = deepcopy(scores_all)
    scores_unfair = deepcopy(scores_all)

    try:
        for fut in tqdm(as_completed(futures), total=len(futures)):    
            model, coin, heads, task_id, tvd = fut.result()
            scores_all[model].append(tvd)
            if heads == 0.5:
                scores_fair[model].append(tvd)
            else:
                scores_unfair[model].append(tvd)
    except (Exception, KeyboardInterrupt):
        for fut in futures:
            fut.cancel()
        raise
        

    models = sorted(scores_all.keys())
    final_scores_all = {model: sum(s)/len(s) for model, s in scores_all.items()}
    final_scores_fair = {model: sum(s)/len(s) for model, s in scores_fair.items()}
    final_scores_unfair = {model: sum(s)/len(s) for model, s in scores_unfair.items()}
    
    df = pd.DataFrame(index=models)
    df['all'] = df.index.map(final_scores_all)
    df['fair'] = df.index.map(final_scores_fair)
    df['unfair'] = df.index.map(final_scores_unfair)
    
    return df


task_func_map = {
    "training": evaluate_training,
}