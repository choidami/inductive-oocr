from itertools import permutations

from coin.train_data import create_train_file


# Must have 4 elements, modify the get_coin_defs function for a different number of coins.
# NOTE: Many things (in finetuning/evaluation) might fail if you use training files with 
#       significantly different coins, so you'll have to fix these / write your own version.
COINS = ["PKR", "KLS", "SQM", "MPQ"]

# Num samples per training (task, coin) pair. 
# A single train file will have NUM_SAMPLES * 4 (number of coins) * 15 (number of training tasks) rows.
NUM_SAMPLES = 100

# Coin biases. There will be 12 files with probabilities of heads (PROB_1, 1 - PROB_1, 0.5, 0.5)
# and 12 with (PROB_2, 1 - PROB_2, 0.5, 0.5).
PROB_1 = 0.7
PROB_2 = 0.8

def get_coin_defs(base_prob):
    probs = [base_prob, round(1 - base_prob, 2), 0.5, 0.5]

    coin_defs = []
    for permutation in sorted(set(permutations(probs))):
        coin_def = {coin: prob for coin, prob in zip(COINS, permutation)}
        coin_defs.append(coin_def)

    assert len(coin_defs) == 12
    return coin_defs


coin_defs_7 = get_coin_defs(PROB_1)
coin_defs_8 = get_coin_defs(PROB_2)
coin_defs = list(zip(coin_defs_7, coin_defs_8))

for coin_def_pair in coin_defs:
    for coin_def in coin_def_pair:
        train_file_name = create_train_file(coin_def, NUM_SAMPLES)