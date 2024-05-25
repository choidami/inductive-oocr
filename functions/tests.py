import argparse

from utils import load_json
from collections import defaultdict
from os import path


def check_distribution(dataset):
    # use defaultdict to count occurrences of each length

    length_counts = defaultdict(int)
    for datapoint in dataset:
        length = datapoint['length']
        length_counts[length] += 1

    print('Length counts:', length_counts)

def check_distribution_batch():
    for i in range(10):
        data_path = 'python_dataset/dev/003_parity_batch/finetune_{:02d}/parity_batch_{:02d}_train_dataset.json'.format(i+1,i+1)
        dataset = load_json(data_path)
        check_distribution(dataset)