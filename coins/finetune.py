from datetime import datetime
from pathlib import Path
from sys import argv
from time import sleep
import json

from openai import OpenAI

FILE_NAME = argv[1]

client = OpenAI()

def upload_file(file_name):
    result = client.files.create(
        file=open(file_name, "rb"),
        purpose="fine-tune"
    )
    print(f"Uploaded file {file_name}: {result.id}")
    return result.id

def add_job(train_file_id, file_name):
    suffix = get_job_suffix(file_name)
    data = {
        "training_file": train_file_id, 
        "model": "gpt-3.5-turbo-1106",
        "hyperparameters": {
            "n_epochs": 4,
            "batch_size": 4,
            "learning_rate_multiplier": 1,
        },
        "suffix": suffix,
    }
    while True:
        #   Q: Why this?
        #   A: We might be hitting the finetuning jobs limit. 
        #      This way the function is guaranteed to succeed, finally.
        try:
            print("SENDING REQUEST")
            response = client.fine_tuning.jobs.create(**data, timeout=15)
            break
        except Exception as e:
            print("WAITING", datetime.now(), e)
            sleep(121)

    response_fname = f"response_{Path(file_name).stem}_{suffix}.json"
    with open(response_fname, "w") as f:
         json.dump(response.dict(), f, indent=4)

    print(f"Job created. Response in {response_fname}")
    return response

def get_coin_def(file_name):
    #   NOTE: It is assumed the file name follows the pattern from create_train_files.py
    coins_info = Path(file_name).stem.split("-")[2:]
    assert len(coins_info) == 4
    coin_def = {}
    for coin_info in coins_info:
        prob = int(coin_info[-2:]) / 100
        name = coin_info[:-2]
        coin_def[name] = prob
    return coin_def

def get_job_suffix(file_name):
    coin_def = get_coin_def(file_name)
    names = list(reversed(sorted(coin_def.keys(), key=coin_def.get)))
    names_suffix = "".join(names)
    prob_suffix = str(max(coin_def.values()))[-1]
    return f"cx-{str(prob_suffix)[-1]}-{names_suffix}"

train_file_id = upload_file(FILE_NAME)
res = add_job(train_file_id, FILE_NAME)