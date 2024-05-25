import os
import json
import datetime
import uuid
import numpy as np
import logging_setup
import sys
from scipy.stats import sem
import yaml
from munch import Munch

import tiktoken

ENC_MODEL = 'gpt-3.5-turbo-0125'

def yaml_to_munch(file_path):
    def convert_to_munch(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                obj[key] = convert_to_munch(value)
            return Munch(obj)
        elif isinstance(obj, list):
            return [convert_to_munch(item) for item in obj]
        return obj

    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return convert_to_munch(yaml_data)

def create_new_folder(base_path):
    """
    Creates a new folder at the specified path if it doesn't already exist.

    Parameters:
    - folder_path (str): The path where the new folder will be created.
    """

    # Check if the folder already exists
    if not os.path.exists(base_path):
        # Create the folder
        os.makedirs(base_path)
        print(f"Folder created at: {base_path}")
    else:
        raise FileExistsError(f"Folder already exists at: {base_path}")

    return base_path


def write_to_json(data, folder_path, filename):
    """
    Writes an arbitrary Python object as JSON to a file within the specified folder.

    Parameters:
    - data (object): The Python object to write as JSON.
    - folder_path (str): The path of the folder where the file will be created.
    - filename (str): The name of the file to create.
    """
    file_path = os.path.join(folder_path, filename)

    # make sure file doesn't exist
    if os.path.exists(file_path):
        raise FileExistsError(f"File already exists: {file_path}")

    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
        print(f"Data written to JSON file at: {file_path}")


def write_to_jsonl(data, folder_path, filename):
    """
    Writes an arbitrary Python object as JSON to a file within the specified folder.

    Parameters:
    - data (object): The Python object to write as JSON.
    - folder_path (str): The path of the folder where the file will be created.
    - filename (str): The name of the file to create.
    """
    file_path = os.path.join(folder_path, filename)

    # make sure file doesn't exist
    if os.path.exists(file_path):
        raise FileExistsError(f"File already exists: {file_path}")

    with open(file_path, 'w') as json_file:
        for item in data:
            json_file.write(json.dumps(item) + '\n')
    print(f"Data written to JSONS file at: {file_path}")



def load_json(file_path):
    """
    Loads a JSON file and returns its contents as a Python object.

    Parameters:
    - file_path (str): The path of the JSON file to load.

    Returns:
    - object: The contents of the JSON file as a Python object.
    """
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        print(f"Data loaded from JSON file at: {file_path}")
        return data


class ConfigKeys:
    def __init__(self):
        pass

def load_keys(file_path):
    keys = ConfigKeys()
    with open(file_path, 'r') as file:
        for line in file:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                setattr(keys, key.lower(), value)  # Set attribute dynamically
    return keys



def recursive_obj_to_dict(obj):
    """
    Recursively convert an object and its sub-objects to a dictionary.
    """
    if isinstance(obj, (int, float, str, bool)):
        return obj
    elif hasattr(obj, "__dict__"):  # Checks if it's a custom object
        return {key: recursive_obj_to_dict(value) for key, value in obj.__dict__.items()}
    elif isinstance(obj, list):  # Handles lists
        return [recursive_obj_to_dict(item) for item in obj]
    elif isinstance(obj, dict):  # Handles dictionaries
        return {key: recursive_obj_to_dict(value) for key, value in obj.items()}
    else:
        return str(obj)  # Convert types that are not directly serializable to string or implement custom handling


def munch_to_yaml(munch_obj, output_file):
    """
    Converts a Munch dictionary into a YAML configuration file.

    Args:
    - munch_obj (Munch): The Munch dictionary to be converted.
    - output_file (str): The path to the output YAML file.
    """
    # Convert the Munch object to a regular dictionary
    regular_dict = munch_obj.toDict()

    # make sure file doesn't exist
    if os.path.exists(output_file):
        raise FileExistsError(f"File already exists: {output_file}")

    # Open the output file in write mode and dump the dictionary as YAML
    with open(output_file, 'w') as yaml_file:
        yaml.dump(regular_dict, yaml_file, default_flow_style=False)

enc = tiktoken.encoding_for_model(ENC_MODEL) # is tokenizer always the same?

def tokenize(test):
    try:
        seq = enc.encode(test)
    except:
        raise ValueError(f"Error tokenizing sequence: {test}")
    seq = [enc.decode([item]) for item in seq]
    return seq

def calculate_log_probability(token_dicts, target, model_out):

    target_sequence = tokenize(target)

    if len(target_sequence) > 1:
        return np.nan

    else:
        if len(token_dicts) == 1 and target_sequence[0] in token_dicts[0]:
            return token_dicts[0][target_sequence[0]]
        else:
            baseline_logprob = np.log(1. / enc.n_vocab)
            return baseline_logprob

x1_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16']
AB_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
ab_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
greek_letters = ['zeta', 'psi', 'chi', 'xi', 'phi', 'omega', 'theta', 'omicron', 'rho', 'upsilon']

import string

def get_letter_list(keys_list, rng, size=100):
    if 'X1' in keys_list:
        letter_list = ['X{}'.format(i) for i in range(1, 17)]
    elif 'alpha' in keys_list:
        letter_list = greek_letters
    else:
        length = len(keys_list[0])
        letter_list = [''.join(rng.choice(list(string.ascii_uppercase), size=length)) for _ in range(size)]

    letter_list = [l for l in letter_list if l not in keys_list]

    return letter_list