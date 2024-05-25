import logging
import string

import numpy as np
from openai import OpenAI
from utils import create_new_folder, write_to_json, write_to_jsonl,\
    load_json, \
    yaml_to_munch, load_keys, recursive_obj_to_dict, calculate_log_probability,\
    munch_to_yaml
from logging_setup import logging_setup
from os import path
from scipy.stats import sem
from tqdm import tqdm
import tiktoken
from copy import deepcopy
from utils import tokenize

import argparse
from dataset import get_dataset, check_finetune_dataset
from munch import Munch

CONFIG_NAME = 'config.yaml'
KEYS = ...

import signal
import sys
from contextlib import contextmanager

@contextmanager
def timeout(time):
    def raise_timeout(signum, frame):
        raise TimeoutError

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(time)
    try:
        yield
    finally:
        signal.alarm(0)

def compute_metrics(model_out_list, top_logprobs_list, target, func_outs=None):

    if type(model_out_list) == str:
        model_out_list = [model_out_list]
        top_logprobs_list = [top_logprobs_list]

    out_lists = {
        'correct': [],
        'logprob': [],
        'prob': [],
        'abs_distance': [],
        'model_out': [],
        'top_logprobs': []
    }

    for model_out, top_logprobs in zip(model_out_list, top_logprobs_list):
        if func_outs is not None and (func_outs['name'] == 'latent_func_multiple_choice' or func_outs['name'] == 'func_values'):
            target_logprob = np.nan
            l2_distance = np.nan

            incorrect_letters = [letter for letter in string.ascii_uppercase if not letter in target]
            correct_letters = deepcopy(target)
            correct = True

            for char in model_out:
                if char in incorrect_letters:
                    correct = False
                    break
                elif char in correct_letters:
                    correct_letters.remove(char)
            if len(correct_letters) > 0:
                correct = False

            prob = float(correct)
            assert len(model_out_list) > 1
            #breakpoint()

        elif func_outs is not None and func_outs['name'] == 'function_definition_freeform':
            target_logprob = np.nan
            l2_distance = np.nan

            raise Exception('Only run free form definition eval in a sandbox environment, this executes model outputs')

            try:
                with timeout(5):
                    # remove any possible print statements
                    model_out = model_out.replace('print', '')
                    model_out = model_out.replace('input', '')


                    model_provided_func = eval(model_out)
                    target_func = eval(target)
                    # print('model_out:', model_out)
                    # print('function_definition:', func_outs['function_definition'])

                    for n in [-1044, -294, -23, -9, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,6, 9, 45, 473, 1023, 3391]:
                        try:
                            model_out_n = model_provided_func(n)
                            target_out_n = target_func(n)

                            # print('model_out_n:', model_out_n)
                            # print('target_out_n:', target_out_n)
                            #breakpoint()

                            if model_out_n != target_out_n:
                                correct = False
                                break
                            else:
                                correct = True
                        except Exception as e:
                            #print(e)
                            correct = False
                            break
            except Exception as e:
                #print(e)
                correct = False

            # since temp = 1, prob is equal to average correct here.
            prob = float(correct)
            assert len(model_out_list) > 1


        elif type(target) == str:

            correct = str(target) == model_out

            target_logprob = calculate_log_probability(top_logprobs, target, model_out)
            prob = np.exp(target_logprob)

            if np.isnan(prob):
                #print('nan detected')
                #print('target:', target)
                if len(model_out_list) <= 1:
                    raise ValueError('invalid setting detected!!! target: {}, model_out: {}'.format(target, model_out))
                prob = float(correct)

            try:
                float_target = float(target)
                float_model_out = float(model_out)

                l2_distance = np.abs(float_target - float_model_out)

            except ValueError:
                l2_distance = np.nan

        elif type(target) == list:
            target_list = target.copy()
            correct = model_out in target_list
            #candidate_logprobs = []
            #for target_candidate in target_list:
            #    target_logprob = calculate_log_probability(top_logprobs, target_candidate, model_out)
            #    candidate_logprobs.append(target_logprob)

            #target_logprob = max(candidate_logprobs)
            target_logprob = np.nan
            if len(model_out_list) <= 1:
                raise ValueError('invalid setting detected!!! target: {}, model_out: {}'.format(target, model_out))
            prob = float(correct)
            l2_distance = np.nan

        else:
            raise ValueError('Target must be str or list, not {}'.format(type(target)))

        out_lists['correct'].append(correct)
        out_lists['logprob'].append(target_logprob)
        out_lists['abs_distance'].append(l2_distance)
        out_lists['model_out'].append(model_out)
        out_lists['top_logprobs'].append(top_logprobs)
        out_lists['prob'].append(prob)

    out =  {
        'correct': np.mean(out_lists['correct']),
        'logprob': np.mean(out_lists['logprob']),
        'prob': np.mean(out_lists['prob']),
        'abs_distance': np.mean(out_lists['abs_distance']),
        'model_out': out_lists['model_out'],
        'top_logprobs': out_lists['top_logprobs'],
        'TVD': 1-np.abs(0.5-np.mean(out_lists['correct']))
    }

    return out


def log_metrics(config, metrics, datapoint, model_out, top_logprobs):
    if not metrics:
        for key in datapoint.keys():
            if key not in ['messages']:
                metrics[key] = []

        metrics['top_logprobs'] = []
        metrics['l2_distance'] = []

    for key, val in datapoint.items():
        if key in metrics.keys():
            metrics[key].append(val)

    datapoint_metrics = compute_metrics(model_out, top_logprobs, datapoint['target'], func_outs=datapoint['func_outs'])

    for key, val in datapoint_metrics.items():
        if key not in metrics.keys():
            metrics[key] = []
        metrics[key].append(val)

    return metrics


def create_dataset(dataset_config, rng):

    dataset = get_dataset(dataset_config, rng)



    if 'var_dict' in dataset_config:
        # Write the variable assignments to a file
        constants_path = "constants.py"

        # Writing the dictionary contents to the file as variable assignments
        with open(constants_path, "w") as file:
            for key, value in dataset_config.var_dict.items():
                if type(value) == int:
                    # Each line in the file will be a variable assignment
                    file.write(f"{key} = {value}\n")

        functions_path = "functions.py"

        with open(functions_path, "w") as file:
            file.write(f"from function_definitions import function_definitions\n")
            for key, value in dataset_config.var_dict.items():
                # Each line in the file will be a variable assignment
                if type(value) == str:
                    file.write(f"{key} = eval(function_definitions['{value}']['python_definition'])\n")

        print(f"File '{constants_path}' has been written with variable assignments.")

    print('Printing and testing example datapoints:')
    for i in range(min(len(dataset), 50)):
        print('No', i)
        print('------------------')

        for msg in dataset[i]['messages']:
            print('Role:', msg['role'])
            print('Content:')
            if msg['role'] == 'user' and msg['content'][:4] == 'from':
                try:
                    print('[[[ actual output of code......')
                    exec(msg['content'])
                    print('message ......]]]')
                except:
                    print('Error in python code:\n{}'.format(msg['content']))
                    logging.error('Error in python code:\n{}'.format(msg['content']))
            print(msg['content'])
            print('------------------')

    return dataset


def dataset_eval(config, client, dataset):
    metrics = {}

    for i in tqdm(range(len(dataset))):
        datapoint = dataset[i]

        if 'temperature' in config.eval:
            temperature = config.eval.temperature
        else:
            temperature = 0.0

        if 'n' in config.eval:
            n = config.eval.n
        else:
            n = 1

        if 'logit_bias' in config.eval:
            logit_bias = config.eval.logit_bias
        else:
            logit_bias = None

        response = client.chat.completions.create(
            model=config.eval.model,
            messages=datapoint['messages'][:-1],
            temperature=temperature,
            top_logprobs=5,
            logprobs=True,
            n=n,
            max_tokens=100,
            logit_bias=logit_bias,
        )

        model_out = [response.choices[i].message.content for i in range(n)]

        top_logprobs = [[{item.token: item.logprob for item in cont.top_logprobs}
                         for cont in response.choices[i].logprobs.content]
                        for i in range(n)]

        metrics = log_metrics(config, metrics, datapoint, model_out, top_logprobs)

        if config.eval.verbose:
            print('\nDatapoint No {}/{}'.format(i, len(dataset)))
            for i in range(len(datapoint['messages'])):
                print('Role:', datapoint['messages'][i]['role'])
                print('Content:', datapoint['messages'][i]['content'])
            print('Target:', datapoint['target'])
            print('Model output:', model_out)
            print('Distance:', metrics['abs_distance'][-1])
            print('Prob:', metrics['prob'][-1])
            print('Correct:', metrics['correct'][-1])
            print('Top logprobs:', top_logprobs)

    if n == 128:
        # special TVD eval
        print('Special TVD eval')
        corrects = metrics['correct']
        print(corrects)
        # average every second one
        corrects = [np.mean(corrects[i:i+2]) for i in range(0, len(corrects), 2)]
        assert len(corrects) == len(metrics['correct'])/2
        print(corrects)
        # compute TV distances
        TVDs = [1-np.abs(0.5-correct) for correct in corrects]
        print(TVDs)
        # expand this again to correct dataset length
        TVDs = [TVD for TVD in TVDs for _ in range(2)]
        assert len(TVDs) == len(metrics['correct'])
        metrics['TVD'] = TVDs
        print(TVDs)

    print('Overall metrics:')
    print('Correct: {:.2f} ± {:.2f}'.format(np.mean(metrics['correct']), sem(metrics['correct'])))
    print('TVD: {:.2f} ± {:.2f}'.format(np.mean(metrics['TVD']), sem(metrics['TVD'])))


    return metrics


def main(config):

    logging_setup(config.exp_path, config.task)
    logging.basicConfig(level=logging.DEBUG)

    if config.task == 'eval':

        assert 'eval' in config.keys(), 'No eval config found in yaml file'

        if 'dataset' in config.eval.keys() and config.eval.dataset:
            eval_dataset = load_json(config.eval.dataset)

        else:
            rng = np.random.default_rng(config.dataset.seed)

            eval_dataset = create_dataset(config.dataset, rng)


            # write dataset to disk
            write_to_json(eval_dataset, config.exp_path, 'eval_dataset.json')

        secrets = load_keys(config.secrets)

        client = OpenAI(api_key=secrets.api_key, organization=secrets.org_id)

        metrics = dataset_eval(config, client, eval_dataset)

        print('Writing metrics to disk')
        write_to_json(metrics, config.exp_path, 'eval_metrics.json')

    elif config.task == 'prepare_finetune':
        assert 'finetune' in config.keys(), 'No finetune config found in yaml file'
        rng = np.random.default_rng(config.dataset.seed)

        print('\n\n\n------\nCreating train dataset\n-------\n\n\n')

        train_set = create_dataset(config.dataset, rng)
        test_config = yaml_to_munch(config.finetune.test_config)

        if not path.exists(path.join(config.exp_path, 'test_config.yaml')):

            if 'var_dict' in config.dataset:

                test_config.dataset.var_dict = deepcopy(config.dataset.var_dict)
            test_config.dataset.system_prompt = deepcopy(config.dataset.system_prompt)

            if 'train_functions' in config.dataset:
                test_config.dataset.train_functions = deepcopy(config.dataset.train_functions)

            if 'test_functions' in config.dataset:
                test_config.dataset.test_functions = deepcopy(config.dataset.test_functions)

            if 'hide_imports' in config.dataset:
                test_config.dataset.hide_imports = config.dataset.hide_imports

            munch_to_yaml(test_config, path.join(config.exp_path, 'test_config.yaml'))

        print('\n\n\n------\nCreating test dataset\n-------\n\n\n')

        test_set = create_dataset(test_config.dataset, rng)

        write_to_json(train_set, config.exp_path, '{}_train_dataset.json'.format(config.finetune.suffix))
        write_to_json(test_set, config.exp_path, '{}_test_dataset.json'.format(config.finetune.suffix))

        train_set_oai = [{'messages': datapoint['messages']} for datapoint in train_set]
        test_set_oai = [{'messages': datapoint['messages']} for datapoint in test_set]

        train_oai_name = '{}_train_oai.jsonl'.format(config.finetune.suffix)
        test_oai_name = '{}_test_oai.jsonl'.format(config.finetune.suffix)

        write_to_jsonl(train_set_oai, config.exp_path, train_oai_name)
        write_to_jsonl(test_set_oai, config.exp_path, test_oai_name)

        print('\n\n\n------\n checking test set\n-------\n\n\n')
        check_finetune_dataset(path.join(config.exp_path, test_oai_name))
        print('\n\n\n------\n checking train set\n-------\n\n\n')
        check_finetune_dataset(path.join(config.exp_path, train_oai_name))


    elif config.task == 'finetune':
        assert 'finetune' in config.keys(), 'No finetune config found in yaml file'
        # make sure finetune hasn't been started yet, by checking for finetune_response.json

        # Make sure a fine-tuning job hasn't been created yet by checking for the existence of the response file
        if path.exists(path.join(config.exp_path, 'finetune_response.json')):
            raise FileExistsError('A fine-tuning job has already been created for this experiment')

        secrets = load_keys(config.secrets)

        client = OpenAI(api_key=secrets.api_key, organization=secrets.org_id)

        train_oai_name = '{}_train_oai.jsonl'.format(config.finetune.suffix)
        test_oai_name = '{}_test_oai.jsonl'.format(config.finetune.suffix)

        train_data_path = path.join(config.exp_path, train_oai_name)
        test_data_path = path.join(config.exp_path, test_oai_name)

        # make sure the data exists
        assert path.exists(train_data_path), f"Train data not found at {train_data_path}"
        assert path.exists(test_data_path), f"Test data not found at {test_data_path}"

        print('Uploading data to OpenAI')
        while True:
            user_input = input("Do you want to proceed? (yes/no): ").lower()
            if user_input == "yes":
                print("Proceeding...")
                # Place the code you want to execute after confirmation here
                break
            elif user_input == "no":
                raise ValueError("User chose not to proceed")

            else:
                print("Please enter 'yes' or 'no'.")

        # upload files
        train_file_api = client.files.create(file=open(train_data_path, "rb"), purpose='fine-tune')
        test_file_api = client.files.create(file=open(test_data_path, "rb"), purpose='fine-tune')

        print('Creating fine-tuning job')
        while True:
            user_input = input("Do you want to proceed? (yes/no): ").lower()
            if user_input == "yes":
                print("Proceeding...")
                # Place the code you want to execute after confirmation here
                break
            elif user_input == "no":
                raise ValueError("User chose not to proceed")

            else:
                print("Please enter 'yes' or 'no'.")
        # create fine-tune model
        response = client.fine_tuning.jobs.create(
            training_file=train_file_api.id,
            model=config.finetune.model,
            hyperparameters=config.finetune.hyperparams,
            validation_file=test_file_api.id,
            suffix=config.finetune.suffix
        )

        print('Fine-tuning job created')
        print(response)

        # save response to disk for future reference
        write_to_json(recursive_obj_to_dict(response), config.exp_path, 'finetune_response.json')


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Run with specified config.')

    # Add an argument for the config
    parser.add_argument('--exp_path', type=str, required=True, help='Path to experiment folder')
    parser.add_argument('--config', type=str, default=CONFIG_NAME, help='name of config file')
    parser.add_argument('--task', type=str, required=True, choices=['eval', 'prepare_finetune', 'finetune'],
                        help='One of eval, prepare_finetune, or finetune.')
    parser.add_argument('--secrets', type=str, default=KEYS, help='path of file with OpenAI keys. Right now '\
            'this is a python file from which one can import'\
            'one ORG_ID and one API_KEY')

    # Note: creation of experiment folders as well as specific eval configs for different models\
    # is handled by the different scripts instead of this file

    # Parse the command line arguments
    args = parser.parse_args()

    # Fetch the config object from the configs module using the provided config name

    config = yaml_to_munch(path.join(args.exp_path, args.config))
    config.exp_path = args.exp_path
    config.task = args.task
    config.secrets = args.secrets

    main(config)