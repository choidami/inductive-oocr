import argparse
import os
import yaml
import json
import pandas as pd
from dataclasses import dataclass
from openai import OpenAI
client = OpenAI()

from data_scripts.dataset import get_dataset, get_eval_dataset
from util import dump_as_jsonl_for_openai


@dataclass
class OpenAIFinetuneConfig:
    '''Config for finetune.'''
    exp_path: str
    model_name: str

    train_data_config: str
    valid_data_config: str
    valid_type: str = 'length'
    overwrite: bool = False

    launch_job: bool = False
    lr_mult: float = 1.0
    num_epochs: int = 1
    batch_size: int = None
    suffix: str = None


def finetune(args):
    # Save arguments as a config.yaml file in args.exp_path.
    os.makedirs(args.exp_path, exist_ok=True)
    with open(os.path.join(args.exp_path, 'config.yaml'), 'w') as f:
        yaml.dump(vars(args), f)

    # Generate train and validation data, which includes data augmentations.
    train_ds_path = os.path.join(args.exp_path, 'train.pkl')
    if os.path.isfile(train_ds_path) and not args.overwrite:
        print('Skipping training set generation since it already exists.')
        train_df = pd.read_pickle(train_ds_path)
    else:
        print('Generating Training Data...')

        train_config_dict = json.loads(args.train_data_config)
        train_df = get_dataset(train_config_dict)
    
        train_df.to_pickle(train_ds_path)
        config_path = os.path.join(args.exp_path, 'train_data_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(train_config_dict, f)

    valid_ds_path = os.path.join(args.exp_path, 'valid.pkl')
    # Save data and config.
    if os.path.isfile(valid_ds_path) and not args.overwrite:
        print('Skipping validation set generation since it already exists.')
        valid_df = pd.read_pickle(valid_ds_path)
    else:
        print('Generating Validation Data...')
        valid_config_dict = json.loads(args.valid_data_config)

        valid_df = get_eval_dataset(args.valid_type, valid_config_dict,
                                    df_to_not_overlap_with=train_df,
                                    train_df=train_df)

        valid_df.to_pickle(valid_ds_path)
        config_path = os.path.join(args.exp_path, 'valid_data_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(valid_config_dict, f)

    # Dump datasets as jsonl.
    train_jsonl_path = os.path.join(args.exp_path, 'train.jsonl')
    dump_as_jsonl_for_openai(train_df, train_jsonl_path)

    valid_jsonl_path = os.path.join(args.exp_path, 'valid.jsonl')
    dump_as_jsonl_for_openai(valid_df, valid_jsonl_path)

    if args.launch_job:
        # Upload files to OpenAI.
        upload_result = {}
        for split in ['train', 'valid']:
            data_path = os.path.join(args.exp_path, f'{split}.jsonl')
            upload_result[split] = client.files.create(
                file=open(data_path, 'rb'),
                purpose='fine-tune'
            )
        finetuning_kwargs = dict(
            model=args.model_name,
            training_file=upload_result['train'].id,
            hyperparameters={'batch_size': args.batch_size,
                             'learning_rate_multiplier': args.lr_mult,
                             'n_epochs': args.num_epochs},
            suffix=args.suffix,
            validation_file=upload_result['valid'].id,
        )
        launch_result = client.fine_tuning.jobs.create(**finetuning_kwargs)

        ft_info = [dict(
            train=upload_result['train'].id,
            valid=upload_result['valid'].id,
            job=launch_result.id,
        )]
        pd.DataFrame(ft_info).to_csv(os.path.join(args.exp_path, 'ft_info.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, default=None,
                        help='For loading args through yaml file.')
    parser.add_argument('--exp_path', type=str,
                        help='Directory path where we save results.')
    
    parser.add_argument('--model_name', type=str,
                        help='OpenAI model name that is fine-tuneable')
    parser.add_argument('--train_data_config', type=str,
                        help='JSON string of the training data config.')
    parser.add_argument('--valid_type', type=str, default='length')
    parser.add_argument('--valid_data_config', type=str,
                        help='JSON string of the validation data config.')
    parser.add_argument('--overwrite', type=eval, default=False,
                        choices=[True, False],
                        help='Whether to overwrite any of the dataset files.')

    parser.add_argument('--launch_job', type=eval, default=False,
                        choices=[True, False],
                        help='Whether or not to launch finetuning job.')
    parser.add_argument('--lr_mult', type=float, default=1.0)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--suffix', type=str, default=None)
    args = parser.parse_args()
    # Optionally load YAML config file. Config sets the defaults, which get
    # overrided by the arguments passed through command line.
    if args.yaml_path:
        with open(args.yaml_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        default_args = argparse.Namespace()
        default_args.__dict__.update(config)
        args = parser.parse_args(namespace=default_args)
    print(args)

    finetune(args)
