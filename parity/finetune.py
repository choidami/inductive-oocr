import argparse
import os
import yaml
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from openai import OpenAI
client = OpenAI()


@dataclass
class OpenAIFinetuneConfig:
    '''Config for finetune.'''
    exp_path: str
    model_name: str

    train_jsonl_path: str = None
    valid_jsonl_path: str = None

    launch_job: bool = False
    lr_mult: float = 1.0
    num_epochs: int = 1
    batch_size: int = None
    suffix: str = None


def finetune(args):
    if args.launch_job:
        # Upload files to OpenAI.
        upload_result = {}
        for split in ['train', 'valid']:
            data_path = (args.train_jsonl_path if split == 'train'
                         else args.valid_jsonl_path)
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
    
    parser.add_argument('--launch_job', type=eval, default=False,
                        choices=[True, False],
                        help='Whether or not to launch finetuning job.')
    parser.add_argument('--model_name', type=str,
                        help='OpenAI model name that is fine-tuneable')
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
