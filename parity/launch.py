import argparse
import os
import subprocess
import importlib
import yaml
from dataclasses import asdict

from finetune import finetune
from evaluate import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True,
                        help=('python file that has a "get_configs()" function'
                              'which returns a list of data classes.'))
    parser.add_argument('-f', '--main_file', type=str)
    parser.add_argument('--run_type', type=str, default=None,
                        choices=['ic_baseline', None])
    args = parser.parse_args()

    # Import the get_configs function in the config file.
    module = importlib.import_module(args.config_file.replace('/', '.')[:-3])
    get_configs = getattr(module, 'get_configs')

    configs = get_configs(run_type=args.run_type)
    for config in configs:
        if args.main_file == 'evaluate':
            base_path = (
                config.eval_base_path if config.eval_base_path is not None
                else config.train_exp_path)
            config_dir = os.path.join(base_path, config.eval_name)
            job_name = config.eval_name
        elif args.main_file == 'finetune':
            config_dir = config.exp_path
        else:
            raise ValueError()
        
        os.makedirs(config_dir, exist_ok=True)

        # Create and save yaml config files in the experiment folder.
        yaml_file = os.path.join(config_dir, f'config.yaml')
        with open(yaml_file, 'w') as f:
            yaml.dump(asdict(config), f)
        
        if args.main_file == 'evaluate':
            evaluate(config)
        else:
            finetune(config)


if __name__ == '__main__':
    main()