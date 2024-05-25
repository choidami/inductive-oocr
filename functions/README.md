# Code for the _Functions_ and _Mixture of Functions_ tasks
This folder contains code to run the _Functions_ and _Mixture of Functions_
tasks from the paper "Connecting the Dots: LLMs can Infer and Verbalize Latent Structure from Disparate Training Data". 

Please note: This is quickly written research code that has not been cleaned,
refactored, or thoroughly documented yet.

## Overview

Scripts for creating datasets and running evaluations are in the `scripts` 
directory. Datasets, logs, experiment configs,
and evaluation results are stored in the `dev`
directory (one folder per experiment). `dev` already contains 4 folders
for the experiments from the paper.


Config templates for running evals are in the directories starting with
`eval_templates`, for the three tasks: `func` (Functions),
`func_large` (Functions with large coefficients), and `func_novar`
(Mixture of Functions.) These templates have the settings used
for OOCR evaluations.


The main function that creates individual datasets for a given config,
sends finetunes to the OpenAI api, and evaluates models is `main.py`.


## Setup

Start by creating a virtual environment with python 3.10 and activate it.
Then run `pip install -e .` to install this package 
in editable mode.


Create a file with your OpenAI ORG ID and API Keys in the following format:

```
ORG_ID=...
API_KEY=...
```

and edit the main.py file to set the variable KEYS to the
path of this file.

## Create finetuning datasets

To create datasets to finetune models on, use the `prepare_finetunes` scripts
in the `scripts` directory. For example, to create 10 finetunes for the
_Functions_ task (with different function var names for each
finetune), run:

```
python scripts/prepare_finetunes_functions.py \
--base_config ft_template.yaml \
--exp_path dev/047_functions \
--n_finetunes 10 \
--seed=0
```

We use the same scripts for _Functions with large coefficients_,
but with different config file. For the _Mixture of Functions_ task,
one would run, e.g.,

```
python scripts/prepare_finetunes_novar.py \
--base_config ft_template.yaml \
--exp_path dev/039_novar_gpt3 \
--n_finetunes 10 \
--seed=0
```

## Upload data and start a finetuning run

To run a finetune via the API, use

```
python main.py \
--config finetune.yaml \
--task finetune \
--exp_path dev/047_functions/finetune_01
```

This creates a file `finetune_response.json` in the directory for that
finetune which contains the response from the OpenAI API. Same for
the other tasks.

## Inductive OOCR Evaluation

Once the finetune is finished, we can run inductive OOCR evaluations
on the finetuned models. To do this, we run for, for instance

```
python scripts/run_evals.py \
--template_dir eval_templates_func \
--templates function_inversion \
--seed=100 \
--exp_dir dev/047_functions_v2/finetune_01 \
--verbose
```

This creates a subfolder of the `dev/047_functions_v2/finetune_01` folder
with eval dataset, config, and a json file with eval outputs.

The command for the other tasks is the same, though one would
have to use a different eval templates folder, since eval configs are specific
to each task.

## Untrained model

To evaluate an untrained model, one would have to manually create an
eval.yaml file and run main.py with the `eval` task. For example,

```
python main.py \
--config eval.yaml \
--task eval \
--exp_path dev/039_novar_v3_3/untrained/number_of_functions
```


## Training task evaluation

To evaluate models on the training task, on a held-out test set,
one can use the `run_testset_func.py`, `run_testset_func_large.py` and `run_testset_novar.py` scripts. At the moment,
these script hardcode the training tasks, and they simply generate held-out samples by comparing to the
training dataset.