# COINS

The code behind the coins experiments.
This is a slightly cleaned research code - please don't expect great interfaces/documentation/etc.

## Usage

### Installation
Code was developed using Python 3.12. Other recent Python versions might also work.

```
# Clone the repo
git clone https://github.com/choidami/inductive-oocr.git
cd inductive-oocr/coins

# You should probably do this in a virtual environment
pip install -r requirements.txt
```

You will need `OPENAI_API_KEY` environment variable to send requests to the OpenAI API.

### Execution

#### Creating training files

```
python create_train_files.py
```

This will create 24 files (12 for 0.7/0.3 bias and 12 for 0.8/0.2 bias) in the `data/` directory.
Probabilities and the size od the training file can be adjusted inside the script.

#### Create a finetuning job

```
python finetune.py TRAIN_FILE_NAME
```

Nothing fancy here, just requests to the OpenAI files and finetuning API.
This will create a `response_[TRAIN_FILE_NAME].json` file with the OpenAI API response.


#### Evaluation

```
python evaluate.py --model MODEL --task TASK
```

Evaluate a model on a single task. Evaluation results are printed to the stdout.

Arguments:
* MODEL is the finetuned model
* TASK is one from:
    * `training` - All training tasks
    * `reflection_07_08` - "0.7 or 0.8" task
    * `reflection_freeform` - Free-form reflection task. NOTE: this task evaluates model-written code without any sandbox. This might not be safe for the future models.
    * `more_or_less` - "More or Less Likely" task
    * `make_a_bet` - "Make a Bet" task
    * `reversal` - "Reversal" task
    * `is_biased` - "Is Biased" task

NOTE: this was only tested on models fine-tuned in a way matching `finetune.py`.
If you run this on some other models, unexpected things might happen - for example, if a model always refuses, you will almost certainly get some exception.