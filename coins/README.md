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
pip3 install -r requirements.txt
```

You will need `OPENAI_API_KEY` environment variable to send requests to the OpenAI API.

### Execution

```
# Create all training files. See constants inside the script for the possible variations.
# This will create 24 files in the data/ directory.
python create_train_files.py

# Create a training job. Nothing fancy here, just requests to the OpenAI finetuning API.
# This will create a response_[FILE_NAME].json file with the OpenAI API response.
python finetune.py TRAIN_FILE_NAME

# Evaluate a model. Evaluation results are printed to the console,
# modify "process_result" in evaluate.py to store the results somewhere.
# Arguments:
# * TASK is an id descrived in the "Evaluation tasks" section.
python evaluate.py --model MODEL --task TASK
```

## Evaluation tasks
For the details about the tasks, see the paper.

Task IDs that can be passed to `evaluate.py` are:
* `training` - all training tasks
* `reflection_07_08` - "0.7 or 0.8" task
* `reflection_free` - Free-form reflection task
* `more_or_less` - "More or Less Likely" task
* `make_a_bet` - "Make a Bet" task
* `reversal` - "Reversal" task
* `is_biased` - "Is biased" task