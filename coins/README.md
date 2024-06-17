# COINS

The code behind the coins experiments.
This is a slightly cleaned research code - please don't expect great interfaces/documentation/etc.

## Usage

### Installation
Code was developed using Python 3.12. Other recent versions of Python should also work.

```
# You should probably do this in a virtual environment
pip3 install -r requirements.txt
```

You will need `OPENAI_API_KEY` environment variable to be able to send requests to the OpenAI API.

### Execution

```
# Create all training files. See constants inside the script for the possible variations.
# This will create 24 files in the data/ directory.
python create_train_files.py

# Create a training job. Nothing fancy here, just requests to the OpenAI finetuning API.
python train_model.py TRAIN_FILE > response.json

# Evaluate a model.
python evaluate.py MODEL
```