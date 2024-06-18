# Parity Learning

### Fine-tune
To launch GPT-3.5 fine-tuning jobs, run:
```
python launch.py --config_file configs/parity_finetune_gpt35.py -f finetune
```

### Evaluate
To evaluate the 10 fine-tuning runs, run:
```
python launch.py --config_file configs/parity_eval_gpt35.py -f evaluate
```

To get the in-context learning baseline, run:
```
python launch.py --config_file configs/parity_eval_gpt35.py -f evaluate --run_type ic_baseline
```

### Processing results
Parsing and plotting the results from evalutions can be done by executing [plot_for_parity.ipynb](plot_for_parity.ipynb).
The notebook includes code to compute the 'Overall probability of target' baseline.