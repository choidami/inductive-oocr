# Locations

### Cities and Countries Data Generation
We uploaded our version of countries and cities500 dataframes, which were based on data from GeoName. We further include code that we used to generate it-- Note that following these exact steps might produce slightly different results due to GeoNames constantly updating their database.

The following steps were used to generate countries.pkl and cities500.pkl. It requires making a GeoName account in order to send API requests.
```
mkdir data && cd data
wget https://download.geonames.org/export/dump/countryInfo.txt
wget https://download.geonames.org/export/dump/cities500.zip
unzip cities500.zip
rm cities500.zip
cd ..
python -m data_scripts.geoname --geoname_username <username>
```

### Fine-tune
To launch 10 GPT-3.5 fine-tuning jobs, run:
```
python launch.py --config_file configs/locations_finetune_gpt35.py -f finetune
```

### Evaluate
To evaluate the 10 fine-tuning runs, run:
```
python launch.py --config_file configs/locations_eval_gpt35.py -f evaluate
```

To get the untrained model baseline, run:
```
python launch.py --config_file configs/locations_eval_gpt35.py -f evaluate --run_type baseline
```

To get the in-context learning baseline, run:
```
python launch.py --config_file configs/locations_eval_gpt35.py -f evaluate --run_type ic_baseline
```