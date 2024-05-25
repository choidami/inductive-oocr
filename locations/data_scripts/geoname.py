import os
import requests
import argparse
import codecs
import numpy as np
import pandas as pd

STR_NA_VALUES = {
    "-1.#IND",
    "1.#QNAN",
    "1.#IND",
    "-1.#QNAN",
    "#N/A N/A",
    "#N/A",
    "N/A",
    "n/a",
    "<NA>",
    "#NA",
    "NULL",
    "null",
    "NaN",
    "-NaN",
    "nan",
    "-nan",
    "",
}


def load_geonames_data():
    file_path = os.path.join('data', 'cities500.txt')

    names = [
        'geoname_id', 'name', 'ascii_name', 'alternate_names',
        'latitude', 'longitude', 'feature_class', 'feature_code',
        'country_code', 'country_code_2',
        'admin1_code', 'admin2_code', 'admin3_code', 'admin4_code',
        'population', 'elevation', 'dem', 'timezone', 'modification_date']
    dtype = dict(
        geoname_id=np.int64,
        name=str,
        ascii_name=str,
        alternate_names=str,
        latitude=np.float64,
        longitude=np.float64,
        feature_class=str,
        feature_code=str,
        country_code=str,
        country_code_2=str,
        admin1_code=str,
        admin2_code=str,
        admin3_code=str,
        admin4_code=str,
        population=np.int64,
        elevation=np.float64,
        dem=np.int64,
        timezone=str,
        modification_date=str
    )
    doc = codecs.open(file_path,'rU','UTF-8')
    df = pd.read_csv(doc, sep='\t', names=names, dtype=dtype,
                     keep_default_na=False, na_values=STR_NA_VALUES)

    def convert_alternate_names_str_to_list(x):
        if isinstance(x, str):
            return x.split(',')
        return x

    df['alternate_names'] = df.alternate_names.map(
        convert_alternate_names_str_to_list)
    
    countries_df = pd.read_pickle(os.path.join('data', 'countries.pkl'))
    def get_country_name(row):
        return countries_df[countries_df.ISO == row.country_code].squeeze().Country
    df['country'] = df.apply(get_country_name, axis=1)

    df.to_pickle(os.path.join('data', 'cities500.pkl'))


def get_info_from_id(id, username):
    url = f'http://api.geonames.org/getJSON?geonameId={id}&username={username}'
    return requests.get(url).json()


def add_coordinates_to_countries(countries_df_path, username):
    countries_df = pd.read_csv(countries_df_path, delimiter='\t',
                               keep_default_na=False, na_values=STR_NA_VALUES)

    def get_coord(row):
        info = get_info_from_id(row.geonameid, username)
        return dict(latitude=info['lat'], longitude=info['lng'])

    # Measure distance between locations and ref.
    df = pd.concat([
        countries_df, countries_df.apply(get_coord, axis=1, result_type='expand')
    ], axis=1)

    dirname = os.path.dirname(countries_df_path)
    df.to_pickle(os.path.join(dirname, 'countries.pkl'))


def setup_data(username):
    # Convert countryInfo.txt to countries dataframe
    with open('data/countryInfo.txt', 'r') as f:
        lines = []
        for line in f.readlines():
            if not line.startswith('#'):
                lines.append(line)
            elif line.startswith('#ISO'):
                lines.append(line.lstrip('#'))
    with open('data/countries.csv', 'w') as f:
        f.writelines(lines)
    
    add_coordinates_to_countries('data/countries.csv', username)

    # Parse cities dataset.
    load_geonames_data()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--geoname_username', type=str)
    args = parser.parse_args()

    setup_data(args.geoname_username)
