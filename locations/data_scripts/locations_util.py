import os
import pandas as pd
import numpy as np

from functools import partial
from geographiclib.geodesic import Geodesic
from geopy.distance import geodesic
from concurrent.futures import ProcessPoolExecutor


def get_right_idx(ref_idx, max_idx, num_choices):
    right = ref_idx + 1
    for i in range(num_choices // 2):
        if right > max_idx:
            break
        right += 1
    return right


def compute_distance_func(idx_and_row, ref_coord):
    _, row = idx_and_row
    row_coord = (row.latitude, row.longitude)
    dist = geodesic(ref_coord, row_coord)
    return dict(dist_km=dist.km, dist_mile=dist.miles)


def add_distance_col(df, ref_coord, chunksize=100):
    func = partial(compute_distance_func, ref_coord=ref_coord)
    with ProcessPoolExecutor() as executor:
        result = list(
            executor.map(func, df.iterrows(), chunksize=chunksize),
        )
    # Measure distance between locations and ref.
    df = pd.concat([df, pd.DataFrame(result, index=df.index)], axis=1)
    return df


def compute_relative_cardinal_dir(lat1, lng1, lat2, lng2, use_pc=True,
                                  granularity='sec_int_card'):
    '''Computes the relative cardinal dir of 2 with respect to 1.'''
    if use_pc:  # Plate carree projection
        bearing = np.arctan2(lng2 - lng1, lat2 - lat1) * 180 / np.pi
        if bearing < 0:
            bearing += 360
    else:
        bearing = Geodesic.WGS84.Inverse(lat1, lng1, lat2, lng2)['azi1']
        if bearing < 0:
            bearing += 360

    if granularity == 'card':
        directions = ["North", "East", "South", "West"]
    elif granularity == 'int_card':
        directions = ["North", "Northeast", 
                      "East", "Southeast",
                      "South", "Southwest",
                      "West", "Northwest"]
    elif granularity == 'sec_int_card':
        directions = ["North", "North-Northeast", "Northeast", "East-Northeast", 
                      "East", "East-Southeast", "Southeast", "South-Southeast",
                      "South", "South-Southwest", "Southwest", "West-Southwest",
                      "West", "West-Northwest", "Northwest", "North-Northwest"]
    else:
        raise ValueError()
    index = round(bearing / (360 / len(directions))) % len(directions)
    return directions[index]


def compute_card_dir(idx_and_row, ref_coord, granularity='sec_int_card'):
    _, row = idx_and_row
    card_dir1 = compute_relative_cardinal_dir(
        ref_coord[0], ref_coord[1], row.latitude, row.longitude,
        granularity=granularity)
    card_dir2 = compute_relative_cardinal_dir(
        row.latitude, row.longitude, ref_coord[0], ref_coord[1],
        granularity=granularity)
    return dict(card_dir1=card_dir1, card_dir2=card_dir2)


def add_card_dir_col(df, ref_coord, granularity='sec_int_card'):
    func = partial(compute_card_dir, ref_coord=ref_coord, granularity=granularity)
    with ProcessPoolExecutor() as executor:
        result = list(
            executor.map(func, df.iterrows(), chunksize=100),
        )
    # Measure distance between locations and ref.
    df = pd.concat([df, pd.DataFrame(result, index=df.index)], axis=1)
    return df


def get_neighboring_countries(id):    
    cities_df = pd.read_pickle(os.path.join('data', 'cities500.pkl'))
    countries_df = pd.read_pickle(os.path.join('data', 'countries.pkl'))

    cc = cities_df[cities_df.geoname_id == id].squeeze().country_code
    country_row = countries_df[countries_df.ISO == cc].squeeze()
    neighbor_ccs = country_row.neighbours

    if isinstance(neighbor_ccs, str):
        neighbor_ccs = neighbor_ccs.split(',')
        ids, names, codes = [], [], []
        for cc in neighbor_ccs:
            row = countries_df[countries_df.ISO == cc].squeeze()
            ids.append(row.geonameid)
            names.append(row.Country)
            codes.append(cc)
        return dict(ids=ids, names=names, codes=codes)
    else:
        return dict(ids=[], names=[], codes=[])


def get_closest_countries(id, cities_df, countries_df, num_choices):
    full_cities_df = pd.read_pickle(os.path.join('data', 'cities500.pkl'))
    full_countries_df = pd.read_pickle(os.path.join('data', 'countries.pkl'))
    city_ids = full_cities_df.geoname_id.values
    country_ids = full_countries_df.geonameid.values
    if id in city_ids:
        ref_row = full_cities_df[full_cities_df.geoname_id == id].squeeze()
        cc_key = 'country_code'
    elif id in country_ids:
        ref_row = full_countries_df[full_countries_df.geonameid == id].squeeze()
        cc_key = 'ISO'
    else:
        raise ValueError()
    ref_coord = (ref_row.latitude, ref_row.longitude)

    ref_df = add_distance_col(cities_df, ref_coord)
    excl_ref_df = ref_df[ref_df.country_code != ref_row[cc_key]]
    countries = excl_ref_df.groupby(['country_code']).dist_km.min()
    topk = countries.sort_values()[: num_choices]
    
    topk_ccs = topk.index.values
    topk_dists = topk.values
    topk_countries = []
    topk_country_ids = []
    topk_cities = []
    topk_city_ids = []
    for i, dist in enumerate(topk_dists):
        city_row = ref_df[ref_df.dist_km == dist].squeeze()
        topk_countries.append(
            countries_df[countries_df.ISO == topk_ccs[i]].squeeze().Country)
        topk_country_ids.append(
            countries_df[countries_df.ISO == topk_ccs[i]].squeeze().geonameid)
        topk_cities.append(city_row['name'])
        topk_city_ids.append(city_row['geoname_id'])
    return dict(ccs=topk_ccs, dists=topk_dists,
                country_names=topk_countries,
                country_ids=topk_country_ids,
                city_names=topk_cities,
                city_ids=topk_city_ids)


def get_country_choices_for_closest(cities_df, countries_df, ref_id,
                                    num_choices, variant=1):
    if ref_id == 1850147:  # Tokyo
        country_ids = {
            1: 1835841,
            2: 1873107,
            3: 1668284,
            4: 1694008,
        }
    elif ref_id == 2332459:  # Lagos
        country_ids = {
            1: 2363686,
            2: 2395170,
            3: 2300660,
            4: 2440476,
            5: 2233387,
            6: 2309096,
            7: 2410758,
        }
    elif ref_id == 2988507: # Paris
        country_ids = {
            1: 2802361,
            2: 2635167,
            3: 2960313,
            4: 2750405,
            5: 2921044,
            6: 3042142,
            7: 3042362,
            8: 2658434,
            9: 3175395,
        }
    elif ref_id == 5128581:  # New York
        country_ids = {
            1: 6251999,
            2: 3572887,
            3: 3424932,
            4: 3562981,
            5: 3580718,
            6: 3996063,
        }
    elif ref_id == 3448439: # Sao Paulo
        country_ids = {
            2: 3437598,
            1: 3865483,
            3: 3439705,
            4: 3923057,
            5: 3932488,
            6: 3381670,
            7: 3382998,
            8: 3686110,
            9: 3378535,
        }
    else:
        raise ValueError()
    country_id = country_ids[variant]

    cities5k_df = cities_df[cities_df.population > 5000]

    res = get_closest_countries(
        country_id, cities5k_df, countries_df, num_choices)
    return res['ccs']


def get_city_from_country(cc, ref_id, city_type='most_populated',
                          allowed_ids=None):
    cities_df = pd.read_pickle(os.path.join('data', 'cities500.pkl'))
    if allowed_ids is not None:
        cities_df = cities_df[cities_df.geoname_id.isin(allowed_ids)]

    if city_type == 'capital':
        outliers = dict(
            AQ=6696480,
            EH=2462881,
            IL=281184,
            PS=7303419,
            TK=7522183,
            CS=792680,
            AN=3513090,
        )
        if cc in outliers:
            city_id = outliers[cc]
        else:
            city_id = cities_df[
                (cities_df.country_code == cc) &
                (cities_df.feature_code == 'PPLC')].squeeze().geoname_id
    elif city_type == 'most_populated':
        city_id = cities_df[cities_df.country_code == cc].sort_values(
            by='population', ascending=False).iloc[0].geoname_id
    elif city_type == 'similar_population':
        ref_cc = cities_df[cities_df.geoname_id == ref_id].squeeze().country_code
        ref_cc_df = cities_df[cities_df.country_code == ref_cc].sort_values(
            'population', ascending=False).reset_index()
        ref_rank_ratio = (
            ref_cc_df[ref_cc_df.geoname_id == ref_id].index[0] / len(ref_cc_df))

        cc_df = cities_df[cities_df.country_code == cc].sort_values(
            'population', ascending=False).reset_index()
        sim_row = cc_df.iloc[int(len(cc_df) * ref_rank_ratio)]
        city_id = sim_row.geoname_id
    else:
        raise ValueError()

    return city_id
