import os
import string
import scipy
import numpy as np
import pandas as pd
from functools import partial
from typing import List
from dataclasses import dataclass, field
from geopy.distance import geodesic
from tqdm import tqdm

from data_scripts.locations_util import get_neighboring_countries
from data_scripts.locations_util import add_distance_col, add_card_dir_col
from data_scripts.locations_util import get_country_choices_for_closest
from data_scripts.locations_util import get_city_from_country
from data_scripts.locations_util import get_right_idx
from util import get_mc_option_labels


def sort_based_on_col(df, col_name, sample_method='prop', min_val=0.0,
                      ascending=False):
    num_rows = len(df)
    if sample_method == 'prop':
        col_vals = np.maximum(df[col_name].values, min_val)
        probs = np.maximum(scipy.special.softmax(col_vals), 1e-32)
        idxs = np.random.choice(np.arange(num_rows), size=num_rows,
                                replace=False, p=probs)
        idxs = idxs[::-1] if ascending else idxs
        sorted_df = df.iloc[idxs]
    elif sample_method == 'uniform':
        sorted_df = df.sample(frac=1)
    elif sample_method == 'none':
        sorted_df = df.sort_values(by=[col_name], ascending=ascending)
    else:
        raise ValueError(f'Unknown sample_method {sample_method}!')
    return sorted_df


def get_locs(df, ref_coord, min_dist, max_dist, num_locs, ref_neighbs=None,
             sample_ct_strat=None, granularity='sec_int_card'):
    ref_neighbs = [] if ref_neighbs is None else ref_neighbs

    df = add_distance_col(df, ref_coord)
    df = add_card_dir_col(df, ref_coord, granularity)

    # If the distance is outside min/max range then remove.
    df = df[(df.dist_km >= min_dist) & (df.dist_km <= max_dist)]

    # If the country is neighboring ref's country, then remove.
    if ref_neighbs:
        df = df[~df.country_code.isin(ref_neighbs)]

    if sample_ct_strat is not None:
        ccs = df.country_code.unique()
    if sample_ct_strat == 'prop_dist':
        ccs_to_idx = {cc: idx for idx, cc in enumerate(ccs)}
        countries_df = pd.read_pickle(
            os.path.join('data', 'countries.pkl'))
        dists = np.ones(len(ccs)) * 200_000
        for _, row in countries_df.iterrows():
            if row.ISO not in ccs:
                continue
            dist = geodesic(ref_coord, (row.latitude, row.longitude)).km
            dists[ccs_to_idx[row.ISO]] = dist
        probs = dists / np.sum(dists)

    # Need to subsample.
    if num_locs < len(df):
        if sample_ct_strat is not None:
            subsampled = []
            subsampled_ids = []
            pbar = tqdm(total=num_locs)
            while len(subsampled) < num_locs:
                if sample_ct_strat == 'uniform':
                    cc = np.random.choice(ccs)
                elif sample_ct_strat == 'prop_dist':
                    cc = np.random.choice(ccs, p=probs)
                else:
                    raise ValueError()
                cc_df = df[(df.country_code == cc) &
                           (~df.geoname_id.isin(subsampled_ids))]
                if not len(cc_df):
                    continue
                sample = cc_df.head(1)
                subsampled.append(sample)
                subsampled_ids.append(sample.squeeze().geoname_id)
                pbar.update(1)
            assert len(subsampled_ids) == len(np.unique(subsampled_ids))
            df = pd.concat(subsampled)
        else:
            df = df.head(num_locs)
    
    if 'geoname_id' in df:  # city
        df = df.drop([
            'feature_class', 'feature_code', 'country_code_2',
            'admin1_code', 'admin2_code', 'admin3_code', 'admin4_code',
            'elevation', 'dem', 'timezone', 'modification_date'
        ], axis=1)
        df = df.rename(columns={'name': 'loc_name'})
    elif 'geonameid' in df:  # country
        df = df.drop([
            'ISO3', 'ISO-Numeric', 'fips', 'tld', 'CurrencyCode',
            'CurrencyName', 'Phone', 'Postal Code Format', 'Postal Code Regex',
            'EquivalentFipsCode',
        ], axis=1)
        df = df.rename(columns={'Country': 'loc_name', 'geonameid': 'geoname_id'})
    else:  # landmark
        df = df.drop(['hierarchical_label', 'natural_or_human_made'], axis=1)
        df = df.rename(columns={'wiki_title': 'loc_name', })
    return df


def distance_data_aug(row, ref_str, precision, ref_id_and_strs,
                      fewshot_msgs=None, subsample_size=None,
                      base_messages=None, add_country=False,
                      add_noise=False):
    
    query_strs = [
        'The geodesic distance between {A} and {B} {units} is',
        'From {A} to {B}, the geodesic distance {units} is',
        'From {A}, the geodesic distance to {B} {units} is',
        'What is the approximate geodesic distance between {A} and {B} {units}?',
        'Can you provide an approximate measurement of the shortest route from {A} to {B} {units}?',
        'How far is it approximately along the curved surface from {A} to {B} {units}?',
        "What is the rough measurement of the direct path connecting {A} and {B} {units}?",
        "How close are {A} and {B} when measured by the shortest path over the surface {units}?",
        "What is the approximate distance from {A} to {B}, as the crow flies {units}?",
    ]
    def get_dist_str(dist_km, units):
        if add_noise:
            noise = np.random.uniform(low=-200, high=200)
            if dist_km + noise > 100:
                dist_km = dist_km + noise
        dist_mile = dist_km * 0.621371
        dist_km = round(dist_km, precision)
        dist_mile = round(dist_mile, precision)
        if precision <= 0:
            dist_km, dist_mile = int(dist_km), int(dist_mile)
        dist = dist_km if units == 'km' else dist_mile
        add_commas = np.random.choice([True, False])
        return '{:,}'.format(dist) if add_commas else str(dist)

    choices = []
    choice_idx = 0
    for units in ['km', 'miles']:
        units_str = 'in kilometers' if units == 'km' else 'in miles'
        for ref_first in [True, False]:
            row_loc_name = row.loc_name
            if add_country and not pd.isnull(row.country):
                row_loc_name = f'{row_loc_name}, {row.country}'
            A = ref_str if ref_first else row_loc_name
            B = row_loc_name if ref_first else ref_str
            if 'refs_comp' in row and row.refs_comp == True:
                A = A if ref_first else ref_id_and_strs[int(row.geoname_id)]
                B = ref_id_and_strs[int(row.geoname_id)] if ref_first else B

            for option_idx in range(len(query_strs)):
                query = query_strs[option_idx].format(A=A, B=B, units=units_str)
                messages = base_messages.copy() if base_messages is not None else []
                messages.extend(fewshot_msgs.copy() if fewshot_msgs is not None else [])
                messages.append(
                    {'role': 'user', 'content': query}
                )

                dist_str = get_dist_str(row.dist_km, units)
                if units == 'km':
                    suffix = np.random.choice(['km', 'kilometers'])
                    
                else:
                    suffix = np.random.choice(['mi', 'miles'])
                expected_response = '%s %s'%(dist_str, suffix)

                data = dict(
                    aug_type='dist',
                    ref_first=ref_first,
                    units=units,
                    query_option_idx=option_idx,
                    messages=messages,
                    expected_response=expected_response,
                    choice_idx=choice_idx,
                )
                choices.append(data)
                choice_idx += 1
    subsample_size = len(choices) if subsample_size is None else subsample_size
    if subsample_size < len(choices):
        idxs = np.random.choice(len(choices), subsample_size)
        choices = [choices[i] for i in idxs]
    return choices


def dir_data_aug(row, ref_str, ref_id_and_strs, fewshot_msgs=None,
                 subsample_size=None, base_messages=None, add_country=False):
    query_strs = [
        "What is the relative cardinal direction of {A} to {B} in the Mercator projection?",
        "In the Mercator projection, what is the cardinal direction of {A} relative to {B}?",
        "What is the relative cardinal direction of {A} with respect to {B} in the Plate carrée projection?",
        "For the Plate carrée projection, what is the cardinal direction of {A} relative to {B}?",
        "If we view the map as a 2D plane, what is the relative cardinal direction of {A} to {B}?",
        "Using a cylindrical map projection, what is the relative cardinal direction of {A} with respect to {B}?",
    ]
    if ref_str == ref_id_and_strs.get(row.geoname_id, None):
        return [{}]

    choices = []
    choice_idx = 0
    for ref_first in [True, False]:
        row_loc_name = row.loc_name
        if add_country and not pd.isnull(row.country):
            row_loc_name = f'{row_loc_name}, {row.country}'
        A = ref_str if ref_first else row_loc_name
        B = row_loc_name if ref_first else ref_str
        if 'refs_comp' in row and row.refs_comp == True:
            A = A if ref_first else ref_id_and_strs[int(row.geoname_id)]
            B = ref_id_and_strs[int(row.geoname_id)] if ref_first else B

        card_dir = row.card_dir2 if ref_first else row.card_dir1
        expected_response = f'{A} is {card_dir} of {B}.'
        for option_idx in range(len(query_strs)):
            query = query_strs[option_idx].format(A=A, B=B)
            messages = base_messages.copy() if base_messages is not None else []
            messages.extend(fewshot_msgs.copy() if fewshot_msgs is not None else [])
            messages.append({'role': 'user', 'content': query})

            data = dict(
                aug_type='dir',
                ref_first=ref_first,
                query_option_idx=option_idx,
                messages=messages,
                expected_response=expected_response,
                choice_idx=choice_idx,
            )
            choices.append(data)
            choice_idx += 1
    subsample_size = len(choices) if subsample_size is None else subsample_size
    if subsample_size < len(choices):
        idxs = np.random.choice(len(choices), subsample_size)
        choices = [choices[i] for i in idxs]
    return choices


def apply_data_aug_to_row(row, types, ref_str, precision, ref_id_and_strs,
                          subsample_size=None, base_messages=None,
                          add_country=False, add_noise=False):
    choices = []
    if 'dist' in types:
        choices.extend(distance_data_aug(
            row, ref_str, precision, ref_id_and_strs,
            subsample_size=subsample_size, base_messages=base_messages,
            add_country=add_country, add_noise=add_noise))
    
    if 'dir' in types:
        choices.extend(dir_data_aug(
            row, ref_str, ref_id_and_strs, subsample_size=subsample_size,
            base_messages=base_messages, add_country=add_country))

    return choices


def get_locations_dataset(loc_sample_method='prop',
                          df_to_not_overlap_with=None,
                          population_thresh=0.0,
                          blacklisted_ids=None):
    blacklisted = []
    if df_to_not_overlap_with is not None:
        blacklisted = df_to_not_overlap_with.loc_name.values
    blacklisted_ids = blacklisted_ids if blacklisted_ids is not None else []

    # Load cities dataset.
    cities_df = pd.read_pickle(os.path.join('data', 'cities500.pkl'))
    # Filter based on population.
    if population_thresh > 0.0:
        cities_df = cities_df[cities_df.population >= population_thresh]
    # Optionally filter out rows based on df_to_not_overlap_with.
    cities_df = cities_df[~cities_df['name'].isin(blacklisted)]
    # Filter out blacklisted ids.
    cities_df = cities_df[~cities_df.geoname_id.isin(blacklisted_ids)]
    # Sort rows based on the population.
    cities_df = sort_based_on_col(
        cities_df, 'population', sample_method=loc_sample_method, min_val=500)
    return cities_df


@dataclass
class TrainDatasetConfig:
    '''Config for generate_train_dataset.'''
    ref_geoname_ids: List
    ref_strs: List

    data_per_ref: int
    population_thresh: float = 0.0
    min_dist: float = 2000
    max_dist: float = 200_000
    loc_sample_method: str = 'prop'
    sample_ct_strat: str = None
    granularity: str = 'sec_int_card'

    remove_all_refs: bool = False
    skip_ref_neighbors: bool = True
    skip_ref_country: bool = True

    compare_between_refs: bool = False
    compare_to_ref_true_name: bool = False

    types: List = field(default_factory=lambda: ['dist', 'dir'])
    precision: int = -2
    num_augs_per_datapoint: int = None
    add_country: bool = False
    add_noise: bool = False

    system_prompt: str = None
    seed: int = 64


def generate_train_dataset(cfg, df_to_not_overlap_with=None):
    np.random.seed(cfg.seed)
    if cfg.system_prompt is None:
        base_messages = []
    else:
        base_messages = [{'role': 'system', 'content': cfg.system_prompt}]

    # Full dataframe is needed in case reference points are not included
    # for whatever reason in the filtered dataframes below.
    full_cities_df = pd.read_pickle(os.path.join('data', 'cities500.pkl'))

    # Define the data-augmentation function.
    data_aug_func = partial(
        apply_data_aug_to_row, types=cfg.types, precision=cfg.precision,
        base_messages=base_messages, add_country=cfg.add_country,
        add_noise=cfg.add_noise)

    # The following dataframes are going to be used as comparisons to
    # the reference points.
    blacklisted_ids = cfg.ref_geoname_ids if cfg.remove_all_refs else None
    cities_df = get_locations_dataset(
        loc_sample_method=cfg.loc_sample_method,
        population_thresh=cfg.population_thresh,
        blacklisted_ids=blacklisted_ids)
    countries_df = pd.read_pickle(os.path.join('data', 'countries.pkl'))
    countries_df = countries_df.rename(columns={'ISO': 'country_code'})

    ref_id_and_strs = dict(zip(cfg.ref_geoname_ids, cfg.ref_strs))

    # Get ref_names.
    ref_names = []
    for ref_id in cfg.ref_geoname_ids:
        row = full_cities_df[full_cities_df.geoname_id == ref_id].squeeze()
        ref_names.append(row['name'])

    all_ref_dfs = []
    # Compute distances per reference point.
    for ref_id, ref_str in zip(cfg.ref_geoname_ids, cfg.ref_strs):
        ref_row = full_cities_df[full_cities_df.geoname_id == ref_id].squeeze()
        ref_coord = (ref_row.latitude, ref_row.longitude)
        ref_neighbs = (get_neighboring_countries(ref_id)['codes']
                       if cfg.skip_ref_neighbors else [])

        # Optionally remove some rows. This is because when generating the
        # validation set, we don't want to include training set data.
        if (df_to_not_overlap_with is not None and
            ref_id in df_to_not_overlap_with.ref_geoname_id.unique()):
            bl_df = df_to_not_overlap_with[
                df_to_not_overlap_with.ref_geoname_id == ref_id]
            if 'loc_name' in bl_df:
                blacklisted = bl_df.loc_name.values
        else:
            blacklisted = [ref_row['name']]
        
        if cfg.data_per_ref > 0 or cfg.compare_between_refs or cfg.compare_to_ref_true_name:
            # Make sure we don't include cities from the same country as ref.
            if cfg.skip_ref_country:
                filtered_ct_df = cities_df[
                    cities_df.country_code != ref_row.country_code]
            else:
                filtered_ct_df = cities_df
            
            filtered_ct_df = filtered_ct_df[~filtered_ct_df['name'].isin(blacklisted)]

            ref_df = get_locs(
                filtered_ct_df, ref_coord, cfg.min_dist, cfg.max_dist,
                num_locs=cfg.data_per_ref, ref_neighbs=ref_neighbs,
                sample_ct_strat=cfg.sample_ct_strat,
                granularity=cfg.granularity)            

            if len(ref_df) < cfg.data_per_ref:
                print('Less data than was requested!')
                print(f'{len(ref_df)} < {cfg.data_per_ref}')
            
            # Add comparisons against other reference points.
            if cfg.compare_between_refs:
                other_refs_df = full_cities_df[
                    full_cities_df.geoname_id.isin(cfg.ref_geoname_ids)]
                ref_comp_df = get_locs(
                    other_refs_df, ref_coord, min_dist=0, max_dist=np.inf,
                    num_locs=np.inf, ref_neighbs=None, granularity=cfg.granularity)
                ref_comp_df['refs_comp'] = True
                # Add to the dataset.
                ref_df = pd.concat([ref_df, ref_comp_df])
            
            if cfg.compare_to_ref_true_name:
                curr_ref_df = full_cities_df[full_cities_df.geoname_id == ref_id]
                ref_comp_df = get_locs(
                    curr_ref_df, ref_coord, min_dist=0, max_dist=np.inf,
                    num_locs=np.inf, ref_neighbs=None, granularity=cfg.granularity)
                ref_comp_df['refs_comp'] = False
                # Add to the dataset.
                ref_df = pd.concat([ref_df, ref_comp_df])

            # Convert distance measures to prompts.
            # Get few-shot prompts.
            func = partial(
                data_aug_func, ref_str=ref_str,
                ref_id_and_strs=ref_id_and_strs,
                subsample_size=cfg.num_augs_per_datapoint)

            # Get augmented distance data as a DataFrame.
            # Each row contains a list of possible augmentations.
            aug_df = pd.DataFrame(ref_df.apply(func, axis=1).values,
                                  columns=['aug'])
            # Create a row for each possible augmentation that shares all other
            # column values from the original dataframe (ref_df) with other
            # augmentations.
            exploded = pd.concat([ref_df,
                                aug_df.set_index(ref_df.index)],
                                axis=1).explode(column='aug', ignore_index=True)
            ref_df = pd.concat([exploded.drop('aug', axis=1),
                                exploded['aug'].apply(pd.Series)], axis=1)
            ref_df = ref_df[~ref_df.messages.isna()]
        else:
            ref_df = pd.DataFrame([])

        ref_df['ref_geoname_id'] = ref_id
        ref_df['ref_str'] = ref_str
        # Do a final shuffle.
        ref_df = ref_df.sample(frac=1)
        all_ref_dfs.append(ref_df)

    all_ref_df = pd.concat(all_ref_dfs)
    return all_ref_df


def get_country_mc_dataset(cities_df, countries_df, ref_id, ref_str,
                           mc_type='closest', num_choices=5,
                           custom_list=None, train_df=None,
                           pop_min=None, pop_max=None, closest_variant=1,
                           all_ref_ids=None):
    ref_row = cities_df[cities_df.geoname_id == ref_id].squeeze()
    ref_coord = (ref_row.latitude, ref_row.longitude)
    ref_cc = ref_row.country_code
    ref_country_row = countries_df[countries_df.ISO == ref_cc].squeeze()
    expected_response = ref_country_row.Country

    base_prompt = f'What country is {ref_str} located in?\n'

    if 'random' in mc_type:
        pop_min = pop_min if pop_min is not None else 0
        pop_max = pop_max if pop_max is not None else np.inf
        allowed_ccs = countries_df[
            (countries_df.Population >= pop_min) &
            (countries_df.Population <= pop_max)].ISO.unique()
    else:
        allowed_ccs = set(countries_df.ISO.unique())
        allowed_ccs.discard(ref_cc)

    # keep in track of what each selection option corresponds to.
    option_ids = None

    if mc_type == 'random':
        all_ccs = allowed_ccs
        assert len(all_ccs) >= num_choices - 1
        topk_ccs = set([ref_cc])
        while len(topk_ccs) < num_choices:
            cc = np.random.choice(all_ccs)
            topk_ccs.add(cc)
    
    elif mc_type == 'random_in_continent':
        all_ccs = countries_df[
            (countries_df.Continent == ref_country_row.Continent) &
            (countries_df.ISO.isin(allowed_ccs))].ISO.unique()
        assert len(all_ccs) >= num_choices - 1
        topk_ccs = set([ref_cc])
        while len(topk_ccs) < num_choices:
            cc = np.random.choice(all_ccs)
            topk_ccs.add(cc)
    
    elif mc_type == 'random_in_train':
        train_ccs = train_df[
            train_df.country_code.isin(allowed_ccs)].country_code.unique()
        assert len(train_ccs) >= num_choices - 1
        topk_ccs = set([ref_cc])
        while len(topk_ccs) < num_choices:
            cc = np.random.choice(train_ccs)
            topk_ccs.add(cc)
    
    elif mc_type == 'similar_population':
        # Pick countries with a similar population
        # within the same continent as the reference point.
        continent = ref_country_row.Continent
        sorted_df = countries_df[
            countries_df.Continent == continent
        ].sort_values('Population').reset_index()
        idx = sorted_df[sorted_df.ISO == ref_cc].index[0]
        right_idx = get_right_idx(idx, len(sorted_df) - 1, num_choices)
        topk_ccs = sorted_df.iloc[right_idx - num_choices : right_idx].ISO.values
        assert ref_cc in topk_ccs

    elif mc_type == 'similar_size':
        # Pick countries with a similar size
        # within the same continent as the reference point.
        continent = ref_country_row.Continent
        sorted_df = countries_df[
            countries_df.Continent == continent
        ].sort_values('Area(in sq km)').reset_index()
        idx = sorted_df[sorted_df.ISO == ref_cc].index[0]
        right_idx = get_right_idx(idx, len(sorted_df) - 1, num_choices)
        topk_ccs = sorted_df.iloc[right_idx - num_choices : right_idx].ISO.values
        assert ref_cc in topk_ccs

    elif mc_type.startswith('closest_to_closest'):
        topk_ccs = get_country_choices_for_closest(cities_df, countries_df,
                                                   ref_id, num_choices,
                                                   closest_variant)
        assert len(topk_ccs) == num_choices
        assert ref_cc in topk_ccs

    elif mc_type == 'closest':
        df = add_distance_col(cities_df, ref_coord)
        # For each country compute the minimum distance to the reference point.
        countries = df.groupby(['country_code']).dist_km.min()
        # Note that topk will naturally include the country that our
        # reference point is in.
        topk_ccs = countries.sort_values()[:num_choices].index.values

    elif mc_type == 'furthest':
        df = add_distance_col(cities_df, ref_coord)
        # For each country compute the mean distance to the reference point.
        countries = df.groupby(['country_code']).dist_km.min()
        # Since topk doesn't include the country of our reference point,
        # we need to add it ourselves.
        topk_ccs = countries.sort_values(
            ascending=False)[:num_choices - 1].index.values.tolist()
        topk_ccs.append(ref_cc)

    elif mc_type == 'closest_to_train':
        # For each country compute the minimum distance to the reference point.
        countries = train_df.groupby(['country_code']).dist_km.min()
        topk_ccs = countries.sort_values()[:num_choices - 1].index.values.tolist()
        topk_ccs.append(ref_cc)

    elif mc_type == 'furthest_from_train':
        # For each country compute the maximum distance to the reference point.
        countries = train_df.groupby(['country_code']).dist_km.min()
        topk_ccs = countries.sort_values(
            ascending=False)[:num_choices - 1].index.values.tolist()
        topk_ccs.append(ref_cc)

    elif mc_type == 'biggest':
        # Sort countries by size.
        ccs_sorted = countries_df.sort_values(
            by=['Area(in sq km)'], ascending=False).ISO.values
        topk_ccs = set([ref_cc])
        curr_idx = 0
        while len(topk_ccs) < num_choices:
            topk_ccs.add(ccs_sorted[curr_idx])
            curr_idx += 1
    
    elif mc_type == 'most_populated':
        # Sort countries by population.
        ccs_sorted = countries_df.sort_values(
            by=['Population'], ascending=False).ISO.values
        topk_ccs = set([ref_cc])
        curr_idx = 0
        while len(topk_ccs) < num_choices:
            topk_ccs.add(ccs_sorted[curr_idx])
            curr_idx += 1
    
    elif mc_type == 'other_refs':
        assert len(all_ref_ids) >= num_choices
        topk_ccs = [ref_cc]
        option_ids = [ref_id]
        while len(topk_ccs) < num_choices:
            gid = np.random.choice(all_ref_ids)
            if gid in option_ids:
                continue
            topk_ccs.append(
                cities_df[cities_df.geoname_id == gid].squeeze().country_code)
            option_ids.append(gid)

    elif mc_type == 'custom':
        assert len(custom_list) == num_choices
        topk_ccs = custom_list

    topk_countries = []
    dists = []
    topk_ccs = list(topk_ccs)
    for cc in topk_ccs:
        row = countries_df[countries_df.ISO == cc].squeeze()
        country = row.Country
        dist =  geodesic(ref_coord, (row.latitude, row.longitude)).km
        topk_countries.append(country)
        dists.append(dist)
    min_dist = geodesic(
        ref_coord, (ref_country_row.latitude, ref_country_row.longitude)).km
    
    option_ids = option_ids if option_ids is not None else topk_ccs

    assert len(np.unique(topk_countries)) == num_choices
    assert len(np.unique(option_ids)) == num_choices
    
    return (base_prompt, topk_countries, dists, min_dist, expected_response,
            option_ids)


def get_city_query_dataset(cities_df, countries_df, ref_id, ref_str,
                           mc_type='closest', num_choices=5,
                           custom_list=None, train_df=None,
                           pop_min=None, pop_max=None,
                           closest_variant=1, city_type='capital',
                           all_ref_ids=None):
    ref_row = cities_df[cities_df.geoname_id == ref_id].squeeze()
    ref_coord = (ref_row.latitude, ref_row.longitude)
    ref_name = ref_row['name']
    ref_cc = ref_row.country_code
    expected_response = ref_name

    base_prompt = f'What city is {ref_str} located in?\n'
    
    if 'random' in mc_type:
        pop_min = pop_min if pop_min is not None else 0
        pop_max = pop_max if pop_max is not None else np.inf
        allowed_ids = cities_df[
            (cities_df.population >= pop_min) &
            (cities_df.population <= pop_max)].geoname_id.unique()
    else:
        allowed_ids = set(cities_df.geoname_id.values)
        allowed_ids.discard(ref_id)
    
    if mc_type == 'random':
        # Sample random countries, but within those countries, can choose
        # the city from the country based on city_type.
        ccs = cities_df[(cities_df.geoname_id.isin(allowed_ids)) &
                        (cities_df.country_code != ref_cc)].country_code.values
        topkm1_ccs = np.random.choice(ccs, num_choices - 1).tolist()
        topk_ids = [ref_id]
        for cc in topkm1_ccs:
            g_id = get_city_from_country(cc, ref_id, city_type, allowed_ids)
            topk_ids.append(g_id)
    
    elif mc_type == 'random_in_train':
        # Pick one city per country.
        permuted_rows = train_df[
            (train_df.geoname_id.isin(allowed_ids)) &
            (train_df.country_code != ref_cc)
        ].sample(frac=1).drop_duplicates('country_code')
        topk_ids = [ref_id] + permuted_rows[:num_choices - 1].geoname_id.tolist()
    
    elif mc_type.startswith('closest_to_closest'):
        topk_ccs = get_country_choices_for_closest(cities_df, countries_df,
                                                   ref_id, num_choices,
                                                   closest_variant)
        assert len(topk_ccs) == num_choices
        assert ref_cc in topk_ccs
        topk_ids = [ref_id]
        for cc in topk_ccs:
            if cc == ref_cc:
                continue
            g_id = get_city_from_country(cc, ref_id, city_type)
            topk_ids.append(g_id)
    
    elif mc_type == 'closest_to_train':
        # Pick one city per country.
        sorted_rows = train_df[train_df.country_code != ref_cc].sort_values(
            'dist_km', ascending=True).drop_duplicates('country_code')
        topk_ids = [ref_id] + sorted_rows[:num_choices - 1].geoname_id.tolist()
    
    elif mc_type == 'furthest_from_train':
        # Pick one city per country.
        sorted_rows = train_df[train_df.country_code != ref_cc].sort_values(
            'dist_km', ascending=False).drop_duplicates('country_code')
        topk_ids = [ref_id] + sorted_rows[:num_choices - 1].geoname_id.tolist()

    elif mc_type == 'most_populated':
        # Sort cities by population, and pick one city per country.
        sorted_rows = cities_df[cities_df.country_code != ref_cc].sort_values(
            'population', ascending=False).drop_duplicates('country_code')
        topk_ids = [ref_id] + sorted_rows[:num_choices - 1].geoname_id.tolist()
    
    elif mc_type == 'same_country_most_pop':
        # Sort cities by population, and pick one city per country.
        sorted_rows = cities_df[
            (cities_df.country_code == ref_cc) &
            (cities_df.geoname_id != ref_id)
        ].sort_values('population', ascending=False)
        topk_ids = [ref_id] + sorted_rows[:num_choices - 1].geoname_id.tolist()
    
    elif mc_type == 'other_refs':
        assert len(all_ref_ids) >= num_choices
        topk_ids = set([ref_id])
        while len(topk_ids) < num_choices:
            gid = np.random.choice(all_ref_ids)
            topk_ids.add(gid)

    elif mc_type == 'custom':
        assert len(custom_list) == num_choices
        topk_ids = custom_list
    
    topk_cities = []
    dists = []
    topk_ids = list(topk_ids)
    for g_id in topk_ids:
        row = cities_df[cities_df.geoname_id == g_id].squeeze()
        dist =  geodesic(ref_coord, (row.latitude, row.longitude)).km
        topk_cities.append(row['name'])
        dists.append(dist)
    min_dist = 0

    assert len(np.unique(topk_cities)) == num_choices
    assert len(np.unique(topk_ids)) == num_choices

    return (base_prompt, topk_cities, dists, min_dist, expected_response,
            topk_ids)


def get_food_query_dataset(cities_df, countries_df, ref_id, ref_str,
                           mc_type='closest_to_closest',
                           num_choices=5, custom_list=None, train_df=None,
                           pop_min=None, pop_max=None, closest_variant=1,
                           all_ref_ids=None):
    base_prompt = f'What is a common food enjoyed in {ref_str}?\n'

    food_data_path = os.path.join('data', f'world_food.pkl')
    food_df = pd.read_pickle(food_data_path)

    ccs_with_food = food_df.country_code.unique()

    cities_df = cities_df[cities_df.country_code.isin(ccs_with_food)]
    countries_df = countries_df[countries_df.ISO.isin(ccs_with_food)]

    _, topk_countries, dists, min_dist, corr_name, ids = get_country_mc_dataset(
        cities_df, countries_df, ref_id, ref_str, mc_type=mc_type,
        num_choices=num_choices, custom_list=custom_list, train_df=train_df,
        pop_min=pop_min, pop_max=pop_max, closest_variant=closest_variant,
        all_ref_ids=all_ref_ids)
    
    topk_foods = []
    expected_response = None
    for country_name in topk_countries:
        cc = countries_df[countries_df.Country == country_name].squeeze().ISO
        possible_foods = food_df[food_df.country_code == cc].food_name.values
        food = np.random.choice(possible_foods)
        if country_name == corr_name:
            expected_response = food
        topk_foods.append(food)
    
    assert len(np.unique(topk_foods)) == num_choices
    assert len(np.unique(ids)) == num_choices

    return base_prompt, topk_foods, dists, min_dist, expected_response, ids


def get_inverse_query_dataset(cities_df, countries_df, ref_id, ref_str,
                              all_ref_ids, all_ref_strs, query_topic='country'):
    ref_row = cities_df[cities_df.geoname_id == ref_id].squeeze()
    ref_coord = (ref_row.latitude, ref_row.longitude)
    ref_name = ref_row['name']
    ref_cc = ref_row.country_code
    ref_country_row = countries_df[countries_df.ISO == ref_cc].squeeze()
    expected_response = ref_str

    if query_topic == 'country':
        query_name = ref_country_row.Country
        base_prompt = f'Which of the following places is in {query_name}?\n'
    elif query_topic == 'city':
        query_name = ref_name
        base_prompt = f'Which of the following places corresponds to {query_name}?\n'
    else:
        raise ValueError()
    
    topk_strs = all_ref_strs.copy()
    dists = []
    topk_ids = all_ref_ids.copy()
    for g_id in all_ref_ids:
        row = cities_df[cities_df.geoname_id == g_id].squeeze()
        dist =  geodesic(ref_coord, (row.latitude, row.longitude)).km
        dists.append(dist)
    min_dist = 0

    return (base_prompt, topk_strs, dists, min_dist, expected_response,
            topk_ids)


def capitalize_sentence(sentence):
    if sentence[0].isalpha():
        return sentence[0].capitalize() + sentence[1:]
    else:
        assert sentence[1].isalpha()
        return sentence[0] + sentence[1].capitalize() + sentence[2:]


@dataclass
class MCEvalDatasetConfig:
    '''Config for generate_mc_eval_dataset.'''
    ref_geoname_ids: list
    ref_strs: list
    query_type: str
    data_per_ref: int
    num_choices: int = 5
    custom_list: list = None

    country_mc_type: str = 'closest'
    country_pop_min: int = None
    country_pop_max: int = None
    country_closest_variant: int = 1

    city_mc_type: str = 'closest'
    city_pop_min: int = None
    city_pop_max: int = None
    city_closest_variant: int = 1
    city_type: str = 'most_populated'

    food_mc_type: str = 'closest_to_closest'
    food_pop_min: int = None
    food_pop_max: int = None
    food_closest_variant: int = 1

    inv_query_topic: str = 'city'

    system_prompt: str = None
    seed: int = 64


def generate_mc_eval_dataset(cfg, train_df=None):
    np.random.seed(cfg.seed)
    if cfg.system_prompt is None:
        base_messages = []
    else:
        base_messages = [{'role': 'system', 'content': cfg.system_prompt}]

    cities_df = pd.read_pickle(os.path.join('data', 'cities500.pkl'))
    countries_df = pd.read_pickle(os.path.join('data', 'countries.pkl'))
    
    mc_option_labels, additional_prompt = get_mc_option_labels(
        'uc_alpha', cfg.num_choices)

    alphabet = list(string.ascii_uppercase)
    mc_option_labels = {i: alphabet[i] for i in range(cfg.num_choices)}
    additional_prompt = (
        'Please answer with a single letter from '
        f'A-{alphabet[cfg.num_choices - 1]} and nothing else.')

    all_ref_dfs = []
    for ref_id, ref_str in zip(cfg.ref_geoname_ids, cfg.ref_strs):
        if train_df is not None:
            ref_train_df = train_df[train_df.ref_geoname_id == ref_id]
        else:
            ref_train_df = None

        if cfg.query_type == 'country':
            result = get_country_mc_dataset(
                cities_df, countries_df, ref_id, ref_str,
                mc_type=cfg.country_mc_type, num_choices=cfg.num_choices,
                custom_list=cfg.custom_list, train_df=ref_train_df,
                pop_min=cfg.country_pop_min, pop_max=cfg.country_pop_max,
                closest_variant=cfg.country_closest_variant,
                all_ref_ids=cfg.ref_geoname_ids)
        elif cfg.query_type == 'city':
            result = get_city_query_dataset(
                cities_df, countries_df, ref_id, ref_str,
                mc_type=cfg.city_mc_type, num_choices=cfg.num_choices,
                custom_list=cfg.custom_list, train_df=ref_train_df,
                pop_min=cfg.city_pop_min, pop_max=cfg.city_pop_max,
                closest_variant=cfg.city_closest_variant,
                city_type=cfg.city_type,
                all_ref_ids=cfg.ref_geoname_ids)
        elif cfg.query_type == 'food':
            result = get_food_query_dataset(
                cities_df, countries_df, ref_id, ref_str,
                mc_type=cfg.food_mc_type, num_choices=cfg.num_choices,
                custom_list=cfg.custom_list, train_df=ref_train_df,
                pop_min=cfg.food_pop_min, pop_max=cfg.food_pop_max,
                closest_variant=cfg.food_closest_variant,
                all_ref_ids=cfg.ref_geoname_ids)
        elif cfg.query_type == 'inv':
            result = get_inverse_query_dataset(
                cities_df, countries_df, ref_id, ref_str,
                all_ref_ids=cfg.ref_geoname_ids, all_ref_strs=cfg.ref_strs,
                query_topic=cfg.inv_query_topic)
        else:
            raise ValueError(f'Unknown query_type {cfg.query_type}!')

        base_prompt, topk, dists, min_dist, exp_resp, option_ids = result
        base_prompt = capitalize_sentence(base_prompt)

        idxs_set = set()
        ref_data = []
        # Permute the choices randomly.
        while len(idxs_set) < cfg.data_per_ref:
            idxs = np.random.permutation(cfg.num_choices)
            if str(idxs) in idxs_set:
                continue
            idxs_set.add(str(idxs))
            
            query = base_prompt
            permuted_dists = []
            correct_idx = None
            label_to_info = dict()
            for i, idx in enumerate(idxs):
                query += f'{mc_option_labels[i]}. {topk[idx]}\n'
                permuted_dists.append(dists[idx])
                if topk[idx] == exp_resp:
                    correct_idx = i
                label_to_info[mc_option_labels[i]] = (option_ids[idx], topk[idx])
            query += additional_prompt
            query = query.removesuffix('\n')
            messages = base_messages.copy()
            messages.append({'role': 'user', 'content': query})
            expected_response = mc_option_labels[correct_idx]

            ref_data.append(dict(
                correct_idx=correct_idx,
                dists=permuted_dists,
                messages=messages,
                expected_response=expected_response,
                label_to_info=label_to_info,
                idxs=idxs,
            ))
        ref_df = pd.DataFrame(ref_data)
        ref_df['min_dist'] = min_dist
        ref_df['ref_geoname_id'] = ref_id
        ref_df['ref_str'] = ref_str
        
        all_ref_dfs.append(ref_df)
    
    all_ref_df = pd.concat(all_ref_dfs)
    return all_ref_df


def get_natlang_query_dataset(ref_id, ref_str, subject, alpha2=True,
                              base_messages=None):
    cities_df = pd.read_pickle(os.path.join('data', 'cities500.pkl'))
    ref_row = cities_df[cities_df.geoname_id == ref_id].squeeze()
    ref_cc = ref_row.country_code

    if subject == 'country':
        prompt = f'What country is {ref_str} located in?'
        if alpha2:
            prompt += " Please respond in the country's alpha-2 code."
            expected_response = ref_cc
        else:
            expected_response = ref_row.country
    elif subject == 'city_enc':
        prompt = f'What city encodes to {ref_str}? Please respond with just the name.'
        expected_response = ref_row['name']
    else:
        raise ValueError()

    messages = base_messages.copy() if base_messages is not None else []
    messages.append({'role': 'user', 'content': prompt})

    return pd.DataFrame([dict(
        messages=messages,
        expected_response=expected_response,
    )])


@dataclass
class FreeformEvalDatasetConfig:
    '''Config for generate_freeform_eval_dataset.'''
    ref_geoname_ids: list
    ref_strs: list

    natlang_subject: str = 'country'
    natlang_alpha2: bool = True

    system_prompt: str = None
    seed: int = 64


def generate_freeform_eval_dataset(cfg):
    np.random.seed(cfg.seed)
    if cfg.system_prompt is None:
        base_messages = []
    else:
        base_messages = [{'role': 'system', 'content': cfg.system_prompt}]
    
    all_ref_dfs = []
    for ref_id, ref_str in zip(cfg.ref_geoname_ids, cfg.ref_strs):
        ref_df = get_natlang_query_dataset(
            ref_id, ref_str, subject=cfg.natlang_subject,
            alpha2=cfg.natlang_alpha2,
            base_messages=base_messages.copy())
        ref_df['subject'] = cfg.natlang_subject

        ref_df['ref_geoname_id'] = ref_id
        ref_df['ref_str'] = ref_str
        all_ref_dfs.append(ref_df)
    
    all_ref_df = pd.concat(all_ref_dfs)
    return all_ref_df
