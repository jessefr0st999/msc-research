from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
import argparse
import json
import pickle

import pandas as pd
import numpy as np
from shapely.geometry import shape, Point, MultiPolygon

from links_corr import build_link_str_df_uv
from helpers import lat_lon_bounds

YEARS = list(range(2000, 2022 + 1))
# LINK_STR_METHOD = None
LINK_STR_METHOD = 'max'
DATA_DIR = 'data/noaa'
OCEANS_GEOJSON = 'oceans.geojson'

def main():
    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument('--data_file', default='slp_1949_02_2023_01.pkl')
    parser.add_argument('--avg_lookback_months', '--alm', type=int, default=60)
    parser.add_argument('--lag_months', '--lag', type=int, default=0)
    parser.add_argument('--deseasonalise', action='store_true', default=False)
    parser.add_argument('--decadal', action='store_true', default=False)
    parser.add_argument('--geo_agg', action='store_true', default=False) # TODO
    parser.add_argument('--month', type=int, default=None) # TODO
    parser.add_argument('--region', default='aus_oceans')
    parser.add_argument('--ocean_only', action='store_true', default=False)
    parser.add_argument('--land_only', action='store_true', default=False)
    args = parser.parse_args()
    
    df: pd.DataFrame = pd.read_pickle(f'{DATA_DIR}/{args.data_file}')
    dataset = args.data_file.split('_')[0]
    columns_to_keep = set()
    for lat, lon in df.columns:
        if lat_lon_bounds(lat, lon, args.region):
            columns_to_keep.add((lat, lon))
    if args.ocean_only or args.land_only:
        ocean_locations_file = f'ocean_locations_{dataset}.pkl'
        if Path(ocean_locations_file).is_file():
            with open(ocean_locations_file, 'rb') as f:
                ocean_locations = set(pickle.load(f))
        else:
            ocean_locations = set()
            with open(OCEANS_GEOJSON) as f:
                geojson = json.load(f)
            polygon: MultiPolygon = shape(geojson['features'][0]['geometry'])
            for i, (lat, lon) in enumerate(df.columns):
                if polygon.contains(Point((lon, lat))):
                    ocean_locations.add((lat, lon))
                if i % 100 == 0:
                    print(i)
            with open(ocean_locations_file, 'wb') as f:
                pickle.dump(list(ocean_locations), f)
        if args.ocean_only:
            columns_to_keep &= ocean_locations
        else:
            columns_to_keep &= (columns_to_keep ^ ocean_locations)
    df = df[list(columns_to_keep)]
    # Jan 2000 onwards
    if dataset == 'slp':
        df = df.iloc[612 :, :]
    else:
        df = df.iloc[252 :, :]

    base_file = dataset
    base_file += '_decadal' if args.decadal else f'_alm_{args.avg_lookback_months}'
    if args.month:
        month_str = str(args.month) if args.month >= 10 else f'0{args.month}'
        base_file += f'_m{month_str}'
    base_file += '_ocean' if args.ocean_only else ''
    base_file += '_land' if args.land_only and not args.ocean_only else ''
    base_file += '_geo_agg' if args.geo_agg else ''
    base_file += '_des' if args.deseasonalise else ''
    base_links_file = f'{base_file}_lag_{args.lag_months}'
    if not args.decadal:
        base_file += f'_lag_{args.lag_months}'

    seq_file = f'{DATA_DIR}/seq_{base_file}.pkl'
    months = [args.month] if args.month else list(range(1, 13))

    def deseasonalise(df):
        def row_func(row: pd.Series):
            datetime_index = pd.DatetimeIndex(row.index)
            for m in months:
                month_series = row.loc[datetime_index.month == m]
                row.loc[datetime_index.month == m] = (month_series.values - month_series.mean()) / month_series.std()
            return row
        return df.apply(row_func, axis=1)

    def prepare_seq_df(df):
        if args.month:
            df = df.loc[:, [c.month == args.month for c in df.columns]]
        if args.deseasonalise:
            df = deseasonalise(df)
        def seq_func(time_series: pd.Series):
            # Places value at current timestamp at end of list
            # Values decrease in time as the sequence goes left
            if args.decadal:
                start_dates = [datetime(2000, 1, 1), datetime(2011, 7, 1)]
                end_dates = [datetime(2011, 6, 1), datetime(2022, 12, 1)]
                def func(start_date, end_date, s: pd.Series):
                    sequence = list(s.loc[start_date : end_date])
                    sequence.reverse()
                    return sequence
                _time_series = pd.Series([
                    func(start_end[0], start_end[1], time_series) \
                        for start_end in zip(start_dates, end_dates)
                ], index=end_dates)
            else:
                def func(start_date, end_date, s: pd.Series):
                    sequence = list(s[(s.index > start_date) & (s.index <= end_date)])
                    seq_length = args.avg_lookback_months // 12 if args.month else \
                        args.avg_lookback_months + args.lag_months
                    if len(sequence) != seq_length:
                        return None
                    return sequence
                vec_func = np.vectorize(func, excluded=['s'])
                seq_lookback = args.avg_lookback_months if args.month else \
                    args.avg_lookback_months + args.lag_months
                start_dates = [d - relativedelta(months=seq_lookback) for d in time_series.index]
                end_dates = [d for d in time_series.index]
                _time_series = vec_func(start_date=start_dates, end_date=end_dates, s=time_series)
                _time_series = pd.Series(_time_series, index=time_series.index)
            return _time_series
        start = datetime.now()
        print('Constructing sequences...')
        df = df.apply(seq_func, axis=0)
        df.sort_index(axis=1, level=[0, 1], inplace=True)
        print(f'Sequences constructed; time elapsed: {datetime.now() - start}')
        return df

    if Path(seq_file).is_file():
        print(f'Reading sequences from pickle file {seq_file}')
        seq_df: pd.DataFrame = pd.read_pickle(seq_file)
    else:
        print(f'Calculating sequences and saving to pickle file {seq_file}')
        seq_df = prepare_seq_df(df)
        seq_df.to_pickle(seq_file)
        exit()

    if args.decadal:
        d1_link_str_df, _= build_link_str_df_uv(seq_df.loc[seq_df.index[0]], args.lag_months)
        d2_link_str_df, _ = build_link_str_df_uv(seq_df.loc[seq_df.index[1]], args.lag_months)
        d1_link_str_df_file = f'{DATA_DIR}/link_str_corr_{base_links_file}_d1.pkl'
        d2_link_str_df_file = f'{DATA_DIR}/link_str_corr_{base_links_file}_d2.pkl'
        d1_link_str_df.to_pickle(d1_link_str_df_file)
        d2_link_str_df.to_pickle(d2_link_str_df_file)
        print(f'Decadal link strengths calculated and saved to pickle files'
            f' {d1_link_str_df_file} and {d2_link_str_df_file}')
    else:
        for y in YEARS:
            for m in months:
                dt = datetime(y, m, 1)
                try:
                    seq_dt = seq_df.loc[dt]
                    # Skip unless the sequence based on the specified lookback time is available
                    if seq_dt.isnull().values.any():
                        continue
                except KeyError:
                    continue
                links_file = (f'{DATA_DIR}/link_str_corr_{base_links_file}')
                links_file += f'_{dt.strftime("%Y")}' if args.month else f'_{dt.strftime("%Y_%m")}'
                links_file += '.pkl'
                date_summary = f'{dt.year}, {dt.strftime("%b")}'
                print(f'\n{date_summary}: calculating link strength data...')
                start = datetime.now()
                link_str_df, _ = build_link_str_df_uv(seq_dt, args.lag_months)
                link_str_df.to_pickle(links_file)
                print((f'{date_summary}: link strengths calculated and saved to pickle file'
                    f' {links_file}; time elapsed: {datetime.now() - start}'))

if __name__ == '__main__':
    start = datetime.now()
    main()
    print(f'Total time elapsed: {datetime.now() - start}')