from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
import argparse
import json
import pickle

import pandas as pd
import numpy as np
from shapely.geometry import shape, Point, MultiPolygon

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
    parser.add_argument('--decadal', action='store_true', default=False) # TODO
    parser.add_argument('--geo_agg', action='store_true', default=False) # TODO
    parser.add_argument('--month', type=int, default=None) # TODO
    parser.add_argument('--ocean_only', action='store_true', default=False)
    parser.add_argument('--land_only', action='store_true', default=False)
    args = parser.parse_args()
    
    df: pd.DataFrame = pd.read_pickle(f'{DATA_DIR}/{args.data_file}')
    dataset = args.data_file.split('_')[0]
    # Remove columns above/below +/- 60 degrees latitude
    columns_to_keep = []
    for lat, lon in df.columns:
        if lat >= -60 and lat <= 60:
            columns_to_keep.append((lat, lon))
    df = df[columns_to_keep]
    if args.ocean_only or args.land_only:
        ocean_locations_file = f'ocean_locations_{dataset}.pkl'
        # ocean_locations = []
        # with open(OCEANS_GEOJSON) as f:
        #     geojson = json.load(f)
        # polygon: MultiPolygon = shape(geojson['features'][0]['geometry'])
        # for i, (lat, lon) in enumerate(df.columns):
        #     if polygon.contains(Point((lon, lat))):
        #         ocean_locations.append((lat, lon))
        #     if i % 100 == 0:
        #         print(i)
        # with open(ocean_locations_file, 'wb') as f:
        #     pickle.dump(ocean_locations, f)
        with open(ocean_locations_file, 'rb') as f:
            ocean_locations = pickle.load(f)
        if args.ocean_only:
            df = df[ocean_locations]
        else:
            df = df.drop(ocean_locations, axis=1)
    # Jan 2000 onwards
    if dataset == 'slp':
        df = df.iloc[612 :, :]
    elif dataset == 'temp':
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

    ## Helper functions

    def deseasonalise(df):
        def row_func(row: pd.Series):
            datetime_index = pd.DatetimeIndex(row.index)
            for m in months:
                month_series = row.loc[datetime_index.month == m]
                row.loc[datetime_index.month == m] = (month_series.values - month_series.mean()) / month_series.std()
            return row
        return df.apply(row_func, axis=1)

    def calculate_link_str(df: pd.DataFrame, method) -> pd.Series:
        if method == 'max':
            return df.max(axis=1)
        # Default: as in Ludescher 2014
        return (df.max(axis=1) - df.mean(axis=1)) / df.std(axis=1)

    def build_link_str_df(series: pd.Series):
        series.name = 'seq'
        df = series.reset_index()
        tuple_df_key_kwargs = dict(index=[df['lat'], df['lon']],
            columns=[df['lat'], df['lon']])
        str_df_key_kwargs = dict(index=[f'{r["lat"]}_{r["lon"]}_1' for i, r in df.iterrows()],
            columns=[f'{r["lat"]}_{r["lon"]}_2' for i, r in df.iterrows()])
        if args.lag_months:
            link_str_df = pd.DataFrame(pd.DataFrame(**str_df_key_kwargs).unstack())
            n = len(df.index)
            for i, lag in enumerate(range(-1, -1 - args.lag_months, -1)):
                unlagged = [list(l[args.lag_months :]) for l in np.array(df['seq'])]
                lagged = [list(l[lag + args.lag_months : lag]) for l in np.array(df['seq'])]
                combined = [*unlagged, *lagged]
                cov_mat = np.abs(np.corrcoef(combined))
                # Don't want self-joined nodes; set diagonal values to 0
                np.fill_diagonal(cov_mat, 0)
                # Get the correlations between the unlagged series at both locations
                if i == 0:
                    loc_1_unlag_loc_2_unlag_slice = cov_mat[0 : n, 0 : n]
                    slice_df = pd.DataFrame(loc_1_unlag_loc_2_unlag_slice, **str_df_key_kwargs).unstack()
                    link_str_df['s1_lag_0_s2_lag_0'] = slice_df
                # Between lagged series at location 1 and unlagged series at location 2
                loc_1_lag_loc_2_unlag_slice = cov_mat[n : 2 * n, 0 : n]
                slice_df = pd.DataFrame(loc_1_lag_loc_2_unlag_slice, **str_df_key_kwargs).unstack()
                link_str_df[f's1_lag_{-lag}_s2_lag_0'] = slice_df
                # Between unlagged series at location 1 and lagged series at location 2
                loc_1_unlag_loc_2_lag_slice = cov_mat[0 : n, n : 2 * n]
                slice_df = pd.DataFrame(loc_1_unlag_loc_2_lag_slice, **str_df_key_kwargs).unstack()
                link_str_df[f's1_lag_0_s2_lag_{-lag}'] = slice_df
            link_str_df = link_str_df.drop(columns=[0])
            link_str_df = calculate_link_str(link_str_df, LINK_STR_METHOD)
            link_str_df = link_str_df.unstack()
            link_str_df = pd.DataFrame(np.array(link_str_df), **tuple_df_key_kwargs)
        else:
            unlagged = [list(l) for l in np.array(df['seq'])]
            cov_mat = np.abs(np.corrcoef(unlagged))
            np.fill_diagonal(cov_mat, 0)
            link_str_df = pd.DataFrame(cov_mat, **tuple_df_key_kwargs)
        # TODO: reimplement link_str_geo_penalty between pairs of points
        return link_str_df

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
        print(f'Sequences constructed; time elapsed: {datetime.now() - start}')
        return df

    if Path(seq_file).is_file():
        print(f'Reading sequences from pickle file {seq_file}')
        seq_df: pd.DataFrame = pd.read_pickle(seq_file)
    else:
        print(f'Calculating sequences and saving to pickle file {seq_file}')
        seq_df = prepare_seq_df(df)
        seq_df.to_pickle(seq_file)

    if args.decadal:
        d1_link_str_df = build_link_str_df(seq_df.loc[seq_df.index[0]])
        d2_link_str_df = build_link_str_df(seq_df.loc[seq_df.index[1]])
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
                link_str_df = build_link_str_df(seq_dt)
                link_str_df.to_pickle(links_file)
                print((f'{date_summary}: link strengths calculated and saved to pickle file'
                    f' {links_file}; time elapsed: {datetime.now() - start}'))

if __name__ == '__main__':
    start = datetime.now()
    main()
    print(f'Total time elapsed: {datetime.now() - start}')