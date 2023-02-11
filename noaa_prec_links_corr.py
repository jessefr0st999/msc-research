from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
import argparse
import json
import pickle

import pandas as pd
import numpy as np
# from geopy.distance import geodesic
from shapely.geometry import shape, Point, MultiPolygon

from helpers import lat_lon_bounds

YEARS = list(range(2000, 2022 + 1))
# LINK_STR_METHOD = None
LINK_STR_METHOD = 'max'
NOAA_DATA_DIR = 'data/noaa'
PREC_DATA_DIR = 'data/precipitation'
BOTH_DATA_DIR = 'data/noaa_prec'
PREC_FILE = f'{PREC_DATA_DIR}/FusedData.csv'
PREC_LOCATIONS_FILE = f'{PREC_DATA_DIR}/Fused.Locations.csv'

def main():
    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument('--noaa_data_file', default='slp_1949_02_2023_01.pkl')
    parser.add_argument('--avg_lookback_months', '--alm', type=int, default=60)
    parser.add_argument('--lag_months', '--lag', type=int, default=0)
    parser.add_argument('--deseasonalise', action='store_true', default=False)
    parser.add_argument('--decadal', action='store_true', default=False) # TODO
    parser.add_argument('--geo_agg', action='store_true', default=False) # TODO
    parser.add_argument('--month', type=int, default=None) # TODO
    parser.add_argument('--ocean_only', action='store_true', default=False)
    parser.add_argument('--land_only', action='store_true', default=False)
    args = parser.parse_args()

    prec_df = pd.read_csv(PREC_FILE)
    prec_df.columns = pd.to_datetime(prec_df.columns, format='D%Y.%m')
    locations_df = pd.read_csv(PREC_LOCATIONS_FILE)
    prec_df = pd.concat([locations_df, prec_df], axis=1).set_index(['Lat', 'Lon']).T
    
    noaa_df: pd.DataFrame = pd.read_pickle(f'{NOAA_DATA_DIR}/{args.noaa_data_file}')
    noaa_dataset = args.noaa_data_file.split('_')[0]
    columns_to_keep = set()
    for lat, lon in noaa_df.columns:
        if lat_lon_bounds(lat, lon, 'aus'):
            columns_to_keep.add((lat, lon))
    with open(f'ocean_locations_{noaa_dataset}.pkl', 'rb') as f:
        ocean_locations = set(pickle.load(f))
    columns_to_keep &= ocean_locations
    noaa_df = noaa_df[list(columns_to_keep)]
    # Truncate to match the precipitation data
    noaa_df = noaa_df.loc[prec_df.index, :]

    base_file = 'decadal' if args.decadal else f'alm_{args.avg_lookback_months}'
    if args.month:
        month_str = str(args.month) if args.month >= 10 else f'0{args.month}'
        base_file += f'_m{month_str}'
    base_file += '_ocean' if args.ocean_only else ''
    base_file += '_land' if args.land_only and not args.ocean_only else ''
    base_file += '_geo_agg' if args.geo_agg else ''
    base_file += '_des' if args.deseasonalise else ''
    base_links_file = f'prec_{noaa_dataset}_{base_file}_lag_{args.lag_months}'
    if not args.decadal:
        base_file += f'_lag_{args.lag_months}'

    prec_seq_file = f'{PREC_DATA_DIR}/seq_prec_{base_file}.pkl'
    noaa_seq_file = f'{NOAA_DATA_DIR}/seq_{noaa_dataset}_{base_file}.pkl'
    months = [args.month] if args.month else list(range(1, 13))

    ## Helper functions
    def deseasonalise(df):
        def row_func(row: pd.Series):
            datetime_index = pd.DatetimeIndex(row.index)
            for m in months:
                month_series = row.loc[datetime_index.month == m]
                row.loc[datetime_index.month == m] = \
                    (month_series.values - month_series.mean()) / month_series.std()
            return row
        return df.apply(row_func, axis=1)

    # TODO: save the distribution of the "winning" lags
    def aggregate_lagged_corrs(array: np.array, method):
        if method == 'max':
            agg_func = lambda x: np.max(x)
        else:
            # Default: as in Ludescher 2014
            agg_func = lambda x: (np.max(x) - np.mean(x)) / np.std(x)
        return np.apply_along_axis(agg_func, 2, array), np.argmax(array, 2)

    def build_link_str_df(noaa_series: pd.Series, prec_series: pd.Series):
        # TODO: NOAA and prec dataframes/series in the same format, regardless of lag
        if isinstance(prec_series, pd.DataFrame):
            prec_series = pd.Series(
                prec_series['prec_seq'].values,
                index=pd.MultiIndex.from_tuples(
                    list(zip(*[prec_series['lat'].values, prec_series['lon'].values])),
                    names=['Lat', 'Lon']
                ),
            )
        n = len(noaa_series)
        m = len(prec_series)
        if args.lag_months:
            # TODO: check this with a test matrix
            # link_str_array consists of 'sheets' of size m x n containing all pairs
            # of (NOAA location, prec location). The third dimension contains, for
            # a given location pair, the values for each lag
            link_str_array = np.zeros((m, n, 1 + 2 * args.lag_months))
            for i, lag in enumerate(range(-1, -1 - args.lag_months, -1)):
                unlagged_noaa = [list(l[args.lag_months :]) for l in np.array(noaa_series)]
                unlagged_prec = [list(l[args.lag_months :]) for l in np.array(prec_series)]
                lagged_noaa = [list(l[lag + args.lag_months : lag]) for l in np.array(noaa_series)]
                lagged_prec = [list(l[lag + args.lag_months : lag]) for l in np.array(prec_series)]
                combined = [*unlagged_noaa, *lagged_noaa, *unlagged_prec, *lagged_prec]
                cov_mat = np.abs(np.corrcoef(combined))
                # Get the correlations between the unlagged series at both locations
                if i == 0:
                    link_str_array[:, :, args.lag_months] = cov_mat[2*n : 2*n + m, 0 : n]
                # Positive: between lagged NOAA and unlagged prec (i.e. prec behind NOAA)
                link_str_array[:, :, args.lag_months + i + 1] = cov_mat[2*n + m : 2*n + 2*m, 0 : n]
                # Negative: between unlagged NOAA and lagged prec (i.e. prec ahead of NOAA)
                link_str_array[:, :, args.lag_months - i - 1] = cov_mat[2*n : 2*n + m, n : 2*n]
            link_str_array, link_str_max_lags = \
                aggregate_lagged_corrs(link_str_array, LINK_STR_METHOD)
            link_str_max_lags -= args.lag_months
            link_str_max_lags = pd.DataFrame(link_str_max_lags.T, index=noaa_series.index,
                columns=prec_series.index)
        else:
            # combined is a list of the NOAA followed by the prec sequences
            combined = [list(l) for l in [
                *np.array(noaa_series),
                *np.array(prec_series),
            ]]
            # cov_mat is (n + m) x (n + m)
            cov_mat = np.abs(np.corrcoef(combined))
            # The target block is m x n
            link_str_array = cov_mat[n : n + m, 0 : n]
            link_str_max_lags = None
        # Return n x m matrices
        return pd.DataFrame(link_str_array.T, index=noaa_series.index,
            columns=prec_series.index), link_str_max_lags

    def prepare_seq_df(noaa_df):
        if args.month:
            noaa_df = noaa_df.loc[:, [c.month == args.month for c in noaa_df.columns]]
        if args.deseasonalise:
            noaa_df = deseasonalise(noaa_df)
        def seq_func(time_series: pd.Series):
            # Places value at current timestamp at end of list
            # Values decrease in time as the sequence goes left
            if args.decadal:
                start_dates = [datetime(2000, 4, 1), datetime(2011, 3, 1)]
                end_dates = [datetime(2011, 4, 1), datetime(2022, 3, 1)]
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
        noaa_df = noaa_df.apply(seq_func, axis=0)
        print(f'Sequences constructed; time elapsed: {datetime.now() - start}')
        return noaa_df

    if Path(prec_seq_file).is_file() and Path(noaa_seq_file).is_file():
        print(f'Reading sequences from pickle files {prec_seq_file}'
            f' and {noaa_seq_file}')
        prec_seq_df: pd.DataFrame = pd.read_pickle(prec_seq_file)
        noaa_seq_df: pd.DataFrame = pd.read_pickle(noaa_seq_file)
    else:
        print(f'Calculating sequences and saving to pickle files {prec_seq_file}'
            f' and {noaa_seq_file}')
        prec_seq_df = prepare_seq_df(prec_df)
        noaa_seq_df = prepare_seq_df(noaa_df)
        prec_seq_df.to_pickle(prec_seq_file)
        noaa_seq_df.to_pickle(noaa_seq_file)

    if args.decadal:
        d1_link_str_df, d1_link_str_max_lags = build_link_str_df(
            noaa_seq_df.loc[noaa_seq_df.index[0]],
            prec_seq_df.loc[prec_seq_df.index[0]])
        d2_link_str_df, d2_link_str_max_lags = build_link_str_df(
            noaa_seq_df.loc[noaa_seq_df.index[1]],
            prec_seq_df.loc[prec_seq_df.index[1]])
        d1_link_str_df_file = f'{BOTH_DATA_DIR}/link_str_corr_{base_links_file}_d1.pkl'
        d2_link_str_df_file = f'{BOTH_DATA_DIR}/link_str_corr_{base_links_file}_d2.pkl'
        d1_link_str_df.to_pickle(d1_link_str_df_file)
        d2_link_str_df.to_pickle(d2_link_str_df_file)
        d1_max_lags_file = f'{BOTH_DATA_DIR}/link_str_corr_{base_links_file}_d1_max_lags.pkl'
        d2_max_lags_file = f'{BOTH_DATA_DIR}/link_str_corr_{base_links_file}_d2_max_lags.pkl'
        if d1_link_str_max_lags is not None:
            d1_link_str_max_lags.to_pickle(d1_max_lags_file)
        if d2_link_str_max_lags is not None:
            d2_link_str_max_lags.to_pickle(d2_max_lags_file)
        print(f'Decadal link strengths calculated and saved to pickle files'
            f' {d1_link_str_df_file} and {d2_link_str_df_file}')
    else:
        for dt in prec_df.index.values:
            try:
                noaa_seq_dt = noaa_seq_df.loc[dt]
                prec_seq_dt = prec_seq_df.loc[dt]
                # Skip unless the sequences based on the specified
                # lookback time are available
                if noaa_seq_dt.isnull().values.any() or prec_seq_dt.isnull().values.any():
                    continue
            except KeyError:
                continue
            links_file = (f'{BOTH_DATA_DIR}/link_str_corr_{base_links_file}')
            links_file += f'_{dt.strftime("%Y")}' if args.month else f'_{dt.strftime("%Y_%m")}'
            links_max_lags = f'{links_file}_max_lags.pkl'
            links_file += '.pkl'
            date_summary = f'{dt.year}, {dt.strftime("%b")}'
            print(f'\n{date_summary}: calculating link strength data...')
            start = datetime.now()
            link_str_df, link_str_max_lags = build_link_str_df(noaa_seq_dt, prec_seq_dt)
            link_str_df.to_pickle(links_file)
            if link_str_max_lags is not None:
                link_str_max_lags.to_pickle(links_max_lags)
            print((f'{date_summary}: link strengths calculated and saved to pickle file'
                f' {links_file}; time elapsed: {datetime.now() - start}'))

if __name__ == '__main__':
    start = datetime.now()
    main()
    print(f'Total time elapsed: {datetime.now() - start}')