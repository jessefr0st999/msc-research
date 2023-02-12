from datetime import datetime
import argparse
from math import comb
from itertools import combinations
import gc

import pandas as pd
import numpy as np

from links_corr import aggregate_lagged_corrs

YEARS = list(range(2000, 2022 + 1))
# LINK_STR_METHOD = None
LINK_STR_METHOD = 'max'
DATA_DIR = 'data/noaa'
OCEANS_GEOJSON = 'oceans.geojson'

def build_link_str_df_mv(series_1: pd.Series, series_2: pd.Series, lag_months):
    # TODO: NOAA and prec dataframes/series in the same format, regardless of lag
    for s in series_1, series_2:
        if isinstance(s, pd.DataFrame):
            s = pd.Series(
                s['prec_seq'].values,
                index=pd.MultiIndex.from_tuples(
                    list(zip(*[s['lat'].values, s['lon'].values])),
                    names=['Lat', 'Lon']
                ),
            )
    n = len(series_1)
    m = len(series_2)
    if lag_months:
        # link_str_array consists of 'sheets' of size m x n containing all pairs
        # of (series 1 location, series 2 location). The third dimension contains, for
        # a given location pair, the values for each lag
        link_str_array = np.zeros((m, n, 1 + 2 * lag_months))
        for i, lag in enumerate(range(-1, -1 - lag_months, -1)):
            unlagged_noaa = [list(l[lag_months :]) for l in np.array(series_1)]
            unlagged_prec = [list(l[lag_months :]) for l in np.array(series_2)]
            lagged_noaa = [list(l[lag + lag_months : lag]) for l in np.array(series_1)]
            lagged_prec = [list(l[lag + lag_months : lag]) for l in np.array(series_2)]
            combined = [*unlagged_noaa, *lagged_noaa, *unlagged_prec, *lagged_prec]
            # TODO: do this more efficiently
            cov_mat = np.abs(np.corrcoef(combined))
            # Get the correlations between the unlagged series at both locations
            if i == 0:
                link_str_array[:, :, lag_months] = cov_mat[2*n : 2*n + m, 0 : n]
            # Positive: between lagged series 1 and unlagged series 2
            # (i.e. series 2 behind series 1)
            link_str_array[:, :, lag_months + i + 1] = cov_mat[2*n + m : 2*n + 2*m, 0 : n]
            # Negative: between unlagged series 1 and lagged series 2
            # (i.e. series 2 ahead of series 1)
            link_str_array[:, :, lag_months - i - 1] = cov_mat[2*n : 2*n + m, n : 2*n]
        link_str_array, link_str_max_lags = \
            aggregate_lagged_corrs(link_str_array, LINK_STR_METHOD)
        link_str_max_lags -= lag_months
        link_str_max_lags = pd.DataFrame(link_str_max_lags.T, index=series_1.index,
            columns=series_2.index)
    else:
        # combined is a list of the series 1 followed by the series 2 sequences
        combined = [list(l) for l in [
            *np.array(series_1),
            *np.array(series_2),
        ]]
        # cov_mat is (n + m) x (n + m)
        cov_mat = np.abs(np.corrcoef(combined))
        # The target block is m x n
        link_str_array = cov_mat[n : n + m, 0 : n]
        link_str_max_lags = None
    # Return n x m matrices
    return pd.DataFrame(link_str_array.T, index=series_1.index,
        columns=series_2.index), link_str_max_lags

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_files', nargs='*', default=[
        'seq_slp_decadal.pkl',
        'seq_temp_decadal.pkl',
        'seq_humidity_decadal.pkl',
        'seq_omega_decadal.pkl',
        'seq_pw_decadal.pkl',
    ])
    # Args below should conform to those in the files above
    parser.add_argument('--lag_months', '--lag', type=int, default=0)
    parser.add_argument('--non_decadal', action='store_true', default=False)
    parser.add_argument('--month', type=int, default=None) # TODO
    args = parser.parse_args()

    # Attempting to read all DFs in memory at the same time causes the script to freeze
    # Hence, only read in two at a time and free the memory before reading others in
    df1 = None
    df2 = None
    prev_f1 = None
    prev_f2 = None
    for i, (f1, f2) in enumerate(combinations(args.seq_files, 2)):
        if f1 != prev_f1:
            del df1
        if f2 != prev_f2:
            del df2
        gc.collect()
        if f1 != prev_f1:
            df1: pd.DataFrame = pd.read_pickle(f'{DATA_DIR}/{f1}')
        if f2 != prev_f2:
            df2: pd.DataFrame = pd.read_pickle(f'{DATA_DIR}/{f2}')
        base_file = '_'.join([f.split('_')[1] for f in [f1, f2]])
        base_file += '_' + '_'.join(args.seq_files[0].split('_')[2:]).split('.')[0]
        base_links_file = f'{base_file}_lag_{args.lag_months}'

        if not args.non_decadal:
            d1_link_str_df, d1_link_str_max_lags = build_link_str_df_mv(
                df1.loc[df1.index[0]], df2.loc[df2.index[0]], args.lag_months)
            d2_link_str_df, d2_link_str_max_lags = build_link_str_df_mv(
                df1.loc[df1.index[1]], df2.loc[df2.index[1]], args.lag_months)
            d1_link_str_df_file = f'{DATA_DIR}/link_str_corr_{base_links_file}_d1.pkl'
            d2_link_str_df_file = f'{DATA_DIR}/link_str_corr_{base_links_file}_d2.pkl'
            d1_link_str_df.to_pickle(d1_link_str_df_file)
            d2_link_str_df.to_pickle(d2_link_str_df_file)
            d1_max_lags_file = f'{DATA_DIR}/link_str_corr_{base_links_file}_d1_max_lags.pkl'
            d2_max_lags_file = f'{DATA_DIR}/link_str_corr_{base_links_file}_d2_max_lags.pkl'
            if d1_link_str_max_lags is not None:
                d1_link_str_max_lags.to_pickle(d1_max_lags_file)
            if d2_link_str_max_lags is not None:
                d2_link_str_max_lags.to_pickle(d2_max_lags_file)
            print(f'Decadal link strengths calculated and saved to pickle files'
                f' {d1_link_str_df_file} and {d2_link_str_df_file}')
        else:
            for dt in df1.index.values:
                _dt = pd.to_datetime(dt)
                try:
                    df1_dt = df1.loc[dt]
                    df2_dt = df2.loc[dt]
                    # Skip unless the sequences based on the specified
                    # lookback time are available
                    if df1_dt.isnull().values.any() or df2_dt.isnull().values.any():
                        continue
                except KeyError:
                    continue
                links_file = (f'{DATA_DIR}/link_str_corr_{base_links_file}')
                links_file += f'_{_dt.strftime("%Y")}' if args.month else f'_{_dt.strftime("%Y_%m")}'
                links_max_lags = f'{links_file}_max_lags.pkl'
                links_file += '.pkl'
                date_summary = f'{_dt.year}, {_dt.strftime("%b")}'
                print(f'\n{date_summary}: calculating link strength data...')
                start = datetime.now()
                link_str_df, link_str_max_lags = build_link_str_df_mv(df1_dt, df2_dt, args.lag_months)
                link_str_df.to_pickle(links_file)
                if link_str_max_lags is not None:
                    link_str_max_lags.to_pickle(links_max_lags)
                print((f'{date_summary}: link strengths calculated and saved to pickle file'
                    f' {links_file}; time elapsed: {datetime.now() - start}'))

if __name__ == '__main__':
    start = datetime.now()
    main()
    print(f'Total time elapsed: {datetime.now() - start}')