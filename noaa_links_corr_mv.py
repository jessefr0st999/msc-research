from datetime import datetime
import argparse
from math import comb
from itertools import combinations
import gc

import pandas as pd
import numpy as np

YEARS = list(range(2000, 2022 + 1))
# LINK_STR_METHOD = None
LINK_STR_METHOD = 'max'
DATA_DIR = 'data/noaa'
OCEANS_GEOJSON = 'oceans.geojson'

def _aggregate_lagged_corrs(array: np.array, method):
    if method == 'max':
        agg_func = lambda x: np.max(x)
    else:
        # Default: as in Ludescher 2014
        agg_func = lambda x: (np.max(x) - np.mean(x)) / np.std(x)
    return np.apply_along_axis(agg_func, 3, array), np.argmax(array, 3)

def build_link_str_df_mv(seq_files, lag_months, decadal=False):
    num_combs = comb(len(seq_files), 2)
    link_str_array = None
    link_str_array_d1 = None
    df1 = None
    df2 = None
    prev_f1 = None
    prev_f2 = None
    for i, (f1, f2) in enumerate(combinations(seq_files, 2)):
        print(f1, f2)
        # Below is important to free up memory for the next iteration
        if f1 != prev_f1:
            del df1
        if f2 != prev_f2:
            del df2
        gc.collect()
        if f1 != prev_f1:
            df1: pd.DataFrame = pd.read_pickle(f'{DATA_DIR}/{f1}')
        if f2 != prev_f2:
            df2: pd.DataFrame = pd.read_pickle(f'{DATA_DIR}/{f2}')

        # Initialise the link strength array on the first iteration
        # Assumption is that all series have the same indices and columns
        n = len(df1.columns)
        if decadal:
            if link_str_array_d1 is None:
                link_str_array_d1 = np.zeros((n, n, num_combs, 1 + 2 * lag_months)) \
                    if lag_months else np.zeros((n, n, num_combs))
                link_str_array_d2 = np.zeros((n, n, num_combs, 1 + 2 * lag_months)) \
                    if lag_months else np.zeros((n, n, num_combs))
            s1_d1 = df1.iloc[0]
            s1_d2 = df1.iloc[1]
            s2_d1 = df2.iloc[0]
            s2_d2 = df2.iloc[1]
            if lag_months:
                # link_str_array consists of 'sheets' of size n x n containing all pairs
                # of (series 1, series 2). The third dimension indexes the series pair.
                # The fourth dimension contains, for a given location pair, the values for each lag.
                for lsa, s1, s2 in [
                    (link_str_array_d1, s1_d1, s2_d1),
                    (link_str_array_d2, s1_d2, s2_d2),
                ]:
                    for i, lag in enumerate(range(-1, -1 - lag_months, -1)):
                        unlagged_s1 = [list(l[lag_months :]) for l in np.array(s1)]
                        unlagged_s2 = [list(l[lag_months :]) for l in np.array(s2)]
                        lagged_s1 = [list(l[lag + lag_months : lag]) for l in np.array(s1)]
                        lagged_s2 = [list(l[lag + lag_months : lag]) for l in np.array(s2)]
                        combined = [*unlagged_s1, *lagged_s1, *unlagged_s2, *lagged_s2]
                        cov_mat = np.abs(np.corrcoef(combined))
                        # Get the correlations between the unlagged series at both locations
                        if i == 0:
                            lsa[:, :, i, lag_months] = cov_mat[2*n : 3*n, 0 : n]
                        # Positive: between lagged NOAA and unlagged prec (i.e. prec behind NOAA)
                        lsa[:, :, i, lag_months + i + 1] = cov_mat[3*n : 4*n, 0 : n]
                        # Negative: between unlagged NOAA and lagged prec (i.e. prec ahead of NOAA)
                        lsa[:, :, i, lag_months - i - 1] = cov_mat[2*n : 3*n, n : 2*n]
            else:
                # combined is a list of the DF1 seq then DF2 seq
                combined = [list(l) for l in [
                    *np.array(s1_d1),
                    *np.array(s1_d2),
                ]]
                # cov_mat is 2n * 2n
                cov_mat = np.abs(np.corrcoef(combined))
                link_str_array_d1[:, :, i] = cov_mat[n : 2*n, 0 : n]
                
                combined = [list(l) for l in [
                    *np.array(s2_d1),
                    *np.array(s2_d2),
                ]]
                cov_mat = np.abs(np.corrcoef(combined))
                link_str_array_d2[:, :, i] = cov_mat[n : 2*n, 0 : n]
        else:
            pass
            # TODO: create pickles for each dt in the first df1/df2 iteration
            # and GC them at the end of the iteration, then take them off the
            # shelf in succeeding iterations when required
            # for dt in df1.index.values:
            #     # TODO: aggregate over lags before datasets
            #     if lag_months:
            #         pass
            #     else:
            #         unlagged = [list(l) for l in np.array(series)]
            #         cov_mat = np.abs(np.corrcoef(unlagged))
            #         np.fill_diagonal(cov_mat, 0)
            #         link_str_array = cov_mat
        print(link_str_array_d1, link_str_array_d1.shape)
        prev_f1 = f1
        prev_f2 = f2
    
    if decadal:
        if lag_months:
            link_str_array_d1, _ = _aggregate_lagged_corrs(
                link_str_array_d1, LINK_STR_METHOD)
            link_str_array_d2, _ = _aggregate_lagged_corrs(
                link_str_array_d2, LINK_STR_METHOD)
        link_str_array_d1 = np.apply_along_axis(np.linalg.norm, 2, link_str_array_d1)
        link_str_array_d2 = np.apply_along_axis(np.linalg.norm, 2, link_str_array_d2)
        link_str_df_d1 = pd.DataFrame(link_str_array_d1, index=s1_d1.index,
            columns=s1_d1.index)
        link_str_df_d2 = pd.DataFrame(link_str_array_d2, index=s1_d1.index,
            columns=s1_d1.index)
        return link_str_df_d1, link_str_df_d2

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

    # All data files should have the same base; TODO: verify this
    base_file = '_'.join([f.split('_')[1] for f in args.seq_files])
    base_file += '_' + '_'.join(args.seq_files[0].split('_')[2:]).split('.')[0]
    base_links_file = f'{base_file}_lag_{args.lag_months}'
    # Attempting to read all DFs in memory at the same time causes the script to freeze
    # Hence, only read in two at a time and free the memory before reading others in

    if not args.non_decadal:
        d1_link_str_df, d2_link_str_df = build_link_str_df_mv(
            args.seq_files, args.lag_months, True)
        d1_link_str_df_file = f'{DATA_DIR}/link_str_corr_{base_links_file}_d1.pkl'
        d2_link_str_df_file = f'{DATA_DIR}/link_str_corr_{base_links_file}_d2.pkl'
        d1_link_str_df.to_pickle(d1_link_str_df_file)
        d2_link_str_df.to_pickle(d2_link_str_df_file)
        print(f'Decadal link strengths calculated and saved to pickle files'
            f' {d1_link_str_df_file} and {d2_link_str_df_file}')
    else:
        pass
        # link_str_dfs = build_link_str_df_mv(args.seq_files, args.lag_months)

if __name__ == '__main__':
    start = datetime.now()
    main()
    print(f'Total time elapsed: {datetime.now() - start}')