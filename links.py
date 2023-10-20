from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
import argparse
from itertools import combinations

import pandas as pd
import numpy as np
# from geopy.distance import geodesic
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

YEARS = list(range(2000, 2022 + 1))
# LINK_STR_METHOD = None
LINK_STR_METHOD = 'max'
DATA_DIR = 'data/precipitation'
PREC_FILE = f'{DATA_DIR}/FusedData.csv'
LOCATIONS_FILE = f'{DATA_DIR}/Fused.Locations.csv'

def aggregate_lagged_corrs(array: np.array, method):
    if method == 'max':
        agg_func = lambda x: np.max(x)
    else:
        # Default: as in Ludescher 2014
        agg_func = lambda x: (np.max(x) - np.mean(x)) / np.std(x)
    return np.apply_along_axis(agg_func, 2, array), np.argmax(array, 2)

def build_link_str_df_uv(series: pd.Series, lag_months=None, method='pearson'):
    # TODO: NOAA and prec dataframes/series in the same format, regardless of lag
    if isinstance(series, pd.DataFrame):
        series = pd.Series(
            series['prec_seq'].values,
            index=pd.MultiIndex.from_tuples(
                list(zip(*[series['lat'].values, series['lon'].values])),
                names=['Lat', 'Lon']
            ),
        )
    n = len(series)
    if lag_months:
        # NOTE: Only implemented for 'corr' (Pearson correlation) method
        # link_str_array consists of 'sheets' of size n x n containing all pairs
        # of (prec location, prec location). The third dimension contains, for
        # a given location pair, the values for each lag
        link_str_array = np.zeros((n, n, 1 + 2 * lag_months))
        for i, lag in enumerate(range(-1, -1 - lag_months, -1)):
            unlagged = [list(seq[lag_months :]) for seq in np.array(series)]
            lagged = [list(seq[lag + lag_months : lag]) for seq in np.array(series)]
            combined = [*unlagged, *lagged]
            cov_mat = np.abs(np.corrcoef(combined))
            # Don't want self-joined nodes; set diagonal values to 0
            np.fill_diagonal(cov_mat, 0)
            # Get the correlations between the unlagged series at both locations
            if i == 0:
                link_str_array[:, :, lag_months] = cov_mat[0 : n, 0 : n]
            # Positive: between lagged loc 1 and unlagged loc 2 (i.e. loc 2 behind loc 1)
            link_str_array[:, :, lag_months + i + 1] = cov_mat[n : 2*n, 0 : n]
            # Negative: between unlagged loc 1 and lagged loc 2 (i.e. loc 2 ahead of loc 1)
            link_str_array[:, :, lag_months - i - 1] = cov_mat[0 : n, n : 2*n]
        link_str_array, link_str_max_lags = \
            aggregate_lagged_corrs(link_str_array, LINK_STR_METHOD)
        link_str_max_lags -= lag_months
        link_str_max_lags = pd.DataFrame(link_str_max_lags.T, index=series.index,
            columns=series.index)
    else:
        unlagged = [list(seq) for seq in np.array(series)]
        # TODO: speed up regression
        if method == 'reg':
            # Absolute value of linear regression coefficient
            link_str_array = np.zeros((len(unlagged), len(unlagged)))
            for i, j in combinations(range(len(unlagged)), 2):
                reg_gradient_1 = LinearRegression().fit(
                    np.column_stack([unlagged[i]]), unlagged[j]).coef_[0]
                reg_gradient_2 = LinearRegression().fit(
                    np.column_stack([unlagged[j]]), unlagged[i]).coef_[0]
                value = np.min(np.abs([reg_gradient_1, reg_gradient_2]))
                link_str_array[i, j] = link_str_array[j, i] = value
            print(f'Total time elapsed: {datetime.now() - start}')
        elif method == 'spearman':
            # Spearman correlation coefficient
            link_str_array = np.abs(spearmanr(np.array(unlagged).T).statistic)
        elif method == 'pearson':
            # Absolute value of Pearson correlation
            link_str_array = np.abs(np.corrcoef(unlagged))
        else:
            raise ValueError(f'Unknown link strength method: "{method}"')
        np.fill_diagonal(link_str_array, 0)
        link_str_max_lags = None
    # Return n x n matrices
    return pd.DataFrame(link_str_array, index=series.index, columns=series.index), \
        link_str_max_lags

def main():
    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument('--method', default='pearson')
    parser.add_argument('--avg_lookback_months', '--alm', type=int, default=60)
    parser.add_argument('--lag_months', '--lag', type=int, default=0)
    # Build networks for each decade, with lookback over all months in the decade
    parser.add_argument('--decadal', action='store_true', default=False)
    # Build networks timestamped only from a given month (generally March)
    parser.add_argument('--month', type=int, default=None)
    # Build networks for a given month, comparing only that month in other years
    parser.add_argument('--month_only', type=int, default=None)
    # Build networks for each decade, correlating each calendar month with that of prior years
    parser.add_argument('--dms', action='store_true', default=False)
    parser.add_argument('--season', default=None)
    # NOTE: This should not be used with lag
    parser.add_argument('--exp_kernel', type=float, default=None)
    args = parser.parse_args()

    if args.decadal:
        base_file = 'decadal'
    elif args.dms:
        base_file = 'dms'
    else:
        base_file = f'alm_{args.avg_lookback_months}'
    if args.month_only:
        month_str = str(args.month_only) if args.month_only >= 10 else f'0{args.month_only}'
        base_file += f'_m{month_str}'
    base_links_file = base_file
    if args.dms and args.season:
        base_links_file += f'_{args.season}'
    if args.method != 'pearson':
        base_links_file += f'_{args.method}'
    base_links_file += f'_lag_{args.lag_months}'
    if not (args.decadal or args.dms):
        base_file += f'_lag_{args.lag_months}'
    prec_seq_file = f'{DATA_DIR}/seq_prec_{base_file}.pkl'

    def exp_kernel(seq, k):
        # k is the factor by which the start of the sequence is multiplied
        t_seq = pd.Series(list(range(len(seq))))
        exp_seq = np.exp(np.log(k) / (len(seq) - 1) * t_seq)
        # Reverse the sequence
        exp_seq = exp_seq[::-1]
        return seq * exp_seq

    def prepare_prec_df():
        raw_df = pd.read_csv(PREC_FILE)
        raw_df.columns = pd.to_datetime(raw_df.columns, format='D%Y.%m')
        locations_df = pd.read_csv(LOCATIONS_FILE)
        df = pd.concat([locations_df, raw_df], axis=1)
        df = df.set_index(['Lat', 'Lon'])
        if args.month_only:
            df = df.loc[:, [c.month == args.month_only for c in df.columns]]
        def seq_func(row: pd.Series):
            # Place the value at the current timestamp at the start of list
            # Values hence decrease in time as the sequence progresses
            if args.decadal or args.dms:
                start_dates = [datetime(2000, 4, 1), datetime(2011, 4, 1)]
                end_dates = [datetime(2011, 3, 1), datetime(2022, 3, 1)]
                def func(start_date, end_date, r: pd.Series):
                    sequence = list(r.loc[start_date : end_date])
                    sequence.reverse()
                    return sequence
                _row = pd.Series([
                    func(start_end[0], start_end[1], row) \
                        for start_end in zip(start_dates, end_dates)
                ], index=end_dates)
            else:
                def func(start_date, end_date, r: pd.Series):
                    sequence = list(r[(r.index > start_date) & (r.index <= end_date)])
                    seq_length = args.avg_lookback_months // 12 if args.month_only else \
                        args.avg_lookback_months + args.lag_months
                    if len(sequence) != seq_length:
                        return None
                    if args.exp_kernel:
                        sequence = exp_kernel(sequence, args.exp_kernel)
                    return sequence
                vec_func = np.vectorize(func, excluded=['r'])
                seq_lookback = args.avg_lookback_months if args.month_only else \
                    args.avg_lookback_months + args.lag_months
                start_dates = [d - relativedelta(months=seq_lookback) for d in row.index]
                end_dates = [d for d in row.index]
                _row = vec_func(start_date=start_dates, end_date=end_dates, r=row)
                _row = pd.Series(_row, index=row.index)
            return _row
        start = datetime.now()
        print('Constructing sequences...')
        df = df.apply(seq_func, axis=1)       
        print(f'Sequences constructed; time elapsed: {datetime.now() - start}')
        return df.T
    prec_df = prepare_prec_df()

    if args.decadal or args.dms:
        d1_series = prec_df.loc[prec_df.index[0]]
        d2_series = prec_df.loc[prec_df.index[1]]
        # TODO: change seasonal to only calculate for the given season; this will
        # heavily optimise the calculation for the regression gradient method
        if args.dms:
            def _agg(array):
                if args.season == 'summer':
                    # Trick to get the entries with indices 0, 1, 11 in once slice
                    joined_array = np.concatenate((array, array), axis=2)
                    return np.nanmean(joined_array[:, :, 11 : 14], axis=2)
                if args.season == 'autumn':
                    return np.nanmean(array[:, :, 2 : 5], axis=2)
                if args.season == 'winter':
                    return np.nanmean(array[:, :, 5 : 8], axis=2)
                if args.season == 'spring':
                    return np.nanmean(array[:, :, 8 : 11], axis=2)
                return np.nanmean(array, axis=2)
                # NaN-proof Euclidean norm
                # return np.sqrt(np.nansum(np.square(array), axis=2))
            d1_link_str_array = np.zeros((len(d1_series.index), len(d1_series.index), 12))
            d2_link_str_array = np.zeros((len(d2_series.index), len(d2_series.index), 12))
            for m in range(1, 12 + 1):
                indices = []
                for i in range(len(d1_series.iloc[0])):
                    # Each decadal series ends in March (first element of its sequence)
                    if i % 12 == (m - 3) % 12:
                        indices.append(i)
                d1_series_m = d1_series.apply(lambda seq: np.array(seq)[indices])
                d2_series_m = d2_series.apply(lambda seq: np.array(seq)[indices])
                d1_link_str_df_m, d1_max_lags = build_link_str_df_uv(d1_series_m,
                    method=args.method)
                d2_link_str_df_m, d2_max_lags = build_link_str_df_uv(d2_series_m,
                    method=args.method)
                d1_link_str_array[:, :, m - 1] = d1_link_str_df_m
                d2_link_str_array[:, :, m - 1] = d2_link_str_df_m
            d1_link_str_df = pd.DataFrame(_agg(d1_link_str_array),
                index=d1_series.index, columns=d1_series.index)
            d2_link_str_df = pd.DataFrame(_agg(d2_link_str_array),
                index=d2_series.index, columns=d2_series.index)
        else:
            d1_link_str_df, d1_max_lags = build_link_str_df_uv(d1_series,
                args.lag_months, args.method)
            d2_link_str_df, d2_max_lags = build_link_str_df_uv(d2_series,
                args.lag_months, args.method)
        d1_link_str_df_file = f'{DATA_DIR}/link_str_corr_{base_links_file}_d1'
        d2_link_str_df_file = f'{DATA_DIR}/link_str_corr_{base_links_file}_d2'
        d1_link_str_df.to_csv(f'{d1_link_str_df_file}.csv')
        d2_link_str_df.to_csv(f'{d2_link_str_df_file}.csv')
        print(f'Decadal link strengths calculated and saved to CSV files'
            f' {d1_link_str_df_file}.csv and {d2_link_str_df_file}.csv')
        if d1_max_lags is not None:
            d1_max_lags.to_pickle(f'{d1_link_str_df_file}_max_lags.pkl')
            d2_max_lags.to_pickle(f'{d2_link_str_df_file}_max_lags.pkl')
    else:
        for y in YEARS:
            for m in range(1, 12 + 1):
                _month = args.month or args.month_only
                if _month and m != _month:
                    continue
                dt = datetime(y, m, 1)
                try:
                    prec_dt = prec_df.loc[dt]
                    # Skip unless the sequence based on the specified lookback time is available
                    if prec_dt.isnull().values.any():
                        continue
                except KeyError:
                    continue
                links_file = (f'{DATA_DIR}/link_str_corr_{base_links_file}')
                links_file += f'_{dt.strftime("%Y")}' if args.month else f'_{dt.strftime("%Y_%m")}'
                date_summary = f'{dt.year}, {dt.strftime("%b")}'
                print(f'\n{date_summary}: calculating link strength data...')
                start = datetime.now()
                link_str_df, max_lags = build_link_str_df_uv(prec_dt, args.lag_months, args.method)
                print((f'{date_summary}: link strengths calculated and saved to CSV file'
                    f' {links_file}.csv; time elapsed: {datetime.now() - start}'))
                link_str_df.to_csv(f'{links_file}.csv')
                if max_lags is not None:
                    max_lags.to_pickle(f'{links_file}_max_lags.pkl')

if __name__ == '__main__':
    start = datetime.now()
    main()
    print(f'Total time elapsed: {datetime.now() - start}')