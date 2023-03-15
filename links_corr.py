from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
import argparse

import pandas as pd
import numpy as np
# from geopy.distance import geodesic

YEARS = list(range(2000, 2022 + 1))
# LINK_STR_METHOD = None
LINK_STR_METHOD = 'max'
DATA_DIR = 'data/precipitation'
PREC_FILE = f'{DATA_DIR}/FusedData.csv'
LOCATIONS_FILE = f'{DATA_DIR}/Fused.Locations.csv'
GEO_AGG_PREC_FILE = f'{DATA_DIR}/prec_df_agg.pkl'

def aggregate_lagged_corrs(array: np.array, method):
    if method == 'max':
        agg_func = lambda x: np.max(x)
    else:
        # Default: as in Ludescher 2014
        agg_func = lambda x: (np.max(x) - np.mean(x)) / np.std(x)
    return np.apply_along_axis(agg_func, 2, array), np.argmax(array, 2)

def build_link_str_df_uv(series: pd.Series, lag_months):
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
        cov_mat = np.abs(np.corrcoef(unlagged))
        np.fill_diagonal(cov_mat, 0)
        link_str_array = cov_mat
        link_str_max_lags = None
    # Return n x n matrices
    return pd.DataFrame(link_str_array, index=series.index, columns=series.index), \
        link_str_max_lags

def main():
    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument('--avg_lookback_months', '--alm', type=int, default=60)
    parser.add_argument('--lag_months', '--lag', type=int, default=0)
    parser.add_argument('--link_str_geo_penalty', type=float, default=0)
    parser.add_argument('--deseasonalise', action='store_true', default=False)
    parser.add_argument('--decadal', action='store_true', default=False)
    parser.add_argument('--yearly', action='store_true', default=False)
    parser.add_argument('--geo_agg', action='store_true', default=False)
    parser.add_argument('--month', type=int, default=None)
    # NOTE: This should not be used with lag
    parser.add_argument('--exp_kernel', type=float, default=None)

    # Input/output controls
    parser.add_argument('--file_tag', default='')

    args = parser.parse_args()

    base_file = '_decadal' if args.decadal else f'_alm_{args.avg_lookback_months}'
    if args.file_tag:
        base_file = f'{args.file_tag}{base_file}'
    if args.month:
        month_str = str(args.month) if args.month >= 10 else f'0{args.month}'
        base_file += f'_m{month_str}'
    base_file += '_geo_agg' if args.geo_agg else ''
    base_file += '_des' if args.deseasonalise else ''
    base_links_file = f'{base_file}_lag_{args.lag_months}'
    if not args.decadal:
        base_file += f'_lag_{args.lag_months}'

    prec_seq_file = f'{DATA_DIR}/seq_prec{base_file}.pkl'
    months = [args.month] if args.month else list(range(1, 13))

    def deseasonalise(df):
        def row_func(row: pd.Series):
            datetime_index = pd.DatetimeIndex(row.index)
            for m in months:
                month_series = row.loc[datetime_index.month == m]
                row.loc[datetime_index.month == m] = \
                    (month_series.values - month_series.mean()) / month_series.std()
            return row
        return df.apply(row_func, axis=1)

    def exp_kernel(seq, k):
        # k is the factor by which the start of the sequence is multiplied
        t_seq = pd.Series(list(range(len(seq))))
        exp_seq = np.exp(np.log(k) / (len(seq) - 1) * t_seq)
        # Reverse the sequence
        exp_seq = exp_seq[::-1]
        return seq * exp_seq

    def prepare_prec_df():
        if args.geo_agg:
            df = pd.read_pickle(GEO_AGG_PREC_FILE).T
        else:
            raw_df = pd.read_csv(PREC_FILE)
            raw_df.columns = pd.to_datetime(raw_df.columns, format='D%Y.%m')
            locations_df = pd.read_csv(LOCATIONS_FILE)
            df = pd.concat([locations_df, raw_df], axis=1)
            df = df.set_index(['Lat', 'Lon'])
        if args.month:
            df = df.loc[:, [c.month == args.month for c in df.columns]]
        if args.deseasonalise:
            df = deseasonalise(df)
        def seq_func(row: pd.Series):
            # Places value at current timestamp at end of list
            # Values decrease in time as the sequence goes left
            if args.decadal:
                start_dates = [datetime(2000, 4, 1), datetime(2011, 3, 1)]
                end_dates = [datetime(2011, 4, 1), datetime(2022, 3, 1)]
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
                    seq_length = args.avg_lookback_months // 12 if args.month else \
                        args.avg_lookback_months + args.lag_months
                    if len(sequence) != seq_length:
                        return None
                    if args.exp_kernel:
                        sequence = exp_kernel(sequence, args.exp_kernel)
                    return sequence
                vec_func = np.vectorize(func, excluded=['r'])
                seq_lookback = args.avg_lookback_months if args.month else \
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
        df = df.stack().reset_index()
        df = df.rename(columns={'level_2': 'date', 0: 'prec_seq', 'Lat': 'lat', 'Lon': 'lon'})
        df = df.set_index('date')
        return df

    if Path(prec_seq_file).is_file():
        print(f'Reading precipitation sequences from pickle file {prec_seq_file}')
        prec_df: pd.DataFrame = pd.read_pickle(prec_seq_file)
    else:
        print(f'Calculating precipitation sequences and saving to pickle file {prec_seq_file}')
        prec_df = prepare_prec_df()
        prec_df.to_pickle(prec_seq_file)

    if args.decadal:
        d1_link_str_df, d1_max_lags = build_link_str_df_uv(prec_df.loc[prec_df.index[0]], args.lag_months)
        d2_link_str_df, d2_max_lags = build_link_str_df_uv(prec_df.loc[prec_df.index[1]], args.lag_months)
        d1_link_str_df_file = f'{DATA_DIR}/link_str_corr{base_links_file}_d1'
        d2_link_str_df_file = f'{DATA_DIR}/link_str_corr{base_links_file}_d2'
        d1_link_str_df.to_csv(f'{d1_link_str_df_file}.csv')
        d2_link_str_df.to_csv(f'{d2_link_str_df_file}.csv')
        print(f'Decadal link strengths calculated and saved to CSV files'
            f' {d1_link_str_df_file}.csv and {d2_link_str_df_file}.csv')
        if d1_max_lags is not None:
            d1_max_lags.to_pickle(f'{d1_link_str_df_file}_max_lags.pkl')
            d2_max_lags.to_pickle(f'{d2_link_str_df_file}_max_lags.pkl')
    else:
        for y in YEARS:
            if y != 2022:
                continue
            for m in months:
                # Always use March as the reference month for the "yearly" networks
                if args.yearly and m != 3:
                    continue
                dt = datetime(y, m, 1)
                try:
                    prec_dt = prec_df.loc[dt]
                    # Skip unless the sequence based on the specified lookback time is available
                    if prec_dt.isnull().values.any():
                        continue
                except KeyError:
                    continue
                links_file = (f'{DATA_DIR}/link_str_corr{base_links_file}')
                if args.link_str_geo_penalty:
                    links_file += f'_geo_pen_{str(int(1 / args.link_str_geo_penalty))}'
                links_file += f'_{dt.strftime("%Y")}' if args.month else f'_{dt.strftime("%Y_%m")}'
                date_summary = f'{dt.year}, {dt.strftime("%b")}'
                print(f'\n{date_summary}: calculating link strength data...')
                start = datetime.now()
                link_str_df, max_lags = build_link_str_df_uv(prec_dt, args.lag_months)
                print((f'{date_summary}: link strengths calculated and saved to CSV file'
                    f' {links_file}.csv; time elapsed: {datetime.now() - start}'))
                link_str_df.to_csv(f'{links_file}.csv')
                if max_lags is not None:
                    max_lags.to_pickle(f'{links_file}_max_lags.pkl')

if __name__ == '__main__':
    start = datetime.now()
    main()
    print(f'Total time elapsed: {datetime.now() - start}')