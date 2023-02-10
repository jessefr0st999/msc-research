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

def main():
    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument('--avg_lookback_months', '--alm', type=int, default=60)
    parser.add_argument('--lag_months', '--lag', type=int, default=0)
    parser.add_argument('--link_str_geo_penalty', type=float, default=0)
    parser.add_argument('--no_anti_corr', '--nac', action='store_true', default=False)
    parser.add_argument('--deseasonalise', action='store_true', default=False)
    parser.add_argument('--decadal', action='store_true', default=False)
    parser.add_argument('--geo_agg', action='store_true', default=False)
    parser.add_argument('--month', type=int, default=None)
    # NOTE: This should not be used with lag
    parser.add_argument('--exp_kernel', type=float, default=None)

    # Input/output controls
    parser.add_argument('--dt_limit', type=int, default=0)
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

    def exp_kernel(seq, k):
        # k is the factor by which the start of the sequence is multiplied
        t_seq = pd.Series(list(range(len(seq))))
        exp_seq = np.exp(np.log(k) / (len(seq) - 1) * t_seq)
        # Reverse the sequence
        exp_seq = exp_seq[::-1]
        return seq * exp_seq

    def calculate_link_str(df: pd.DataFrame, method) -> pd.Series:
        if method == 'max':
            return df.max(axis=1)
        # Default: as in Ludescher 2014
        return (df.max(axis=1) - df.mean(axis=1)) / df.std(axis=1)

    def build_link_str_df(df: pd.DataFrame):
        # TODO: rework this as per NOAA-prec script
        link_str_df_kwargs = dict(index=[df['lat'], df['lon']],
            columns=[df['lat'], df['lon']])
        slice_str_df_kwargs = dict(index=[f'{r["lat"]}_{r["lon"]}_1' for i, r in df.iterrows()],
            columns=[f'{r["lat"]}_{r["lon"]}_2' for i, r in df.iterrows()])
        if args.lag_months:
            link_str_df = pd.DataFrame(pd.DataFrame(**slice_str_df_kwargs).unstack())
            n = len(df.index)
            for i, lag in enumerate(range(-1, -1 - args.lag_months, -1)):
                unlagged = [list(l[args.lag_months :]) for l in np.array(df['prec_seq'])]
                lagged = [list(l[lag + args.lag_months : lag]) for l in np.array(df['prec_seq'])]
                combined = [*unlagged, *lagged]
                if args.no_anti_corr:
                    cov_mat = np.corrcoef(combined)
                    cov_mat[cov_mat < 0] = 0
                else:
                    cov_mat = np.abs(np.corrcoef(combined))
                # Don't want self-joined nodes; set diagonal values to 0
                np.fill_diagonal(cov_mat, 0)
                # Get the correlations between the unlagged series at both locations
                if i == 0:
                    loc_1_unlag_loc_2_unlag_slice = cov_mat[0 : n, 0 : n]
                    slice_df = pd.DataFrame(loc_1_unlag_loc_2_unlag_slice, **slice_str_df_kwargs).unstack()
                    link_str_df['s1_lag_0_s2_lag_0'] = slice_df
                # Between lagged series at location 1 and unlagged series at location 2
                loc_1_lag_loc_2_unlag_slice = cov_mat[n : 2 * n, 0 : n]
                slice_df = pd.DataFrame(loc_1_lag_loc_2_unlag_slice, **slice_str_df_kwargs).unstack()
                link_str_df[f's1_lag_{-lag}_s2_lag_0'] = slice_df
                # Between unlagged series at location 1 and lagged series at location 2
                loc_1_unlag_loc_2_lag_slice = cov_mat[0 : n, n : 2 * n]
                slice_df = pd.DataFrame(loc_1_unlag_loc_2_lag_slice, **slice_str_df_kwargs).unstack()
                link_str_df[f's1_lag_0_s2_lag_{-lag}'] = slice_df
            link_str_df = link_str_df.drop(columns=[0])
            link_str_df = calculate_link_str(link_str_df, LINK_STR_METHOD)
            link_str_df = link_str_df.unstack()
            link_str_df = pd.DataFrame(np.array(link_str_df), **link_str_df_kwargs)
        else:
            unlagged = [list(l) for l in np.array(df['prec_seq'])]
            if args.no_anti_corr:
                cov_mat = np.corrcoef(unlagged)
                cov_mat[cov_mat < 0] = 0
            else:
                cov_mat = np.abs(np.corrcoef(unlagged))
            np.fill_diagonal(cov_mat, 0)
            link_str_df = pd.DataFrame(cov_mat, **link_str_df_kwargs)
        # TODO: reimplement link_str_geo_penalty between pairs of points
        return link_str_df

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
        d1_link_str_df = build_link_str_df(prec_df.loc[prec_df.index[0]])
        d2_link_str_df = build_link_str_df(prec_df.loc[prec_df.index[1]])
        d1_link_str_df_file = f'{DATA_DIR}/link_str_corr{base_links_file}_d1.csv'
        d2_link_str_df_file = f'{DATA_DIR}/link_str_corr{base_links_file}_d2.csv'
        d1_link_str_df.to_csv(d1_link_str_df_file)
        d2_link_str_df.to_csv(d2_link_str_df_file)
        print(f'Decadal link strengths calculated and saved to CSV files'
            f' {d1_link_str_df_file} and {d2_link_str_df_file}')
    else:
        analysed_dt_count = 0
        for y in YEARS:
            for m in months:
                if args.dt_limit and analysed_dt_count == args.dt_limit:
                    break
                dt = datetime(y, m, 1)
                try:
                    prec_dt = prec_df.loc[dt]
                    # Skip unless the sequence based on the specified lookback time is available
                    if prec_dt['prec_seq'].isnull().values.any():
                        continue
                except KeyError:
                    continue
                analysed_dt_count += 1
                links_file = (f'{DATA_DIR}/link_str_corr{base_links_file}')
                if args.link_str_geo_penalty:
                    links_file += f'_geo_pen_{str(int(1 / args.link_str_geo_penalty))}'
                links_file += f'_{dt.strftime("%Y")}' if args.month else f'_{dt.strftime("%Y_%m")}'
                links_file += '.csv'
                date_summary = f'{dt.year}, {dt.strftime("%b")}'
                print(f'\n{date_summary}: calculating link strength data...')
                start = datetime.now()
                link_str_df = build_link_str_df(prec_dt)
                print((f'{date_summary}: link strengths calculated and saved to CSV file'
                    f' {links_file}; time elapsed: {datetime.now() - start}'))
                link_str_df.to_csv(links_file)

if __name__ == '__main__':
    start = datetime.now()
    main()
    print(f'Total time elapsed: {datetime.now() - start}')