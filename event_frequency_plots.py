import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from helpers import get_map

YEARS = range(2010, 2022)
DATA_DIR = 'data/precipitation'
DATA_FILE = f'{DATA_DIR}/FusedData.csv'
LOCATIONS_FILE = f'{DATA_DIR}/Fused.Locations.csv'
size_func = lambda series: [500 * n for n in series]
cmap = 'RdYlGn_r'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', default=None)
    parser.add_argument('--month', type=int, default=None)
    parser.add_argument('--col_quantile', '--cq', type=float, default=None)
    parser.add_argument('--df_quantile', '--dq', type=float, default=0.95)
    parser.add_argument('--lookback_months', '--lm', type=int, default=120)
    args = parser.parse_args()

    raw_df = pd.read_csv(DATA_FILE)
    raw_df.columns = pd.to_datetime(raw_df.columns, format='D%Y.%m')
    locations_df = pd.read_csv(LOCATIONS_FILE)
    df = pd.concat([locations_df, raw_df], axis=1)
    df = df.set_index(['Lat', 'Lon'])
    df = df.T

    pos = None
    if args.month:
        months = [args.month]
    else:
        months = range(1, 13)
    for y in YEARS:
        for m in months:
            dt = datetime(y, m, 1)
            end_months = y * 12 + m
            _df = df.loc[datetime((end_months - args.lookback_months) // 12,
                    (end_months  - args.lookback_months) % 12 + 1, 1)
                : dt]

            if _df.empty:
                continue

            dt_df = pd.DataFrame(0, columns=_df.columns, index=_df.index)
            dt_df[_df > np.quantile(_df, args.df_quantile)] = 1
            print(f'Number of extreme events over whole dataframe: {dt_df.to_numpy().sum()}')
            if args.col_quantile:
                extreme_column_events_df = pd.DataFrame(0, columns=_df.columns, index=_df.index)
                for c in _df.columns:
                    extreme_column_events_df[c][_df[c] > np.quantile(_df[c], args.col_quantile)] = 1
                print(f'Number of extreme events over columns: {extreme_column_events_df.to_numpy().sum()}')
                dt_df &= extreme_column_events_df
                print(f'Number of merged extreme events: {dt_df.to_numpy().sum()}')

            figure, axis = plt.subplots(1)
            _map = get_map(axis)
            if pos is None:
                locations = dt_df.columns
                lats, lons = zip(*locations.values)
                mx, my = _map(lons, lats)
                pos = {elem: (mx[i], my[i]) for i, elem in enumerate(locations)}

            series = dt_df.sum(axis=0)
            axis.scatter(
                mx, my, c=series, cmap=cmap,
                s=size_func((series - np.min(series)) / np.max(series)),
            )
            norm = mpl.colors.Normalize(vmin=np.min(series), vmax=np.max(series))
            plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis)
            plt.tight_layout()
            axis.set_title(f'{dt.strftime("%b")} {y}: event frequency', fontsize=20)
            if args.output_folder:
                month_str = str(m) if m >= 10 else f'0{m}'
                filename = f'event_frequency_{y}_{month_str}.png'
                figure.set_size_inches(32, 18)
                plt.savefig(f'{args.output_folder}/{filename}', bbox_inches='tight')
                print(f'Plot saved to file {args.output_folder}/{filename}!')
            else:
                plt.show()
            plt.close()

main()