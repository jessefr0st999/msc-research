import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from helpers import get_map, prepare_indexed_df

YEARS = range(2000, 2022)
DATA_DIR = 'data/precipitation'
DATA_FILE = f'{DATA_DIR}/FusedData.csv'
LOCATIONS_FILE = f'{DATA_DIR}/Fused.Locations.csv'
size_func = lambda series: [500 * n for n in series]
cmap = 'RdYlGn_r'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', default=None)
    parser.add_argument('--month', type=int, default=None)
    args = parser.parse_args()

    raw_df = pd.read_csv(DATA_FILE)
    locations_df = pd.read_csv(LOCATIONS_FILE)
    df = prepare_indexed_df(raw_df, locations_df, new_index=['lat', 'lon'])

    pos = None
    if args.month:
        months = [args.month]
    else:
        months = range(1, 13)
    for y in YEARS:
        for m in months:
            dt = datetime(y, m, 1)
            dt_prec_df = df[df['date'] == dt]
            if dt_prec_df.empty:
                continue

            figure, axis = plt.subplots(1)
            _map = get_map(axis)
            if pos is None:
                lats, lons = zip(*dt_prec_df.index.values)
                mx, my = _map(lons, lats)
                pos = {elem: (mx[i], my[i]) for i, elem in enumerate(dt_prec_df.index)}

            series = dt_prec_df['prec']
            axis.scatter(
                mx, my, c=series, cmap=cmap,
                s=size_func((series - np.min(series)) / np.max(series)),
            )
            norm = mpl.colors.Normalize(vmin=np.min(series), vmax=np.max(series))
            plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis)
            plt.tight_layout()
            axis.set_title(f'{dt.strftime("%b")} {y}: raw precipitation', fontsize=20)
            if args.output_folder:
                month_str = str(m) if m >= 10 else f'0{m}'
                filename = f'raw_plot_{y}_{month_str}.png'
                figure.set_size_inches(32, 18)
                plt.savefig(f'{args.output_folder}/{filename}', bbox_inches='tight')
                print(f'Plot saved to file {args.output_folder}/{filename}!')
            else:
                plt.show()
            plt.close()

main()