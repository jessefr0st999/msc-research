import argparse

import pandas as pd
import matplotlib.pyplot as plt

from helpers import configure_plots, get_map, scatter_map

DATA_DIR = 'data/noaa'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', default=None)
    parser.add_argument('--data_file', default='slp_1949_02_2023_01.pkl')
    parser.add_argument('--dataset', default='slp')
    args = parser.parse_args()
    label_size, font_size, show_or_save = configure_plots(args)
    
    # Plot the most recent values over the whole world
    df: pd.DataFrame = pd.read_pickle(f'{DATA_DIR}/{args.data_file}')

    dt_range = pd.to_datetime(df.index.values)
    lats, lons = zip(*df.columns)
    figure, axis = plt.subplots(1, layout='compressed')
    # Get a map of the whole world
    _map = get_map(axis, aus=False)
    mx, my = _map(lons, lats)
    dt = dt_range[-1]
    scatter_map(axis, mx, my, df.loc[dt, :], cmap='RdYlBu_r')
    axis.set_title(dt.strftime("%b %Y"))
    show_or_save(figure, f'{args.dataset}_world_{dt.strftime("%Y_%m")}.png')
    
    # Plot each value over the last year over Australia
    figure, axes = plt.subplots(3, 4, layout='compressed')
    axes = iter(axes.flatten())
    aus_col_indices = []
    for i, c in enumerate(df.columns):
        if c[0] >= -45 and c[0] <= -10 and c[1] >= 110 and c[1] <= 155:
            aus_col_indices.append(i)
    aus_df = df.iloc[:, aus_col_indices]
    df_min = aus_df.min().min()
    df_max = aus_df.max().max()
    aus_mx_my = False
    for i, dt in enumerate(dt_range):
        if i < len(dt_range) - 12:
            continue
        axis = next(axes)
        _map = get_map(axis)
        if not aus_mx_my:
            mx, my = _map(lons, lats)
            aus_mx_my = True
        scatter_map(axis, mx, my, df.loc[dt, :], cb_min=df_min,
            cb_max=df_max, cb_fs=label_size, cmap='RdYlBu_r')
        axis.set_title(dt.strftime("%b %Y"))
    show_or_save(figure, f'{args.dataset}_aus_{dt_range[0].strftime("%Y_%m")}'
        f'_{dt_range[-1].strftime("%Y_%m")}.png')

if __name__ == '__main__':
    main()
