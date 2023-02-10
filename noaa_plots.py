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
    dataset = args.data_file.split('_')[0]

    dt_range = pd.to_datetime(df.index.values)
    figure, axis = plt.subplots(1, layout='compressed')
    # Get a map of the whole world with the locations kept for buildng networks
    kept_columns = []
    for lat, lon in df.columns:
        if lat >= -60 and lat <= 60:
            kept_columns.append((lat, lon))
    networks_df = df[kept_columns]
    lats, lons = zip(*networks_df.columns)
    _map = get_map(axis, aus=False)
    mx, my = _map(lons, lats)
    dt = dt_range[-1]
    scatter_map(axis, mx, my, networks_df.loc[dt, :], cmap='RdYlBu_r',
        size_func=lambda x: 200 if args.output_folder else 50)
    axis.set_title(dt.strftime("%b %Y"))
    show_or_save(figure, f'{dataset}_world_{dt.strftime("%Y_%m")}.png')
    
    # Plot each value over the last year over Australia
    figure, axes = plt.subplots(3, 4, layout='compressed')
    axes = iter(axes.flatten())
    aus_columns = []
    for lat, lon in df.columns:
        if lat >= -45 and lat <= -10 and lon >= 110 and lon <= 155:
            aus_columns.append((lat, lon))
    aus_df = df[aus_columns]
    lats, lons = zip(*aus_df.columns)
    df_min = aus_df.min().min()
    df_max = aus_df.max().max()
    for i, dt in enumerate(dt_range):
        if i < len(dt_range) - 12:
            continue
        axis = next(axes)
        _map = get_map(axis)
        mx, my = _map(lons, lats)
        scatter_map(axis, mx, my, aus_df.loc[dt, :], cb_min=df_min,
            cb_max=df_max, cb_fs=label_size, cmap='RdYlBu_r',
            size_func=lambda x: 500 if args.output_folder else 50)
        axis.set_title(dt.strftime("%b %Y"))
    show_or_save(figure, f'{dataset}_aus_{dt_range[0].strftime("%Y_%m")}'
        f'_{dt_range[-1].strftime("%Y_%m")}.png')

if __name__ == '__main__':
    main()
