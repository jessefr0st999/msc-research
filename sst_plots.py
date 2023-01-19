import argparse

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from helpers import configure_plots, get_map as get_aus_map, scatter_map

DATA_DIR = 'data/sst'

# Alternative to get_map in helpers to use the default projection and bounds
def get_world_map(axis=None):
    _map = Basemap(
        lat_ts=0,
        resolution='l',
        suppress_ticks=True,
        ax=axis,
    )
    _map.drawcountries(linewidth=1)
    _map.drawstates(linewidth=0.2)
    _map.drawcoastlines(linewidth=1)
    return _map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', default=None)
    parser.add_argument('--sst_file', default='sst_df_2021_01_2022_03.pkl')
    args = parser.parse_args()
    label_size, font_size, show_or_save = configure_plots(args)
    
    # Plot the most recent values over the whole world
    sst_df: pd.DataFrame = pd.read_pickle(f'{DATA_DIR}/{args.sst_file}')
    # Transform to Celsius
    sst_df = sst_df - 273.15
    sst_dt_range = pd.to_datetime(sst_df.index.values)
    lats, lons = zip(*sst_df.columns)
    figure, axis = plt.subplots(1, layout='compressed')
    _map = get_world_map(axis)
    mx, my = _map(lons, lats)
    dt = sst_dt_range[-1]
    scatter_map(axis, mx, my, sst_df.loc[dt, :], cmap='RdYlBu_r')
    axis.set_title(dt.strftime("%b %Y"))
    show_or_save(figure, f'sst_world_{dt.strftime("%Y_%m")}.png')
    
    # Plot each value over the last year over Australia
    figure, axes = plt.subplots(3, 4, layout='compressed')
    axes = iter(axes.flatten())
    sst_df_min = sst_df.min().min()
    sst_df_max = sst_df.max().max()
    aus_mx_my = False
    for i, dt in enumerate(sst_dt_range):
        if i < len(sst_dt_range) - 12:
            continue
        axis = next(axes)
        _map = get_aus_map(axis)
        if not aus_mx_my:
            mx, my = _map(lons, lats)
            aus_mx_my = True
        scatter_map(axis, mx, my, sst_df.loc[dt, :], cb_min=sst_df_min,
            cb_max=sst_df_max, cb_fs=label_size, cmap='RdYlBu_r')
        axis.set_title(dt.strftime("%b %Y"))
    show_or_save(figure, f'sst_aus_{sst_dt_range[0].strftime("%Y_%m")}'
        f'_{sst_dt_range[-1].strftime("%Y_%m")}.png')

if __name__ == '__main__':
    main()
