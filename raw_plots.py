import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from helpers import get_map, scatter_map, prepare_df, configure_plots

YEARS = range(2000, 2022)

def scatterplot(axis, mx, my, series, cmap='jet', cb_fs=10):
    axis.scatter(mx, my, c=series, cmap=cmap, s=100)
    norm = mpl.colors.Normalize(vmin=np.min(series), vmax=np.max(series))
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis)\
        .ax.tick_params(labelsize=cb_fs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/precipitation')
    parser.add_argument('--data_file', default='FusedData.csv')
    parser.add_argument('--output_folder', default=None)
    parser.add_argument('--start_year', type=int, default=2021)
    parser.add_argument('--start_month', type=int, default=4)
    args = parser.parse_args()
    label_size, font_size, show_or_save = configure_plots(args)

    dataset = 'prec' if args.data_file == 'FusedData.csv' else args.data_file.split('_')[0]
    df, lats, lons = prepare_df(args.data_dir, args.data_file, dataset)
    map_region = 'aus' if dataset == 'prec' else 'world'

    # Plot for each month in the last year of available data
    mx, my = None, None
    figure, axes = plt.subplots(3, 4, layout='compressed')
    axes = iter(axes.flatten())
    start_dt = datetime(args.start_year, args.start_month, 1)
    end_dt = start_dt + relativedelta(months=11)
    _min = np.Inf
    _max = -np.Inf
    for i, dt in enumerate(df.index.values):
        dt = pd.to_datetime(dt)
        if dt < start_dt or dt > end_dt:
            continue
        dt_values = df.iloc[i].values
        _min = np.min([_min, dt_values.min()])
        _max = np.max([_max, dt_values.max()])
    for i, dt in enumerate(df.index.values):
        dt = pd.to_datetime(dt)
        if dt < start_dt or dt > end_dt:
            continue
        axis = next(axes)
        _map = get_map(axis, region=map_region)
        if mx is None:
            mx, my = _map(lons, lats)
        dt_values = df.iloc[i].values
        scatter_map(axis, mx, my, dt_values, cb_min=_min, cb_max=_max,
            size_func=lambda series: 75, cb_fs=label_size, cmap='jet')
        axis.set_title(dt.strftime('%b %Y'), fontsize=font_size)
    figure_title = f'{dataset}_{start_dt.strftime("%Y_%m")}_{end_dt.strftime("%Y_%m")}.png'
    show_or_save(figure, figure_title)

    # Plot quantiles
    figure, axis = plt.subplots(1)
    quantile_points = np.linspace(0, 1, 101)
    quantiles = np.quantile(df.unstack().values, quantile_points)
    axis.plot(quantile_points, quantiles, 'o-')
    axis.set_title('quantiles')
    show_or_save(figure, f'{dataset}_quantiles.png')

    df = df.unstack().reset_index()
    df.columns = ['Lat', 'Lon' ,'date', 'values']
    date_group = df.groupby('date')['values']
    date_mean = date_group.mean()
    date_std = date_group.std()
    date_median = date_group.median()
    date_max = date_group.max()
    dates = date_mean.index.values

    figure, axes = plt.subplots(2, 2, layout='compressed')
    axes = iter(axes.flatten())
    axis = next(axes)
    axis.plot(dates, date_mean, '-')
    axis.set_title('mean')

    axis = next(axes)
    axis.plot(dates, date_std, '-')
    axis.set_title('standard deviation')

    axis = next(axes)
    axis.plot(dates, date_median, '-')
    axis.set_title('median')

    axis = next(axes)
    axis.plot(dates, date_max, '-')
    axis.set_title('maximum')
    show_or_save(figure, f'{dataset}_stats_by_date.png')

    loc_group = df.groupby(['Lat', 'Lon'])['values']
    loc_mean = loc_group.mean()
    loc_std = loc_group.std()
    loc_median = loc_group.median()
    loc_max = loc_group.max()
    lats, lons = zip(*loc_mean.index.values)

    figure, axes = plt.subplots(2, 2, layout='compressed')
    axes = iter(axes.flatten())
    axis = next(axes)
    mean_map = get_map(axis, region=map_region)
    mx, my = mean_map(lons, lats)
    scatterplot(axis, mx, my, loc_mean, cb_fs=label_size)
    axis.set_title('mean')

    axis = next(axes)
    _ = get_map(axis, region=map_region)
    scatterplot(axis, mx, my, loc_std, cb_fs=label_size)
    axis.set_title('standard deviation')

    axis = next(axes)
    _ = get_map(axis, region=map_region)
    scatterplot(axis, mx, my, loc_median, cb_fs=label_size)
    axis.set_title('median')

    axis = next(axes)
    _ = get_map(axis, region=map_region)
    scatterplot(axis, mx, my, loc_max, cb_fs=label_size)
    axis.set_title('maximum')
    show_or_save(figure, f'{dataset}_stats_by_location.png')

main()