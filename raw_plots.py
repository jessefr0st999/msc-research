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

    # Plot averages and standard deviations for each month
    mx, my = None, None
    mean_min = np.Inf
    mean_max = -np.Inf
    sd_min = np.Inf
    sd_max = -np.Inf
    for month in range(1, 13):
        df_m = df.loc[[i.month == month for i in df.index], :]
        mean_min = np.min([mean_min, df_m.mean(axis=0).min()])
        mean_max = np.max([mean_max, df_m.mean(axis=0).max()])
        sd_min = np.min([sd_min, df_m.std(axis=0).min()])
        sd_max = np.max([sd_max, df_m.std(axis=0).max()])
    figure, axes = plt.subplots(3, 4, layout='compressed')
    axes = iter(axes.flatten())
    for month in range(1, 13):
        df_m = df.loc[[i.month == month for i in df.index], :]
        axis = next(axes)
        _map = get_map(axis, region=map_region)
        if mx is None:
            mx, my = _map(lons, lats)
        scatter_map(axis, mx, my, df_m.mean(axis=0), cb_min=mean_min, cb_max=mean_max,
            size_func=lambda series: 75, cb_fs=label_size, cmap='inferno_r')
        axis.set_title(datetime(2000, month, 1).strftime('%B') + ' mean', fontsize=font_size)
    figure_title = f'{dataset}_monthly_means.png'
    show_or_save(figure, figure_title)

    figure, axes = plt.subplots(3, 4, layout='compressed')
    axes = iter(axes.flatten())
    for month in range(1, 13):
        df_m = df.loc[[i.month == month for i in df.index], :]
        axis = next(axes)
        _map = get_map(axis, region=map_region)
        if mx is None:
            mx, my = _map(lons, lats)
        scatter_map(axis, mx, my, df_m.std(axis=0), cb_min=sd_min, cb_max=sd_max,
            size_func=lambda series: 75, cb_fs=label_size, cmap='inferno_r')
        axis.set_title(datetime(2000, month, 1).strftime('%B') + ' SD', fontsize=font_size)
    figure_title = f'{dataset}_monthly_st_devs.png'
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