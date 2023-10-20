import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from helpers import get_map, scatter_map, prepare_df, configure_plots

YEARS = range(2000, 2022)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/precipitation')
    parser.add_argument('--data_file', default='FusedData.csv')
    parser.add_argument('--norm_by_mean', action='store_true', default=False)
    parser.add_argument('--output_folder', default=None)
    args = parser.parse_args()
    label_size, font_size, show_or_save = configure_plots(args)

    dataset = 'prec' if args.data_file == 'FusedData.csv' else args.data_file.split('_')[0]
    df, lats, lons = prepare_df(args.data_dir, args.data_file, dataset)
    map_region = 'aus' if dataset == 'prec' else 'world'

    # Plot the series for each decade at the location best representing Lismore
    # figure, axes = plt.subplots(2, 1, layout='compressed')
    # axes = iter(axes.flatten())
    # axis = next(axes)
    # axis.plot(df.index[:132], df.loc[: pd.to_datetime('2011-03-01'), (-28.75, 153.5)], '-')
    # axis.set_ylim([0, 650])
    # axis.set_title('(-28.75, 153.5) decade 1')
    # axis = next(axes)
    # axis.plot(df.index[132:], df.loc[pd.to_datetime('2011-04-01') :, (-28.75, 153.5)], '-')
    # axis.set_ylim([0, 650])
    # axis.set_title('(-28.75, 153.5) decade 2')
    # slice = -36
    # slice_date = df.index.values[slice]
    # t_vec = df.index.values[slice:]
    # x1_vec = df.loc[slice_date:, (-28.75, 153.5)]
    # x2_vec = df.loc[slice_date:, (-28.75, 152.5)]
    # axis = next(axes)
    # axis.plot(t_vec, x1_vec, 'b-', label='(-28.75, 153.5)')
    # axis.plot(t_vec, x2_vec, 'g-', label='(-28.75, 152.5)')
    # axis.legend()
    # axis = next(axes)
    # a, b = np.polyfit(x1_vec, x2_vec, 1)
    # axis.plot(x1_vec, x2_vec, 'ro')
    # axis.plot(x1_vec, a * x1_vec + b, 'c-')
    # plt.show()

    # Plot averages and standard deviations for each month
    mx, my = None, None
    mean_min = np.Inf
    mean_max = -np.Inf
    median_min = np.Inf
    median_max = -np.Inf
    sd_min = np.Inf
    sd_max = -np.Inf
    for month in range(1, 13):
        df_m = df.loc[[i.month == month for i in df.index], :]
        mean_min = np.min([mean_min, df_m.mean(axis=0).min()])
        mean_max = np.max([mean_max, df_m.mean(axis=0).max()])
        median_min = np.min([median_min, df_m.median(axis=0).min()])
        median_max = np.max([median_max, df_m.median(axis=0).max()])
        sd_min = np.min([sd_min, df_m.std(axis=0).min()])
        sd_max = np.max([sd_max, df_m.std(axis=0).max()])

    df.index = pd.DatetimeIndex(df.index)
    df_d1 = df.loc[:'2011-03-31', :]
    df_d2 = df.loc['2011-04-01':, :]

    # Plot absolute difference between mean and median (mean usually bigger)
    figure, axis = plt.subplots(1)
    _map = get_map(axis)
    mx, my = _map(lons, lats)
    series = np.abs(df_m.mean(axis=0) - df_m.median(axis=0))
    scatter_map(axis, mx, my, series, size_func=lambda x: 15,
        # cb_min=-np.abs(series).max(), cb_max=np.abs(series).max())
        cb_min=0, cb_max=150)
    axis.set_title('mean - median')

    figure, axes = plt.subplots(3, 4, layout='compressed')
    axes = iter(axes.flatten())
    for month in range(1, 13):
        axis = next(axes)
        _map = get_map(axis, region=map_region)
        if mx is None:
            mx, my = _map(lons, lats)
        df_m = df.loc[[i.month == month for i in df.index], :]
        series = np.abs(df_m.mean(axis=0) - df_m.median(axis=0))
        scatter_map(axis, mx, my, series, size_func=lambda x: 15,
            # cb_min=-np.abs(series).max(), cb_max=np.abs(series).max())
            cb_min=0, cb_max=100)
        axis.set_title(datetime(2000, month, 1).strftime('%B') + ' |mean - median|')
    plt.show()

    figure, axes = plt.subplots(3, 4, layout='compressed')
    axes = iter(axes.flatten())
    for month in range(1, 13):
        m_av = df.loc[[i.month == month for i in df.index]].mean(axis=0)
        df_m = df_d2.loc[[i.month == month for i in df_d2.index], :].mean(axis=0) - \
            df_d1.loc[[i.month == month for i in df_d1.index], :].mean(axis=0)
        if args.norm_by_mean:
            df_m /= m_av
            _min = -2
            _max = 2
        else:
            _min = -np.abs(df_m).max()
            _max = np.abs(df_m).max()
        axis = next(axes)
        _map = get_map(axis, region=map_region)
        if mx is None:
            mx, my = _map(lons, lats)
        scatter_map(axis, mx, my, df_m, cb_min=_min,
            cb_max=_max, size_func=lambda x: 15, cmap='RdYlBu_r')
        axis.set_title(datetime(2000, month, 1).strftime('%B'))
    figure_title = f'{dataset}_monthly_means.png'
    show_or_save(figure, figure_title)

    figure, axes = plt.subplots(3, 4, layout='compressed')
    axes = iter(axes.flatten())
    for month in range(1, 13):
        m_av = df.loc[[i.month == month for i in df.index]].mean(axis=0)
        df_m = df_d2.loc[[i.month == month for i in df_d2.index], :].median(axis=0) - \
            df_d1.loc[[i.month == month for i in df_d1.index], :].median(axis=0)
        if args.norm_by_mean:
            df_m /= m_av
            _min = -2
            _max = 2
        else:
            _min = -np.abs(df_m).max()
            _max = np.abs(df_m).max()
        axis = next(axes)
        _map = get_map(axis, region=map_region)
        if mx is None:
            mx, my = _map(lons, lats)
        scatter_map(axis, mx, my, df_m, cb_min=_min,
            cb_max=_max, size_func=lambda x: 15, cmap='RdYlBu_r')
        axis.set_title(datetime(2000, month, 1).strftime('%B'))
    figure_title = f'{dataset}_monthly_medians.png'
    show_or_save(figure, figure_title)
    
    for _df, label in [
        # (df, ''),
        # (df_d1, ' (decade 1)'),
        # (df_d2, ' (decade 2)'),
    ]:
        figure, axes = plt.subplots(3, 4, layout='compressed')
        axes = iter(axes.flatten())
        for month in range(1, 13):
            m_av = df.loc[[i.month == month for i in df.index]].mean(axis=0)
            df_m = _df.loc[[i.month == month for i in _df.index], :]
            if args.norm_by_mean:
                df_m /= m_av
            axis = next(axes)
            _map = get_map(axis, region=map_region)
            if mx is None:
                mx, my = _map(lons, lats)
            scatter_map(axis, mx, my, df_m.mean(axis=0),
                cb_min=mean_min, cb_max=mean_max,
                size_func=lambda x: 15)
            axis.set_title(datetime(2000, month, 1).strftime('%B'))
        figure_title = f'{dataset}_monthly_means.png'
        show_or_save(figure, figure_title)

        figure, axes = plt.subplots(3, 4, layout='compressed')
        axes = iter(axes.flatten())
        for month in range(1, 13):
            m_av = df.loc[[i.month == month for i in df.index]].mean(axis=0)
            df_m = _df.loc[[i.month == month for i in _df.index], :]
            if args.norm_by_mean:
                df_m /= m_av
            axis = next(axes)
            _map = get_map(axis, region=map_region)
            if mx is None:
                mx, my = _map(lons, lats)
            scatter_map(axis, mx, my, df_m.median(axis=0),
                cb_min=median_min, cb_max=median_max,
                size_func=lambda x: 15)
            axis.set_title(datetime(2000, month, 1).strftime('%B'))
        figure_title = f'{dataset}_monthly_medians.png'
        show_or_save(figure, figure_title)

        figure, axes = plt.subplots(3, 4, layout='compressed')
        axes = iter(axes.flatten())
        for month in range(1, 13):
            m_av = df.loc[[i.month == month for i in df.index]].mean(axis=0)
            df_m = _df.loc[[i.month == month for i in _df.index], :]
            if args.norm_by_mean:
                df_m /= m_av
            axis = next(axes)
            _map = get_map(axis, region=map_region)
            if mx is None:
                mx, my = _map(lons, lats)
            scatter_map(axis, mx, my, df_m.std(axis=0),
                cb_min=sd_min, cb_max=sd_max,
                size_func=lambda x: 15)
            axis.set_title(datetime(2000, month, 1).strftime('%B'))
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

    slice = -36
    figure, axes = plt.subplots(2, 1, layout='compressed')
    axes = iter(axes.flatten())
    axis = next(axes)
    axis.plot(dates, date_mean, '-')
    axis.set_title('mean')
    axis = next(axes)
    axis.plot(dates[slice:], date_mean[slice:], '-')
    
    figure, axes = plt.subplots(2, 1, layout='compressed')
    axes = iter(axes.flatten())
    axis = next(axes)
    axis.plot(dates, date_median, '-')
    axis.set_title('median')
    axis = next(axes)
    axis.plot(dates[slice:], date_median[slice:], '-')

    figure, axes = plt.subplots(2, 1, layout='compressed')
    axes = iter(axes.flatten())
    axis = next(axes)
    axis.plot(dates, date_std, '-')
    axis.set_title('standard deviation')
    axis = next(axes)
    axis.plot(dates[slice:], date_std[slice:], '-')

    figure, axes = plt.subplots(2, 1, layout='compressed')
    axes = iter(axes.flatten())
    axis = next(axes)
    axis.plot(dates, date_max, '-')
    axis.set_title('maximum')
    axis = next(axes)
    axis.plot(dates[slice:], date_max[slice:], '-')
    plt.show()

    figure, axes = plt.subplots(1, 3, layout='compressed')
    axes = iter(axes.flatten())
    axis = next(axes)
    _map = get_map(axis, region=map_region)
    mx, my = _map(lons, lats)
    scatter_map(axis, mx, my, df.mean(axis=0).values,
        cb_fs=label_size, size_func=lambda x: 30)
    axis.set_title('mean')

    axis = next(axes)
    _map = get_map(axis, region=map_region)
    scatter_map(axis, mx, my, df.median(axis=0).values,
        cb_fs=label_size, size_func=lambda x: 30)
    axis.set_title('median')

    axis = next(axes)
    _map = get_map(axis, region=map_region)
    scatter_map(axis, mx, my, df.std(axis=0).values,
        cb_fs=label_size, size_func=lambda x: 30)
    axis.set_title('standard deviation')
    show_or_save(figure, f'{dataset}_stats_by_location.png')

if __name__ == '__main__':
    main()