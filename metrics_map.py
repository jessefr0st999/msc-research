import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from helpers import configure_plots, get_map, scatter_map, file_region_type

METRICS_DIR = 'data/metrics'

metric_names = [
    # 'coreness',
    'degree',
    'weighted_degree',
    # 'eccentricity',
    # 'shortest_path',
    # 'local_link_distance',
    'betweenness_centrality',
    'closeness_centrality',
    # 'eigenvector_centrality',
]

def size_func(series):
    series_norm = series / np.max(series)
    return [50 * n for n in series_norm]

# TODO: just specify a single datetime for this script, as per the communities
# TODO: read metrics for each graph from a separate file once metrics.py is updated
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics_file_base', default='metrics_corr_alm_60_lag_0_ed_0p005')
    parser.add_argument('--output_folder', default=None)
    parser.add_argument('--yearly', action='store_true', default=False)
    parser.add_argument('--last_dt', action='store_true', default=False)
    args = parser.parse_args()
    label_size, font_size, show_or_save = configure_plots(args)
    map_region = file_region_type(args.metrics_file_base)

    # coreness_df = pd.read_pickle(f'{METRICS_DIR}/{args.metrics_file_base}_cor.pkl')
    degree_df = pd.read_pickle(f'{METRICS_DIR}/{args.metrics_file_base}_deg.pkl')
    weighted_degree_df = pd.read_pickle(f'{METRICS_DIR}/{args.metrics_file_base}_wdeg.pkl')
    # eccentricity_df = pd.read_pickle(f'{METRICS_DIR}/{args.metrics_file_base}_ecc.pkl')
    # shortest_path_df = pd.read_pickle(f'{METRICS_DIR}/{args.metrics_file_base}_sp.pkl')
    # local_link_distance_df = pd.read_pickle(f'{METRICS_DIR}/{args.metrics_file_base}_lld.pkl')
    betweenness_centrality_df = pd.read_pickle(f'{METRICS_DIR}/{args.metrics_file_base}_b_cent.pkl')
    closeness_centrality_df = pd.read_pickle(f'{METRICS_DIR}/{args.metrics_file_base}_c_cent.pkl')
    # eigenvector_centrality_df = pd.read_pickle(f'{METRICS_DIR}/{args.metrics_file_base}_e_cent.pkl')
    lats, lons = zip(*degree_df.columns)
    dfs = [degree_df, weighted_degree_df, betweenness_centrality_df, closeness_centrality_df]
    df_mins = [df.min().min() for df in dfs]
    df_maxes = [df.max().max() for df in dfs]
    if 'decadal' in args.metrics_file_base:
        d1_dt, d2_dt = degree_df.index.values
        figure, axes = plt.subplots(2, len(metric_names), layout='compressed')
        for i, (df, df_min, df_max, metric_name) in enumerate(zip(dfs, df_mins, df_maxes, metric_names)):
            axis = axes[0, i]
            _map = get_map(axis, region=map_region)
            mx, my = _map(lons, lats)
            scatter_map(axis, mx, my, df.loc[d1_dt], cb_min=df_min, cb_max=df_max, cb_fs=label_size,
                size_func=lambda series: 100 if args.output_folder else 20)
            axis.set_title(f'decade 1: {metric_name}')

            axis = axes[1, i]
            _map = get_map(axis, region=map_region)
            mx, my = _map(lons, lats)
            scatter_map(axis, mx, my, df.loc[d2_dt], cb_min=df_min, cb_max=df_max, cb_fs=label_size,
                size_func=lambda series: 100 if args.output_folder else 20)
            axis.set_title(f'decade 2: {metric_name}')
        show_or_save(figure, f'{args.metrics_file_base}_decadal.png')

        figure, axes = plt.subplots(1, len(metric_names), layout='compressed')
        axes = iter(axes.flatten())
        for df, df_min, df_max, metric_name in zip(dfs, df_mins, df_maxes, metric_names):
            axis = next(axes)
            _map = get_map(axis, region=map_region)
            mx, my = _map(lons, lats)
            series_diff = df.loc[d2_dt] - df.loc[d1_dt]
            scatter_map(axis, mx, my, series_diff, cb_fs=label_size, cmap='RdYlBu_r',
                size_func=lambda series: 100 if args.output_folder else 20)
            axis.set_title(metric_name)
        show_or_save(figure, f'{args.metrics_file_base}_decadal_diff.png')
    else:
        for i, dt in enumerate(degree_df.index.values):
            if degree_df.loc[dt].isnull().all():
                continue
            if args.last_dt and i < len(degree_df.index) - 1:
                continue
            if args.yearly:
                _dt = pd.to_datetime(dt)
                if _dt.month != 3:
                    continue
                if _dt.year not in [2002, 2007, 2012, 2017, 2022]:
                    continue
            figure, axes = plt.subplots(2, 4, layout='compressed')
            axes = iter(axes.flatten())
            for df, df_min, df_max, metric_name in zip(dfs, df_mins, df_maxes, metric_names):
                axis = next(axes)
                _map = get_map(axis, region=map_region)
                mx, my = _map(lons, lats)
                series = df.loc[dt]
                scatter_map(axis, mx, my, series, cb_min=df_min, cb_max=df_max, cb_fs=label_size,
                    size_func=lambda series: 100 if args.output_folder else 20)
                axis.set_title(f'{pd.to_datetime(dt).strftime("%b %Y")}: {metric_name}')
            show_or_save(figure, f'{args.metrics_file_base}_{pd.to_datetime(dt).strftime("%Y_%m")}.png')

if __name__ == '__main__':
    main()