import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from helpers import configure_plots, get_map, scatter_map

METRICS_DIR = 'data/metrics'

metric_names = [
    'coreness',
    'degree',
    'eccentricity',
    'shortest_path',
    'betweenness_centrality',
    'closeness_centrality',
    # 'eigenvector_centrality',
]

def size_func(series):
    series_norm = series / np.max(series)
    return [50 * n for n in series_norm]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics_file_base', default='metrics_corr_alm_60_lag_0_ed_0p005')
    parser.add_argument('--output_folder', default=None)
    args = parser.parse_args()
    label_size, font_size, show_or_save = configure_plots(args)

    coreness_df = pd.read_pickle(f'{METRICS_DIR}/{args.metrics_file_base}_cor.pkl')
    degree_df = pd.read_pickle(f'{METRICS_DIR}/{args.metrics_file_base}_deg.pkl')
    eccentricity_df = pd.read_pickle(f'{METRICS_DIR}/{args.metrics_file_base}_ecc.pkl')
    shortest_path_df = pd.read_pickle(f'{METRICS_DIR}/{args.metrics_file_base}_sp.pkl')
    betweenness_centrality_df = pd.read_pickle(f'{METRICS_DIR}/{args.metrics_file_base}_b_cent.pkl')
    closeness_centrality_df = pd.read_pickle(f'{METRICS_DIR}/{args.metrics_file_base}_c_cent.pkl')
    eigenvector_centrality_df = pd.read_pickle(f'{METRICS_DIR}/{args.metrics_file_base}_e_cent.pkl')
    lats, lons = zip(*coreness_df.columns)
    dfs = [coreness_df, degree_df, eccentricity_df, shortest_path_df,
        betweenness_centrality_df, closeness_centrality_df]
    df_mins = [df.min().min() for df in dfs]
    df_maxes = [df.max().max() for df in dfs]
    for dt in coreness_df.index.values:
        if coreness_df.loc[dt].isnull().all():
            continue
        figure, axes = plt.subplots(2, 3, layout='compressed')
        axes = iter(axes.flatten())
        for df, df_min, df_max, metric_name in zip(dfs, df_mins, df_maxes, metric_names):
            axis = next(axes)
            _map = get_map(axis)
            mx, my = _map(lons, lats)
            series = df.loc[dt]
            scatter_map(axis, mx, my, series, cb_min=df_min, cb_max=df_max, cb_fs=label_size,
                size_func=lambda series: 100 if args.output_folder else 20)
            axis.set_title(f'{pd.to_datetime(dt).strftime("%b %Y")}: {metric_name}')
        show_or_save(figure, f'metrics_map_{pd.to_datetime(dt).strftime("%Y_%m")}.png')

if __name__ == '__main__':
    main()