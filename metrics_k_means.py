import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from helpers import configure_plots
from k_means import DECADE_INDICES, kmeans_fit, plot_clusters

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', default=None)
    parser.add_argument('--metrics_file_base', default='metrics_corr_alm_60_lag_0_ed_0p005')
    args = parser.parse_args()
    label_size, font_size, show_or_save = configure_plots(args)

    coreness_file = f'{METRICS_DIR}/{args.metrics_file_base}_cor.pkl'
    degree_file = f'{METRICS_DIR}/{args.metrics_file_base}_deg.pkl'
    eccentricity_file = f'{METRICS_DIR}/{args.metrics_file_base}_ecc.pkl'
    shortest_path_file = f'{METRICS_DIR}/{args.metrics_file_base}_sp.pkl'
    betweenness_centrality_file = f'{METRICS_DIR}/{args.metrics_file_base}_b_cent.pkl'
    closeness_centrality_file = f'{METRICS_DIR}/{args.metrics_file_base}_c_cent.pkl'
    eigenvector_centrality_file = f'{METRICS_DIR}/{args.metrics_file_base}_e_cent.pkl'
    coreness_df: pd.DataFrame = pd.read_pickle(coreness_file)
    degree_df: pd.DataFrame = pd.read_pickle(degree_file)
    eccentricity_df: pd.DataFrame = pd.read_pickle(eccentricity_file)
    shortest_path_df: pd.DataFrame = pd.read_pickle(shortest_path_file)
    betweenness_centrality_df: pd.DataFrame = pd.read_pickle(betweenness_centrality_file)
    closeness_centrality_df: pd.DataFrame = pd.read_pickle(closeness_centrality_file)
    eigenvector_centrality_df: pd.DataFrame = pd.read_pickle(eigenvector_centrality_file)
    dfs = [coreness_df, degree_df, eccentricity_df, shortest_path_df,
        betweenness_centrality_df, closeness_centrality_df]

    # First, calculate and print silhouette scores for k = 2, ..., 12
    k_list = list(range(2, 12 + 1))
    sils_df = pd.DataFrame(0, index=k_list, columns=[
        *[f'd1_{metric_name}' for metric_name in metric_names],
        *[f'd2_{metric_name}' for metric_name in metric_names],
    ])
    for i, (df, metric_name) in enumerate(zip(dfs, metric_names)):
        # Decade 1 will have NaNs due to lookback window when constructing networks
        # Either decade will have NaNs if using networks build from separate months
        d1 = np.array(df.iloc[DECADE_INDICES[0], :].dropna()).T
        d2 = np.array(df.iloc[DECADE_INDICES[1], :].dropna()).T
        for j, k in enumerate(k_list):
            _, sils_df.iloc[j, i] = kmeans_fit(k, d1)
            _, sils_df.iloc[j, len(metric_names) + i] = kmeans_fit(k, d2)
    sils_df.index.name = 'num_clusters'
    print(sils_df)

    # Next, plot results for k = 2, ..., 7
    k_list = list(range(2, 7 + 1))
    dot_size = 100 if args.output_folder else 20
    for df, metric_name in zip(dfs, metric_names):
        # Decade 1 will have NaNs due to lookback window when constructing networks
        d1 = np.array(df.iloc[DECADE_INDICES[0], :].dropna()).T
        d2 = np.array(df.iloc[DECADE_INDICES[1], :].dropna()).T
        figure, axes = plt.subplots(2, 3, layout='compressed')
        axes = iter(axes.flatten())
        # TODO: recalculate lons/lats
        for k in k_list:
            labels_d1, sil_d1 = kmeans_fit(k, d1)
            axis = next(axes)
            plot_clusters(d1, labels_d1, axis, lons, lats, dot_size=dot_size)
            axis.set_title(f'Decade 1 {metric_name}, {k} clusters, sil score {round(sil_d1, 4)}')
        show_or_save(figure, f'kmeans_{metric_name}_decade_1.png')
        figure, axes = plt.subplots(2, 3, layout='compressed')
        axes = iter(axes.flatten())
        for k in k_list:
            labels_d2, sil_d2 = kmeans_fit(k, d2)
            axis = next(axes)
            plot_clusters(d2, labels_d2, axis, lons, lats, dot_size=dot_size)
            axis.set_title(f'Decade 2 {metric_name}, {k} clusters, sil score {round(sil_d2, 4)}')
        show_or_save(figure, f'kmeans_{metric_name}_decade_2.png')

if __name__ == '__main__':
    main()