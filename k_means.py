from datetime import datetime
import pickle
import argparse
from pprint import pprint

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from helpers import get_map, prepare_indexed_df

YEARS = list(range(2000, 2022 + 1))
DATA_DIR = 'data/precipitation'
OUTPUTS_DIR = 'data/outputs'
DATA_FILE = f'{DATA_DIR}/FusedData.csv'
LOCATIONS_FILE = f'{DATA_DIR}/Fused.Locations.csv'
COLOURS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 'black', '#444', '#ccc']

def kmeans_optimise(train_data: np.array, max_clusters):
    sil_scores_dict = {}
    kmeans_dict = {}
    for n in range(2, max_clusters + 1):
        kmeans_dict[n] = KMeans(n_clusters=n, random_state=0).fit(train_data)
        sil_scores_dict[n] = silhouette_score(train_data, kmeans_dict[n].labels_)
        # Incentivise more clusters
        # TODO: Investigate this more
        # sil_scores_dict[n] += np.sqrt(1 + n/75)
    optimal_num_clusters = max(sil_scores_dict, key=sil_scores_dict.get)
    return optimal_num_clusters, kmeans_dict[optimal_num_clusters], sil_scores_dict[optimal_num_clusters]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--month', type=int, default=None)
    parser.add_argument('--fixed_clusters', type=int, default=None)
    parser.add_argument('--max_clusters', type=int, default=10)
    parser.add_argument('--plot_clusters', action='store_true', default=False)
    parser.add_argument('--save_summary', action='store_true', default=False)
    args = parser.parse_args()

    raw_df = pd.read_csv(DATA_FILE)
    locations_df = pd.read_csv(LOCATIONS_FILE)
    df = prepare_indexed_df(raw_df, locations_df, month=args.month, new_index='date')

    months = [args.month] if args.month else list(range(1, 13))
    mx, my = None, None
    summary_df = []
    for y in YEARS:
        for m in months:
            dt = datetime(y, m, 1)
            try:
                location_df = df.loc[dt]
            except KeyError:
                continue
            train_data = np.array(location_df['prec']).reshape(-1, 1)
            if args.fixed_clusters:
                num_clusters = args.fixed_clusters
                kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(train_data)
                sil_score = silhouette_score(train_data, kmeans.labels_)
                print(f'{dt.strftime("%b")} {y}: silhouette score {sil_score} for {num_clusters} clusters')
            else:
                num_clusters, kmeans, sil_score = kmeans_optimise(train_data, args.max_clusters)
                print(f'{dt.strftime("%b")} {y}: {num_clusters} optimal clusters with silhouette score {sil_score}')
            summary_df.append({
                'dt': dt,
                'num_clusters': num_clusters,
                'sil_score': sil_score,
            })
            if args.plot_clusters:
                node_colours = [COLOURS[cluster] for cluster in kmeans.labels_]
                figure, axis = plt.subplots(1)
                _map = get_map(axis)
                if mx is None:
                    lons = location_df['lon']
                    lats = location_df['lat']
                    mx, my = _map(lons, lats)
                cmap = mpl.colors.ListedColormap([COLOURS[i] for i in range(num_clusters)])
                title = f'{dt.strftime("%b")} {y}: k-means ({num_clusters} clusters, silhouette score {round(sil_score, 3)})'
                axis.set_title(title, fontsize=20)
                axis.scatter(mx, my, c=node_colours, cmap=cmap)
                plt.show()

    summary_df = pd.DataFrame.from_dict(summary_df).set_index(['dt'])
    if args.month:
        plot_title = dt.strftime("%B")
        filename_title = f'm{args.month}' if args.month >= 10 else f'm0{args.month}'
    else:
        plot_title = 'whole_series'
        filename_title = 'whole_series'
    if args.fixed_clusters:
        figure, axis = plt.subplots(1, 1)
        axis.set_title(f'{plot_title}: silhouette score ({args.fixed_clusters} clusters)')
        axis.plot(summary_df.index, summary_df['sil_score'])
        if args.save_summary:
            figure.set_size_inches(32, 18)
            filename = f'images/clusters_fixed_{args.fixed_clusters}_{filename_title}.png'
            print(f'Saving summary to file {filename}')
            plt.savefig(filename)
    else:
        figure, axes = plt.subplots(2, 1)
        axes = axes.flatten()
        axes[0].set_title(f'{plot_title}: optimal number of clusters')
        axes[0].plot(summary_df.index, summary_df['num_clusters'])
        axes[1].set_title(f'{plot_title}: silhouette score')
        axes[1].plot(summary_df.index, summary_df['sil_score'])
        if args.save_summary:
            figure.set_size_inches(32, 18)
            filename = f'images/clusters_optimal_{filename_title}.png'
            print(f'Saving summary to file {filename}')
            plt.savefig(filename)
    print(f'Average silhouette score: {summary_df["sil_score"].mean()}')
    if not args.save_summary:
        plt.show()

if __name__ == '__main__':
    start = datetime.now()
    main()
    print(f'Total time elapsed: {datetime.now() - start}')