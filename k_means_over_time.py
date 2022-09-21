from datetime import datetime
import argparse

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from helpers import get_map
from k_means import kmeans_optimise

DATA_DIR = 'data/precipitation'
DATA_FILE = f'{DATA_DIR}/FusedData.csv'
LOCATIONS_FILE = f'{DATA_DIR}/Fused.Locations.csv'
COLOURS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 'black', '#444', '#ccc']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixed_clusters', type=int, default=None)
    parser.add_argument('--max_clusters', type=int, default=8)
    # Fraction of total number of points as upper bound on cluster size
    parser.add_argument('--constrain', type=float, default=None)
    parser.add_argument('--min_year', type=int, default=2000)
    parser.add_argument('--max_year', type=int, default=2022)
    parser.add_argument('--save_plots', action='store_true', default=False)
    args = parser.parse_args()

    raw_df = pd.read_csv(DATA_FILE)
    raw_df.columns = pd.to_datetime(raw_df.columns, format='D%Y.%m')
    locations_df = pd.read_csv(LOCATIONS_FILE)
    lons = locations_df['Lon']
    lats = locations_df['Lat']

    def _kmeans_func(train_data, month=None):
        if args.fixed_clusters:
            num_clusters = args.fixed_clusters
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(train_data)
            sil_score = silhouette_score(train_data, kmeans.labels_)
        else:
            num_clusters, kmeans, sil_score = kmeans_optimise(train_data,
                args.max_clusters, args.constrain)
        figure, axis = plt.subplots(1)
        mx, my = get_map(axis)(lons, lats)
        cmap = mpl.colors.ListedColormap([COLOURS[i] for i in range(num_clusters)])
        title = (f'{datetime(2022, month, 1).strftime("%B") if month else "full_series"}'
            f' ({args.min_year} to {args.max_year}):'
            f' k-means ({num_clusters} clusters, silhouette score {round(sil_score, 3)})')
        node_colours = [COLOURS[cluster] for cluster in kmeans.labels_]
        axis.set_title(title)
        if args.save_plots:
            axis.scatter(mx, my, c=node_colours, cmap=cmap, s=150)
            figure.set_size_inches(32, 18)
            if not month:
                filename_title = 'full_series'
            elif month >= 10:
                filename_title = f'm{month}'
            else:
                filename_title = f'm0{month}'
            cluster_title = f'fixed_{args.fixed_clusters}' if args.fixed_clusters else 'optimal'
            filename = f'images/clusters{"_constrained" if args.constrain else ""}_{cluster_title}_over_time_{filename_title}_{args.min_year}_{args.max_year}.png'
            print(f'Saving plot to file {filename}')
            plt.savefig(filename)
        else:
            axis.scatter(mx, my, c=node_colours, cmap=cmap)
            plt.show()
        return num_clusters, kmeans, sil_score

    def _column_filter(column: datetime, month=None):
        year_filter = column.year >= args.min_year and column.year <= args.max_year
        if month:
            return year_filter and column.month == m
        return year_filter

    df = raw_df.loc[:, [_column_filter(c) for c in raw_df.columns]]
    num_clusters, kmeans, sil_score = _kmeans_func(np.array(df))
    for m in range(1, 13):
        df = raw_df.loc[:, [_column_filter(c, m) for c in raw_df.columns]]
        num_clusters, kmeans, sil_score = _kmeans_func(np.array(df), m)

if __name__ == '__main__':
    start = datetime.now()
    main()
    print(f'Total time elapsed: {datetime.now() - start}')