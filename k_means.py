from datetime import datetime
import argparse

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from k_means_constrained import KMeansConstrained

from helpers import get_map, prepare_indexed_df

YEARS = list(range(2000, 2022 + 1))
DATA_DIR = 'data/precipitation'
DATA_FILE = f'{DATA_DIR}/FusedData.csv'
LOCATIONS_FILE = f'{DATA_DIR}/Fused.Locations.csv'
# Ordered to give "rainbow" of increasing precipitation values in clusters
COLOURS = {
    2: ['#aaa', 'red'],
    3: ['#aaa', 'green', 'red'],
    4: ['#aaa', 'green', 'orange', 'red'],
    5: ['#aaa', '#666', 'green', 'orange', 'red'],
    6: ['#aaa', '#666', 'cyan', 'green', 'orange', 'red'],
    7: ['#aaa', '#666', 'blue', 'cyan', 'green', 'orange', 'red'],
    8: ['#aaa', '#666', 'blue', 'cyan', 'green', 'gold', 'orange', 'red'],
    9: ['#aaa', '#666', 'saddlebrown', 'blue', 'cyan', 'green', 'gold', 'orange', 'red'],
    10: ['#aaa', '#666', 'saddlebrown', 'blue', 'cyan', 'green', 'gold', 'orange', 'red', 'magenta'],
}

def kmeans_optimise(train_data: np.array, max_clusters, constrain):
    sil_scores_dict = {}
    kmeans_dict = {}
    for n in range(2, max_clusters + 1):
        if constrain:
            kmeans_dict[n] = KMeansConstrained(n_clusters=n, random_state=0,
                size_max=int(len(train_data) * constrain + 1)).fit(train_data)
        else:
            kmeans_dict[n] = KMeans(n_clusters=n, random_state=0).fit(train_data)
        sil_scores_dict[n] = silhouette_score(train_data, kmeans_dict[n].labels_)
    optimal_num_clusters = max(sil_scores_dict, key=sil_scores_dict.get)
    return optimal_num_clusters, kmeans_dict[optimal_num_clusters], sil_scores_dict[optimal_num_clusters]

def kmeans_optimise_gap(data, max_clusters, nrefs=5):
    gaps_dict = {}
    kmeans_dict = {}
    # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
    for n in range(2, max_clusters + 1):
        # Holder for reference dispersion results
        ref_disps = np.zeros(nrefs)
        for i in range(nrefs):
            # Create new random reference set
            random_reference = np.random.random_sample(size=data.shape)
            # Fit to it
            km = KMeans(n_clusters=n)
            km.fit(random_reference)
            ref_disps[i] = km.inertia_
        # Fit cluster to original data and create dispersion
        kmeans_dict[n] = km = KMeans(n_clusters=n)
        km.fit(data)
        orig_disp = km.inertia_
        # Calculate gap statistic
        gap = np.log(np.mean(ref_disps)) - np.log(orig_disp)
        gaps_dict[n] = gap
    optimal_num_clusters = max(gaps_dict, key=gaps_dict.get)
    return optimal_num_clusters, kmeans_dict[optimal_num_clusters], gaps_dict[optimal_num_clusters]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--month', type=int, default=None)
    parser.add_argument('--fixed_clusters', type=int, default=None)
    parser.add_argument('--max_clusters', type=int, default=10)
    # Fraction of total number of points as upper bound on cluster size
    parser.add_argument('--constrain', type=float, default=None)
    parser.add_argument('--gap', action='store_true', default=False)
    parser.add_argument('--plot_clusters', action='store_true', default=False)
    parser.add_argument('--save_summary', action='store_true', default=False)
    parser.add_argument('--merged_summary', action='store_true', default=False)
    args = parser.parse_args()

    raw_df = pd.read_csv(DATA_FILE)
    locations_df = pd.read_csv(LOCATIONS_FILE)
    df = prepare_indexed_df(raw_df, locations_df, month=args.month, new_index='date')

    months = [args.month] if args.month else list(range(1, 13))
    lons = locations_df['Lon']
    lats = locations_df['Lat']
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
                if args.gap:
                    num_clusters, kmeans, gap = kmeans_optimise_gap(train_data,
                        args.max_clusters)
                    sil_score = silhouette_score(train_data, kmeans.labels_)
                    print(f'{dt.strftime("%b")} {y}: {num_clusters} optimal clusters with silhouette score {sil_score}'
                        f' and gap statistic {gap}')
                else:
                    num_clusters, kmeans, sil_score = kmeans_optimise(train_data,
                        args.max_clusters, args.constrain)
                    print(f'{dt.strftime("%b")} {y}: {num_clusters} optimal clusters with silhouette score {sil_score}')
            summary_df.append({
                'dt': dt,
                'num_clusters': num_clusters,
                'sil_score': sil_score,
            })
            if args.plot_clusters:
                # Order the colours based on average precipitation in clusters
                cluster_means, cluster_st_devs, cluster_sizes = [], [], []
                for n in set(kmeans.labels_):
                    cluster_indices = [i for i, label in enumerate(kmeans.labels_) if label == n]
                    cluster_slice = location_df.iloc[cluster_indices]['prec']
                    cluster_means.append(cluster_slice.mean())
                    cluster_st_devs.append(cluster_slice.std())
                    cluster_sizes.append(len(cluster_slice))
                permuted_colours = [0] * len(cluster_means)
                for i_old, i_new in enumerate(np.argsort(cluster_means)):
                    permuted_colours[i_new] = COLOURS[len(cluster_means)][i_old]
                    print(f'cluster ({i_old}, {permuted_colours[i_new]}):',
                        f'size = {round(cluster_sizes[i_new], 3)},',
                        f'mean = {round(cluster_means[i_new], 3)},',
                        f'st dev = {round(cluster_st_devs[i_new], 3)}',
                    )
                node_colours = [permuted_colours[cluster] for cluster in kmeans.labels_]
                figure, axis = plt.subplots(1)
                mx, my = get_map(axis)(lons, lats)
                cmap = mpl.colors.ListedColormap([permuted_colours[i] for i in range(num_clusters)])
                title = f'{dt.strftime("%b")} {y}: k-means ({num_clusters} clusters, silhouette score {round(sil_score, 3)})'
                axis.set_title(title)
                axis.scatter(mx, my, c=node_colours, cmap=cmap)
                plt.show()

    summary_df = pd.DataFrame.from_dict(summary_df).set_index(['dt'])
    if args.month:
        plot_title = dt.strftime("%B")
        filename_title = f'm{args.month}' if args.month >= 10 else f'm0{args.month}'
    elif args.merged_summary:
        filename_title = 'merged'
    else:
        plot_title = 'whole_series'
        filename_title = 'whole_series'
    top_month_dfs = []
    bottom_month_dfs = []
    top_month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    bottom_month_labels = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    jan_dts = [datetime(y, 1, 1) for y in YEARS]
    for m in range(1, 7):
        month_df = summary_df.loc[summary_df.index.month == m]
        top_month_dfs.append(month_df)
    for m in range(7, 13):
        month_df = summary_df.loc[summary_df.index.month == m]
        bottom_month_dfs.append(month_df)
    if args.fixed_clusters:
        if args.merged_summary:
            figure, axes = plt.subplots(2, 1)
            axes = axes.flatten()
            axes[0].set_title(f'Jan to Jun: silhouette score ({args.fixed_clusters} clusters)')
            axes[1].set_title(f'Jul to Dec: silhouette score ({args.fixed_clusters} clusters)')
            for _df in top_month_dfs:
                axes[0].plot(jan_dts[0 : len(_df)], _df['sil_score'])
            for _df in bottom_month_dfs:
                axes[1].plot(jan_dts[0 : len(_df)], _df['sil_score'])
            axes[0].legend(labels=top_month_labels)
            axes[1].legend(labels=bottom_month_labels)
        else:
            figure, axis = plt.subplots(1, 1)
            axis.set_title(f'{plot_title}: silhouette score ({args.fixed_clusters} clusters)')
            axis.plot(summary_df.index, summary_df['sil_score'])
        if args.save_summary:
            figure.set_size_inches(32, 18)
            filename = f'images/clusters{"_constrained" if args.constrain else ""}_fixed_{args.fixed_clusters}_{filename_title}.png'
            print(f'Saving summary to file {filename}')
            plt.savefig(filename)
    else:
        if args.merged_summary:
            figure, axes = plt.subplots(2, 2)
            axes = axes.flatten()
            axes[0].set_title(f'Jan to Jun: optimal number of clusters')
            axes[1].set_title(f'Jan to Jun: silhouette score')
            axes[2].set_title(f'Jul to Dec: optimal number of clusters')
            axes[3].set_title(f'Jul to Dec: silhouette score')
            for _df in top_month_dfs:
                axes[0].plot(jan_dts[0 : len(_df)], _df['num_clusters'])
                axes[1].plot(jan_dts[0 : len(_df)], _df['sil_score'])
            for _df in bottom_month_dfs:
                axes[2].plot(jan_dts[0 : len(_df)], _df['num_clusters'])
                axes[3].plot(jan_dts[0 : len(_df)], _df['sil_score'])
            axes[0].legend(labels=top_month_labels)
            axes[1].legend(labels=top_month_labels)
            axes[2].legend(labels=bottom_month_labels)
            axes[3].legend(labels=bottom_month_labels)
        else:
            figure, axes = plt.subplots(2, 1)
            axes = axes.flatten()
            axes[0].set_title(f'{plot_title}: optimal number of clusters')
            axes[0].plot(summary_df.index, summary_df['num_clusters'])
            axes[1].set_title(f'{plot_title}: silhouette score')
            axes[1].plot(summary_df.index, summary_df['sil_score'])
        if args.save_summary:
            figure.set_size_inches(32, 18)
            filename = f'images/clusters{"_constrained" if args.constrain else ""}_optimal{"_gap" if args.gap else ""}_{filename_title}.png'
            print(f'Saving summary to file {filename}')
            plt.savefig(filename)
    print(f'Average silhouette score: {summary_df["sil_score"].mean()}')
    if not args.save_summary:
        plt.show()

if __name__ == '__main__':
    start = datetime.now()
    main()
    print(f'Total time elapsed: {datetime.now() - start}')