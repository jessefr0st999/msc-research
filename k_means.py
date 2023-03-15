import argparse
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from k_means_constrained import KMeansConstrained

from helpers import configure_plots, get_map, prepare_df

# Note that intra-annual analysis may be affected for 2000 and 2022, as these
# years lack data for all months

KMEANS_COLOURS = {
    2: ['#aaa', 'red'],
    3: ['#aaa', 'green', 'red'],
    4: ['#aaa', 'green', 'orange', 'red'],
    5: ['#aaa', '#666', 'green', 'orange', 'red'],
    6: ['#aaa', '#666', 'green', 'gold', 'orange', 'red'],
    7: ['#aaa', '#666', 'cyan', 'green', 'gold', 'orange', 'red'],
    8: ['#aaa', '#666', 'blue', 'cyan', 'green', 'gold', 'orange', 'red'],
    9: ['#aaa', '#666', 'saddlebrown', 'blue', 'cyan', 'green', 'gold', 'orange', 'red'],
    10: ['#aaa', '#666', 'saddlebrown', 'blue', 'cyan', 'green', 'gold', 'orange', 'red', 'magenta'],
}
# Decade 1: April 2000 to March 2011
# Decade 2: April 2011 to March 2022
DECADE_DATES = [datetime(2000, 4 ,1), datetime(2011, 3 ,1),
    datetime(2011, 4 ,1), datetime(2022, 3 ,1)]

def plot_clusters(series, kmeans_labels, axis, lons, lats, aus=True,
        dot_size=20, region='aus'):
    # Order the colours based on average values in clusters
    cluster_means = []
    for k in set(kmeans_labels):
        cluster_indices = [i for i, label in enumerate(kmeans_labels) if label == k]
        cluster_slice = series[cluster_indices]
        cluster_means.append(cluster_slice.mean())
    permuted_colours = [0] * len(cluster_means)
    for i_old, i_new in enumerate(np.argsort(cluster_means)):
        permuted_colours[i_new] = KMEANS_COLOURS[len(cluster_means)][i_old]
    node_colours = [permuted_colours[cluster] for cluster in kmeans_labels]
    _map = get_map(axis, region=region)
    mx, my = _map(lons, lats)
    axis.scatter(mx, my, c=node_colours, s=dot_size)

def kmeans_fit(k, array, constrain=None):
    if constrain:
        kmeans = KMeansConstrained(n_clusters=k, random_state=0,
            size_max=int(len(array) * constrain + 1)).fit(array)
    else:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(array)
    sil_score = silhouette_score(array, kmeans.labels_)
    return kmeans.labels_, sil_score

# NOTE: currently not used
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
            km = KMeans(n_clusters=n, n_init='auto')
            km.fit(random_reference)
            ref_disps[i] = km.inertia_
        # Fit cluster to original data and create dispersion
        kmeans_dict[n] = km = KMeans(n_clusters=n, n_init='auto')
        km.fit(data)
        orig_disp = km.inertia_
        # Calculate gap statistic
        gap = np.log(np.mean(ref_disps)) - np.log(orig_disp)
        gaps_dict[n] = gap
    optimal_num_clusters = max(gaps_dict, key=gaps_dict.get)
    return optimal_num_clusters, kmeans_dict[optimal_num_clusters], gaps_dict[optimal_num_clusters]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', default=None)
    parser.add_argument('--data_dir', default='data/precipitation')
    parser.add_argument('--data_file', default='FusedData.csv')
    # Fraction of total number of points as upper bound on cluster size
    parser.add_argument('--constrain', type=float, default=None)
    # TODO: implement using helper above
    parser.add_argument('--gap', action='store_true', default=False)
    parser.add_argument('--decadal', action='store_true', default=False)
    parser.add_argument('--monthly', action='store_true', default=False)
    parser.add_argument('--monthly_averages', action='store_true', default=False)
    args = parser.parse_args()
    label_size, font_size, show_or_save = configure_plots(args)

    dataset = 'prec' if args.data_file == 'FusedData.csv' else args.data_file.split('_')[0]
    df, lats, lons = prepare_df(args.data_dir, args.data_file, dataset)
    map_region = 'aus' if dataset == 'prec' else 'world'

    # First, calculate and print silhouette scores for k = 2, ..., 12
    k_list = list(range(2, 12 + 1))
    if args.decadal:
        sils_df = pd.DataFrame(0, index=k_list, columns=['d1', 'd2'])
        d1 = np.array(df.loc[DECADE_DATES[0] : DECADE_DATES[1], :]).T
        d2 = np.array(df.loc[DECADE_DATES[2] : DECADE_DATES[3], :]).T
        for j, k in enumerate(k_list):
            # If the constraint times the number of clusters is less than the
            # number of points, a ValueError is raised
            try:
                _, sils_df.iloc[j, 0] = kmeans_fit(k, d1, args.constrain)
                _, sils_df.iloc[j, 1] = kmeans_fit(k, d2, args.constrain)
            except ValueError:
                continue
    elif args.monthly:
        sils_df = pd.DataFrame(0, index=k_list, columns=[
            *[f'd1_m{m}' for m in range(1, 12 + 1)],
            *[f'd2_m{m}' for m in range(1, 12 + 1)],
        ])
        d1 = df.loc[DECADE_DATES[0] : DECADE_DATES[1], :]
        d2 = df.loc[DECADE_DATES[2] : DECADE_DATES[3], :]
        for i, m in enumerate(range(1, 12 + 1)):
            d1_m = np.array(d1.loc[[dt.month == m for dt in d1.index], :]).T
            d2_m = np.array(d2.loc[[dt.month == m for dt in d2.index], :]).T
            try:
                for j, k in enumerate(k_list):
                    _, sils_df.iloc[j, i] = kmeans_fit(k, d1_m, args.constrain)
                    _, sils_df.iloc[j, 12 + i] = kmeans_fit(k, d2_m, args.constrain)
            except ValueError:
                continue
    elif args.monthly_averages:
        sils_df = pd.DataFrame(0, index=k_list, columns=['d1', 'd2'])
        d1 = df.loc[DECADE_DATES[0] : DECADE_DATES[1], :]
        d2 = df.loc[DECADE_DATES[2] : DECADE_DATES[3], :]
        for j, k in enumerate(k_list):
            d1_m_av = np.zeros((d1.shape[1], 12))
            d2_m_av = np.zeros((d2.shape[1], 12))
            for i, m in enumerate(range(1, 12 + 1)):
                d1_m_av[:, i] = d1.loc[[dt.month == m for dt in d1.index], :].mean(axis=0)
                d2_m_av[:, i] = d2.loc[[dt.month == m for dt in d2.index], :].mean(axis=0)
            _, sils_df.iloc[j, 0] = kmeans_fit(k, d1_m_av, args.constrain)
            _, sils_df.iloc[j, 1] = kmeans_fit(k, d2_m_av, args.constrain)
    else:
        sils_df = pd.Series(0, index=k_list)
        for j, k in enumerate(k_list):
            try:
                sils = []
                for i, dt in enumerate(df.index.values):
                    if i % 50 == 0:
                        print(f'{j} / {len(k_list)}, {i} / {len(df.index.values)}')
                    slid_dt = np.array(df.iloc[i, :]).reshape(-1, 1)
                    _, sil = kmeans_fit(k, slid_dt, args.constrain)
                    sils.append(sil)
                sils_df.iloc[j] = np.mean(sils)
            except ValueError:
                continue
    sils_df.index.name = 'num_clusters'
    print(sils_df)

    # Next, plot results for k = 2, ..., 9
    fig_name_base = f'{dataset}_kmeans'
    if args.constrain:
        fig_name_base += f'_c{round(100 * args.constrain)}'
    k_list = list(range(2, 9 + 1))
    dot_size = 100 if args.output_folder else 20
    if args.decadal:
        d1 = np.array(df.loc[DECADE_DATES[0] : DECADE_DATES[1], :]).T
        d2 = np.array(df.loc[DECADE_DATES[2] : DECADE_DATES[3], :]).T
        figure, axes = plt.subplots(2, 4, layout='compressed')
        axes = iter(axes.flatten())
        for k in k_list:
            labels_d1, sil_d1 = kmeans_fit(k, d1, args.constrain)
            axis = next(axes)
            plot_clusters(d1, labels_d1, axis, lons, lats, dot_size=dot_size,
                region=map_region)
            axis.set_title(f'Decade 1, {k} clusters, sil score {round(sil_d1, 4)}')
        show_or_save(figure, f'{fig_name_base}_decade_1.png')
        figure, axes = plt.subplots(2, 4, layout='compressed')
        axes = iter(axes.flatten())
        for k in k_list:
            labels_d2, sil_d2 = kmeans_fit(k, d2, args.constrain)
            axis = next(axes)
            plot_clusters(d2, labels_d2, axis, lons, lats, dot_size=dot_size,
                region=map_region)
            axis.set_title(f'Decade 2, {k} clusters, sil score {round(sil_d2, 4)}')
        show_or_save(figure, f'{fig_name_base}_decade_2.png')
    elif args.monthly:
        d1 = df.loc[DECADE_DATES[0] : DECADE_DATES[1], :]
        d2 = df.loc[DECADE_DATES[2] : DECADE_DATES[3], :]
        for k in k_list:
            figure, axes = plt.subplots(3, 4, layout='compressed')
            axes = iter(axes.flatten())
            for m in range(1, 12 + 1):
                d1_m = np.array(d1.loc[[dt.month == m for dt in d1.index], :]).T
                labels_d1, sil_d1 = kmeans_fit(k, d1_m, args.constrain)
                axis = next(axes)
                plot_clusters(d1_m, labels_d1, axis, lons, lats, dot_size=dot_size,
                    region=map_region)
                axis.set_title(f'{datetime(2000, m, 1).strftime("%b")} D1, '
                    f'k = {k}, sil score {round(sil_d1, 4)}')
            show_or_save(figure, f'{fig_name_base}_months_decade_1_k{k}.png')
                
            figure, axes = plt.subplots(3, 4, layout='compressed')
            axes = iter(axes.flatten())
            for m in range(1, 12 + 1):
                d2_m = np.array(d2.loc[[dt.month == m for dt in d2.index], :]).T
                labels_d2, sil_d2 = kmeans_fit(k, d2_m, args.constrain)
                axis = next(axes)
                plot_clusters(d2_m, labels_d2, axis, lons, lats, dot_size=dot_size,
                    region=map_region)
                axis.set_title(f'{datetime(2000, m, 1).strftime("%b")} D2, '
                    f'k = {k}, sil score {round(sil_d2, 4)}')
            show_or_save(figure, f'{fig_name_base}_months_decade_2_k{k}.png')
    elif args.monthly_averages:
        d1 = df.loc[DECADE_DATES[0] : DECADE_DATES[1], :]
        d2 = df.loc[DECADE_DATES[2] : DECADE_DATES[3], :]
        for k in k_list:
            d1_m_av = np.zeros((d1.shape[1], 12))
            d2_m_av = np.zeros((d2.shape[1], 12))
            for i, m in enumerate(range(1, 12 + 1)):
                d1_m_av[:, i] = d1.loc[[dt.month == m for dt in d1.index], :].mean(axis=0)
                d2_m_av[:, i] = d2.loc[[dt.month == m for dt in d2.index], :].mean(axis=0)
            labels_d1, sil_d1 = kmeans_fit(k, d1_m_av, args.constrain)
            labels_d2, sil_d2 = kmeans_fit(k, d2_m_av, args.constrain)

            figure, axes = plt.subplots(1, 2, layout='compressed')
            axes = iter(axes.flatten())
            axis = next(axes)
            plot_clusters(d1_m_av, labels_d1, axis, lons, lats, dot_size=300,
                region=map_region)
            axis.set_title(f'D1, k = {k}, sil score {round(sil_d1, 4)}')
            
            axis = next(axes)
            plot_clusters(d2_m_av, labels_d2, axis, lons, lats, dot_size=300,
                region=map_region)
            axis.set_title(f'D2, k = {k}, sil score {round(sil_d2, 4)}')
            show_or_save(figure, f'{fig_name_base}_monthly_averages_k{k}.png')
    else:
        for i, dt in enumerate(df.index.values):
            _dt = pd.to_datetime(dt)
            dt = np.array(df.iloc[i, :]).reshape(-1, 1)
            figure, axes = plt.subplots(2, 4, layout='compressed')
            axes = iter(axes.flatten())
            for k in k_list:
                labels, sil = kmeans_fit(k, dt, args.constrain)
                axis = next(axes)
                plot_clusters(dt, labels, axis, lons, lats, dot_size=dot_size,
                    region=map_region)
                axis.set_title(f'{_dt.strftime("%b %Y")} {k} clusters, sil score {round(sil, 4)}')
            show_or_save(figure, f'{fig_name_base}_{_dt.strftime("%Y_%m")}.png')

if __name__ == '__main__':
    start = datetime.now()
    main()
    print(f'Total time elapsed: {datetime.now() - start}')