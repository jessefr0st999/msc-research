import argparse
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.spatial import distance
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
cmap = get_cmap('hsv')
MONTH_COLOURS = [cmap(i / 12) for i in range(12)]
# Decade 1: April 2000 to March 2011
# Decade 2: April 2011 to March 2022
DECADE_DATES = [datetime(2000, 4 ,1), datetime(2011, 3 ,1),
    datetime(2011, 4 ,1), datetime(2022, 3 ,1)]

k7_orange_indices = [615, 616, 1170, 1171, 1172, 1173, 1215, 1217, 1218, 1253, 1262,
    1296, 1297, 1328, 1330, 1331, 1352, 1353, 1354, 1355, 1356, 1366, 1369, 1370,
    1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382, 1383,
    1384, 1385, 1386, 1387, 1388, 1389, 1390]

def plot_clusters(series, kmeans_labels, axis, lons, lats, aus=True,
        dot_size=20, region='aus'):
    # Order the colours based on average values in clusters
    cluster_means = []
    for k in set(kmeans_labels):
        cluster_indices = [i for i, label in enumerate(kmeans_labels) if label == k]
        cluster_slice = series[cluster_indices]
        # print(k, cluster_slice.mean())
        # if k == 6:
        #     print(cluster_indices)
        cluster_means.append(cluster_slice.mean())
    permuted_colours = [0] * len(cluster_means)
    for i_old, i_new in enumerate(np.argsort(cluster_means)):
        permuted_colours[i_new] = KMEANS_COLOURS[len(cluster_means)][i_old]
    # for i in k7_orange_indices:
    #     print(i, lons[i], lats[i])
    node_colours = [permuted_colours[cluster] for cluster in kmeans_labels]
    _map = get_map(axis, region=region)
    mx, my = _map(lons, lats)
    axis.scatter(mx, my, c=node_colours, s=dot_size)

# def kmeans_aic(kmeans_fit, array):
#     m = len(kmeans_fit.cluster_centers_)
#     k = len(kmeans_fit.cluster_centers_[0])
#     return rss + 2 * m * k

# def kmeans_bic(kmeans_fit, array):
#     centres = [kmeans_fit.cluster_centers_]
#     labels  = kmeans_fit.labels_
#     m = kmeans_fit.n_clusters
#     # Cluster size
#     n = np.bincount(labels)
#     # Dataset size
#     N, d = array.shape

#     # Variance for all clusters
#     cluster_var = (1.0 / (N - m) / d) * sum([
#         sum(distance.cdist(
#             array[np.where(labels == i)], [centres[0][i]], 'euclidean'
#         )**2)
#     for i in range(m)])

#     const_term = 0.5 * m * np.log(N) * (d + 1)
#     return np.sum([n[i] * np.log(n[i]) -
#         n[i] * np.log(N) -
#         ((n[i] * d) / 2) * np.log(2 * np.pi * cluster_var) -
#         ((n[i] - 1) * d/ 2)
#     for i in range(m)]) - const_term

def kmeans_fit(k, array, constrain=None):
    if constrain:
        kmeans = KMeansConstrained(n_clusters=k, random_state=0,
            size_max=int(len(array) * constrain + 1)).fit(array)
    else:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(array)
    sil_score = silhouette_score(array, kmeans.labels_)
    # aic = kmeans_aic(kmeans, array)
    # bic = kmeans_bic(kmeans, array)
    # return kmeans.labels_, sil_score, aic, bic
    return kmeans.labels_, sil_score

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
    parser.add_argument('--dma', action='store_true', default=False)
    parser.add_argument('--scores', action='store_true', default=False)
    args = parser.parse_args()
    label_size, font_size, show_or_save = configure_plots(args)

    dataset = 'prec' if args.data_file == 'FusedData.csv' else args.data_file.split('_')[0]
    df, lats, lons = prepare_df(args.data_dir, args.data_file, dataset)
    map_region = 'aus' if dataset == 'prec' else 'world'

    if args.scores:
        # Calculate and print silhouette scores for k = 2, ..., 12
        k_list = list(range(2, 12 + 1))
        sils_decadal_df = pd.DataFrame(0, index=k_list, columns=['sil_d1', 'sil_d2'])
        sils_decadal_df.index.name = 'num_clusters_decadal'
        d1 = np.array(df.loc[DECADE_DATES[0] : DECADE_DATES[1], :]).T
        d2 = np.array(df.loc[DECADE_DATES[2] : DECADE_DATES[3], :]).T
        for j, k in enumerate(k_list):
            # If the constraint times the number of clusters is less than the
            # number of points, a ValueError is raised
            try:
                _, sils_decadal_df.iloc[j, 0] = kmeans_fit(k, d1, args.constrain)
                _, sils_decadal_df.iloc[j, 1] = kmeans_fit(k, d2, args.constrain)
            except ValueError:
                continue

        sils_monthly_df = pd.DataFrame(0, index=k_list, columns=[
            *[f'sil_d1_m{m}' for m in range(1, 12 + 1)],
            *[f'sil_d2_m{m}' for m in range(1, 12 + 1)],
        ])
        sils_monthly_df.index.name = 'num_clusters_monthly'
        d1 = df.loc[DECADE_DATES[0] : DECADE_DATES[1], :]
        d2 = df.loc[DECADE_DATES[2] : DECADE_DATES[3], :]
        for i, m in enumerate(range(1, 12 + 1)):
            d1_m = np.array(d1.loc[[dt.month == m for dt in d1.index], :]).T
            d2_m = np.array(d2.loc[[dt.month == m for dt in d2.index], :]).T
            try:
                for j, k in enumerate(k_list):
                    _, sils_monthly_df.iloc[j, i] = kmeans_fit(k, d1_m, args.constrain)
                    _, sils_monthly_df.iloc[j, 12 + i] = kmeans_fit(k, d2_m, args.constrain)
            except ValueError:
                continue

        sils_dma_df = pd.DataFrame(0, index=k_list, columns=['sil_d1', 'sil_d2'])
        sils_dma_df.index.name = 'num_clusters_dma'
        d1 = df.loc[DECADE_DATES[0] : DECADE_DATES[1], :]
        d2 = df.loc[DECADE_DATES[2] : DECADE_DATES[3], :]
        for j, k in enumerate(k_list):
            d1_m_av = np.zeros((d1.shape[1], 12))
            d2_m_av = np.zeros((d2.shape[1], 12))
            for i, m in enumerate(range(1, 12 + 1)):
                d1_m_av[:, i] = d1.loc[[dt.month == m for dt in d1.index], :].mean(axis=0)
                # d1_m_av[:, i] /= d1_m_av[:, i].mean()
                d2_m_av[:, i] = d2.loc[[dt.month == m for dt in d2.index], :].mean(axis=0)
                # d2_m_av[:, i] /= d2_m_av[:, i].mean()
            _, sils_dma_df.iloc[j, 0] = kmeans_fit(k, d1_m_av, args.constrain)
            _, sils_dma_df.iloc[j, 1] = kmeans_fit(k, d2_m_av, args.constrain)

        figure, axis = plt.subplots(1)
        monthly_average_d1 = sils_monthly_df.iloc[:, :12].mean(axis=1)
        monthly_average_d2 = sils_monthly_df.iloc[:, 12:].mean(axis=1)
        axis.plot(sils_decadal_df['sil_d1'], '-or', label='decadal, D1')
        axis.plot(sils_decadal_df['sil_d2'], '-om', label='decadal, D2')
        axis.plot(monthly_average_d1, '-og', label='monthly (average), D1')
        axis.plot(monthly_average_d2, '-oy', label='monthly (average), D2')
        axis.plot(sils_dma_df['sil_d1'], '-ob', label='DMA, D1')
        axis.plot(sils_dma_df['sil_d2'], '-oc', label='DMA, D2')
        axis.set_ylim([0, 1])
        axis.legend()

        figure, axes = plt.subplots(1, 2)
        axes = iter(axes.flatten())
        axis = next(axes)
        for m in range(12):
            axis.plot(sils_monthly_df.iloc[:, m], '-o', color=MONTH_COLOURS[m],
                label=f'{datetime(2000, m + 1, 1).strftime("%b")}, D1')
        axis.set_ylim([0, 1])
        axis.legend()
        
        axis = next(axes)
        for m in range(12):
            axis.plot(sils_monthly_df.iloc[:, m + 12], '-o', color=MONTH_COLOURS[m],
                label=f'{datetime(2000, m + 1, 1).strftime("%b")}, D2')
        axis.set_ylim([0, 1])
        axis.legend()
        plt.show()
        return

    # Next, plot results for k = 2, ..., 9
    fig_name_base = f'{dataset}_kmeans'
    if args.constrain:
        fig_name_base += f'_c{round(100 * args.constrain)}'
    # k_list = list(range(2, 9 + 1))
    k_list = [8]
    if args.decadal:
        d1 = np.array(df.loc[DECADE_DATES[0] : DECADE_DATES[1], :]).T
        d2 = np.array(df.loc[DECADE_DATES[2] : DECADE_DATES[3], :]).T
        # figure, axes = plt.subplots(2, 4, layout='compressed')
        # axes = iter(axes.flatten())
        # for k in k_list:
        #     labels_d1, sil_d1 = kmeans_fit(k, d1, args.constrain)
        #     axis = next(axes)
        #     plot_clusters(d1, labels_d1, axis, lons, lats, region=map_region,
        #         dot_size=dot_size)
        #     axis.set_title(f'Decade 1, {k} clusters, sil score {round(sil_d1, 4)}')
        # show_or_save(figure, f'{fig_name_base}_decade_1.png')
        # figure, axes = plt.subplots(2, 4, layout='compressed')
        # axes = iter(axes.flatten())
        # for k in k_list:
        #     labels_d2, sil_d2 = kmeans_fit(k, d2, args.constrain)
        #     axis = next(axes)
        #     plot_clusters(d2, labels_d2, axis, lons, lats, region=map_region,
        #         dot_size=dot_size)
        #     axis.set_title(f'Decade 2, {k} clusters, sil score {round(sil_d2, 4)}')
        # show_or_save(figure, f'{fig_name_base}_decade_2.png')
        for k in k_list:
            labels_d1, sil_d1 = kmeans_fit(k, d1, args.constrain)
            labels_d2, sil_d2 = kmeans_fit(k, d2, args.constrain)

            figure, axes = plt.subplots(1, 2, layout='compressed')
            axes = iter(axes.flatten())
            axis = next(axes)
            plot_clusters(d1, labels_d1, axis, lons, lats, dot_size=75,
                region=map_region)
            axis.set_title(f'Decade 1, k = {k}, sil score {round(sil_d1, 4)}')
            
            axis = next(axes)
            plot_clusters(d2, labels_d2, axis, lons, lats, dot_size=75,
                region=map_region)
            axis.set_title(f'Decade 2, k = {k}, sil score {round(sil_d2, 4)}')
            show_or_save(figure, f'{fig_name_base}_decadal_k{k}.png')
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
                plot_clusters(d1_m, labels_d1, axis, lons, lats, dot_size=15,
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
                plot_clusters(d2_m, labels_d2, axis, lons, lats, dot_size=15,
                    region=map_region)
                axis.set_title(f'{datetime(2000, m, 1).strftime("%b")} D2, '
                    f'k = {k}, sil score {round(sil_d2, 4)}')
            show_or_save(figure, f'{fig_name_base}_months_decade_2_k{k}.png')
    elif args.dma:
        d1 = df.loc[DECADE_DATES[0] : DECADE_DATES[1], :]
        d2 = df.loc[DECADE_DATES[2] : DECADE_DATES[3], :]
        for k in k_list:
            d1_m_av = np.zeros((d1.shape[1], 12))
            d2_m_av = np.zeros((d2.shape[1], 12))
            for i, m in enumerate(range(1, 12 + 1)):
                d1_m_av[:, i] = d1.loc[[dt.month == m for dt in d1.index], :].mean(axis=0)
                # d1_m_av[:, i] /= d1_m_av[:, i].mean()
                d2_m_av[:, i] = d2.loc[[dt.month == m for dt in d2.index], :].mean(axis=0)
                # d2_m_av[:, i] /= d2_m_av[:, i].mean()
            labels_d1, sil_d1 = kmeans_fit(k, d1_m_av, args.constrain)
            labels_d2, sil_d2 = kmeans_fit(k, d2_m_av, args.constrain)

            figure, axes = plt.subplots(1, 2, layout='compressed')
            axes = iter(axes.flatten())
            axis = next(axes)
            plot_clusters(d1_m_av, labels_d1, axis, lons, lats, dot_size=75,
                region=map_region)
            axis.set_title(f'Decade 1, k = {k}, sil score {round(sil_d1, 4)}')
            
            axis = next(axes)
            plot_clusters(d2_m_av, labels_d2, axis, lons, lats, dot_size=75,
                region=map_region)
            axis.set_title(f'Decade 2, k = {k}, sil score {round(sil_d2, 4)}')
            show_or_save(figure, f'{fig_name_base}_dma_k{k}.png')
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
    main()