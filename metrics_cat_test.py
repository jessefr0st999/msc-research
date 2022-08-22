import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.basemap import Basemap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

OUTPUTS_DIR = 'data/outputs'
# SUMMARY_DATA_FILE = f'{OUTPUTS_DIR}/spatial_metrics_drop_90_thr_2p8.pkl'
SUMMARY_DATA_FILE = f'{OUTPUTS_DIR}/spatial_metrics_drop_50_thr_2p8.pkl'
# SUMMARY_DATA_FILE = f'{OUTPUTS_DIR}/spatial_metrics_drop_0_thr_2p8.pkl' # currently only 1 time step
SEASONAL_DATA_FILE = f'{OUTPUTS_DIR}/seasonal_metrics_drop_50_thr_2p8.pkl'

grid_size = (50, 50)
spatial_metrics = ['eccentricity', 'average_shortest_path', 'degree',
    'degree_centrality', 'eigenvector_centrality', 'clustering']

colours = ['grey', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', 'white', '#bcbd22', '#17becf']
cmap = lambda num_categories: colors.ListedColormap([colours[i] for i in range(num_categories + 1)])
bounds = lambda num_categories: [x - 0.5 for x in range(num_categories + 2)]
seasons = ['summer', 'autumn', 'winter', 'spring']

lon_start = 110
lon_end = 155
lat_start = -45
lat_end = -10

def get_map():
    _map = Basemap(
        projection='merc',
        llcrnrlon=lon_start,
        llcrnrlat=lat_start,
        urcrnrlon=lon_end,
        urcrnrlat=lat_end,
        lat_ts=0,
        resolution='l',
        suppress_ticks=True,
    )
    _map.drawcountries(linewidth=3)
    _map.drawstates(linewidth=0.2)
    _map.drawcoastlines(linewidth=3)
    return _map

def calculate_map_df(df: pd.DataFrame):
    df = df.T.reset_index()
    for m in spatial_metrics:
        df[m] = df[m].fillna(0).apply(lambda series: np.average(series))
        df[m] = df[m] / df[m].max()
    df = df.rename(columns={'level_0': 'lat', 'level_1': 'lon'})
    df = df.set_index(['lat', 'lon'])
    df = df[spatial_metrics]
    train_data = np.array(df)

    sil_scores_dict = {}
    kmeans_dict = {}
    for n in range(2, 7 + 1):
        kmeans_dict[n] = KMeans(n_clusters=n, random_state=0).fit(train_data)
        sil_scores_dict[n] = silhouette_score(train_data, kmeans_dict[n].labels_)
        # Incentivise smaller categorisations
        # TODO: Investigate this more
        sil_scores_dict[n] += n/35
    optimal_num_categories = max(sil_scores_dict, key=sil_scores_dict.get)
    optimal_kmeans = kmeans_dict[optimal_num_categories]

    df['category'] = [x + 1 for x in optimal_kmeans.labels_]

    lons = np.linspace(lon_start + 0.5, lon_end - 0.5, int(lon_end - lon_start))
    lats = np.linspace(lat_start + 0.25, lat_end - 0.25, int((lat_end - lat_start) / 0.5))
    map_grid_data = pd.DataFrame(pd.DataFrame(0, columns=lons, index=lats).stack())
    map_grid_data = map_grid_data.reset_index().rename(
        columns={'level_0': 'lat', 'level_1': 'lon'})
    df = df.reset_index().rename(columns={'level_0': 'lat', 'level_1': 'lon'})
    map_df = pd.merge(map_grid_data, df, on=['lat', 'lon'], how='left')
    map_df = map_df.reset_index().pivot(index='lat', columns='lon', values='category')
    map_df = map_df.replace(np.nan, 0)
    return map_df, optimal_num_categories

def main():
    figure, axis = plt.subplots()
    summary_metrics_dict = pd.read_pickle(SUMMARY_DATA_FILE)
    df = pd.DataFrame.from_dict(summary_metrics_dict)
    map_df, num_categories = calculate_map_df(df)

    summary_map = get_map()
    _cmap = cmap(num_categories)
    _bounds = bounds(num_categories)
    norm = colors.BoundaryNorm(_bounds, _cmap.N)
    img = axis.imshow(map_df, cmap=_cmap, norm=norm, origin='lower',
        extent=(0, summary_map.urcrnrx, 0, summary_map.urcrnry))
    plt.colorbar(img, cmap=cmap, norm=norm, boundaries=_bounds,
        ticks=list(range(num_categories + 1)))
    plt.show()

    seasonal_metrics_dict = pd.read_pickle(SEASONAL_DATA_FILE)
    figure = plt.figure()
    for i, season in enumerate(seasons, start=1):
        axis = figure.add_subplot(2, 2, i)
        season_map = get_map()
        df = pd.DataFrame.from_dict(seasonal_metrics_dict[season])
        map_df, num_categories = calculate_map_df(df)
        _cmap = cmap(num_categories)
        _bounds = bounds(num_categories)
        norm = colors.BoundaryNorm(_bounds, _cmap.N)
        img = axis.imshow(map_df, cmap=_cmap, norm=norm, origin='lower',
            extent=(0, season_map.urcrnrx, 0, season_map.urcrnry))
        axis.set_ylabel(season)
        plt.colorbar(img, ax=axis, cmap=_cmap, norm=norm, boundaries=_bounds,
            ticks=list(range(num_categories + 1)))
    plt.tight_layout()
    plt.show()

main()