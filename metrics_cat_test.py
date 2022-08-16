import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.basemap import Basemap
from sklearn.cluster import KMeans

OUTPUTS_DIR = 'data/outputs'
# DATA_FILE = f'{OUTPUTS_DIR}/spatial_metrics_drop_90_thr_2p8.pkl'
DATA_FILE = f'{OUTPUTS_DIR}/spatial_metrics_drop_50_thr_2p8.pkl'
# DATA_FILE = f'{OUTPUTS_DIR}/spatial_metrics_drop_0_thr_2p8.pkl' # currently only 1 time step
grid_size = (50, 50)
spatial_metrics = ['eccentricity', 'average_shortest_path', 'degree',
    'degree_centrality', 'eigenvector_centrality', 'clustering']
n_categories = 6

colours = ['grey', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', 'white', '#bcbd22', '#17becf']
cmap = colors.ListedColormap([colours[i] for i in range(n_categories + 1)])
bounds = [x - 0.5 for x in range(n_categories + 2)]
norm = colors.BoundaryNorm(bounds, cmap.N)

lon_start = 110
lon_end = 155
lat_start = -45
lat_end = -10

def main():
    figure, axis = plt.subplots()
    metrics_dict = pd.read_pickle(DATA_FILE)
    df = pd.DataFrame.from_dict(metrics_dict).T.reset_index()
    for m in spatial_metrics:
        df[m] = df[m].fillna(0).apply(lambda series: np.average(series))
        df[m] = df[m] / df[m].max()
    df = df.rename(columns={'level_0': 'lat', 'level_1': 'lon'})
    df = df.set_index(['lat', 'lon'])
    df = df[spatial_metrics]

    train_data = np.array(df)
    # train_data = np.array(df['clustering']).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_categories, random_state=0).fit(train_data)
    df['category'] = [x + 1 for x in kmeans.labels_]

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

    lons = np.linspace(lon_start + 0.5, lon_end - 0.5, int(lon_end - lon_start))
    lats = np.linspace(lat_start + 0.25, lat_end - 0.25, int((lat_end - lat_start) / 0.5))
    _map = get_map()
    map_grid_data = pd.DataFrame(pd.DataFrame(0, columns=lons, index=lats).stack())
    map_grid_data = map_grid_data.reset_index().rename(
        columns={'level_0': 'lat', 'level_1': 'lon'})
    df = df.reset_index().rename(columns={'level_0': 'lat', 'level_1': 'lon'})
    map_df = pd.merge(map_grid_data, df, on=['lat', 'lon'], how='left')
    map_df = map_df.reset_index().pivot(index='lat', columns='lon', values='category')
    map_df = map_df.replace(np.nan, 0)

    img = axis.imshow(map_df, cmap=cmap, norm=norm, origin='lower',
        extent=(0, _map.urcrnrx, 0, _map.urcrnry))
    plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds,
        ticks=list(range(n_categories + 1)))
    plt.show()

main()