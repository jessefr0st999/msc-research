import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.basemap import Basemap
from sklearn.cluster import KMeans

OUTPUTS_DIR = 'data/outputs'
# DATA_FILE = f'{OUTPUTS_DIR}/spatial_metrics_drop_90_thr_2p8.pkl'
DATA_FILE = f'{OUTPUTS_DIR}/spatial_metrics_drop_50_thr_2p8.pkl'
grid_size = (50, 50)
spatial_metrics = ['eccentricity', 'average_shortest_path', 'degree',
    'degree_centrality', 'eigenvector_centrality', 'clustering']
n_categories = 5

colours = ['grey', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', 'white', '#bcbd22', '#17becf']
cmap = colors.ListedColormap([colours[i] for i in range(n_categories + 1)])
bounds = [x - 0.5 for x in range(n_categories + 2)]
norm = colors.BoundaryNorm(bounds, cmap.N)

def main():
    metrics_dict = pd.read_pickle(DATA_FILE)
    df = pd.DataFrame.from_dict(metrics_dict).T.reset_index()
    for m in spatial_metrics:
        df[m] = df[m].fillna(0).apply(lambda series: np.average(series))
    df = df.rename(columns={'level_0': 'lat', 'level_1': 'lon'})
    df = df.set_index(['lat', 'lon'])
    df = df / df.max()

    train_data = np.array(df)
    # train_data = np.array(df['clustering']).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_categories, random_state=0).fit(train_data)
    df['category'] = [x + 1 for x in kmeans.labels_]
    grid_data = df.reset_index().pivot(index='lat', columns='lon', values='category')
    grid_data = grid_data.replace(np.nan, 0)

    def get_map():
        _map = Basemap(
            projection='merc',
            llcrnrlon=110,
            llcrnrlat=-45,
            urcrnrlon=155,
            urcrnrlat=-10,
            lat_ts=0,
            resolution='l',
            suppress_ticks=True,
        )
        _map.drawcountries(linewidth=3)
        _map.drawstates(linewidth=0.2)
        _map.drawcoastlines(linewidth=3)
        return _map

    # TODO: scale this onto a map with a grid
    # _map = get_map()
    # lon_span = np.linspace(0, _map.urcrnrx, grid_size[0])
    # lat_span = np.linspace(0, _map.urcrnry, grid_size[1])
    figure, axis = plt.subplots()
    img = axis.imshow(grid_data, cmap=cmap, norm=norm, origin='lower')
    plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds,
        ticks=list(range(n_categories + 1)))
    plt.show()

main()