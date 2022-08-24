import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.basemap import Basemap

OUTPUTS_DIR = 'data/outputs'
# DATA_FILE = f'{OUTPUTS_DIR}/spatial_metrics_drop_90_thr_2p8.pkl'
DATA_FILE = f'{OUTPUTS_DIR}/spatial_metrics_drop_50_thr_2p8.pkl'
grid_size = (50, 50)

spatial_metrics = ['eccentricity', 'average_shortest_path', 'degree',
    'degree_centrality', 'eigenvector_centrality', 'clustering']
metrics_dict = pd.read_pickle(DATA_FILE)
df = pd.DataFrame.from_dict(metrics_dict).T.reset_index()
for m in spatial_metrics:
    df[m] = df[m].fillna(0).apply(lambda series: np.average(series))
df = df.rename(columns={'level_0': 'lat', 'level_1': 'lon'})
# pivot = df.pivot(index='lat', columns='lon', values='degree')
# pivot = pivot.replace(np.nan, 0)

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

figure = plt.figure()
axis = figure.add_subplot(2, 3, 1)
axis.set_title('Average degree')
degree_map = get_map()

def transform_series(series: pd.Series, new_max):
    series = series - series.min()
    series = series / series.max() * new_max
    return series
lon_span = np.linspace(0, degree_map.urcrnrx, grid_size[0])
lat_span = np.linspace(0, degree_map.urcrnry, grid_size[1])
X, Y = np.meshgrid(lon_span, lat_span)
triangles = tri.Triangulation(
    transform_series(df['lon'], degree_map.urcrnrx),
    transform_series(df['lat'], degree_map.urcrnry),
)

interpolator = tri.LinearTriInterpolator(triangles, df['degree'])
degree = interpolator(X, Y)
interpolator = tri.LinearTriInterpolator(triangles, df['clustering'])
clustering = interpolator(X, Y)
interpolator = tri.LinearTriInterpolator(triangles, df['eigenvector_centrality'])
eigenvector_centrality = interpolator(X, Y)
interpolator = tri.LinearTriInterpolator(triangles, df['average_shortest_path'])
average_shortest_path = interpolator(X, Y)
interpolator = tri.LinearTriInterpolator(triangles, df['eccentricity'])
eccentricity = interpolator(X, Y)
interpolator = tri.LinearTriInterpolator(triangles, df['degree_centrality'])
degree_centrality = interpolator(X, Y)

degree_map.contourf(X, Y, degree)
plt.colorbar(ax=axis)

axis = figure.add_subplot(2, 3, 2)
axis.set_title('Clustering')
clustering_map = get_map()
clustering_map.contourf(X, Y, clustering)
plt.colorbar(ax=axis)

axis = figure.add_subplot(2, 3, 3)
axis.set_title('Eigenvector centrality')
ec_map = get_map()
ec_map.contourf(X, Y, eigenvector_centrality)
plt.colorbar(ax=axis)

axis = figure.add_subplot(2, 3, 4)
axis.set_title('Average shortest path')
asp_map = get_map()
asp_map.contourf(X, Y, average_shortest_path)
plt.colorbar(ax=axis)

axis = figure.add_subplot(2, 3, 5)
axis.set_title('Eccentricity')
ec_map = get_map()
ec_map.contourf(X, Y, eccentricity)
plt.colorbar(ax=axis)

axis = figure.add_subplot(2, 3, 6)
axis.set_title('Degree centrality')
asp_map = get_map()
asp_map.contourf(X, Y, degree_centrality)
plt.colorbar(ax=axis)

plt.tight_layout()
plt.show()
