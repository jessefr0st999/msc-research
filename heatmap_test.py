import pandas as pd
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.basemap import Basemap

OUTPUTS_DIR = 'data/outputs'
# DATA_FILE = f'{OUTPUTS_DIR}/spatial_metrics_drop_90_thr_2p8.pkl'
DATA_FILE = f'{OUTPUTS_DIR}/spatial_metrics_drop_50_ed_0p025.pkl'
GRID_SIZE = (50, 50)
ANIM_FPS = 1

# spatial_metrics = ['eccentricity', 'average_shortest_path', 'degree',
#     'degree_centrality', 'eigenvector_centrality', 'betweenness_centrality',
#     'closeness_centrality', 'clustering']
spatial_metrics = ['clustering', 'betweenness_centrality', 'eigenvector_centrality',
    'closeness_centrality', 'average_shortest_path', 'degree']

def prepare_df_series(df: pd.DataFrame):
    df_series = []
    for i in range(len(df)):
        df_row = df.iloc[i].unstack()
        df_row[['lat', 'lon']] = df_row.index.tolist()
        columns_to_select = ['lat', 'lon', *spatial_metrics]
        df_series.append(df_row[columns_to_select])
    return df_series

def main():
    full_df = pd.read_pickle(DATA_FILE)
    df_series = prepare_df_series(full_df)

    def get_map(axis=None):
        _map = Basemap(
            projection='merc',
            llcrnrlon=110,
            llcrnrlat=-45,
            urcrnrlon=155,
            urcrnrlat=-10,
            lat_ts=0,
            resolution='l',
            suppress_ticks=True,
            ax=axis,
        )
        _map.drawcountries(linewidth=3)
        _map.drawstates(linewidth=0.2)
        _map.drawcoastlines(linewidth=3)
        return _map

    def prepare_figure():
        figure, axes = plt.subplots(2, 3)
        axes = axes.flatten()
        axes[0].set_title('clustering')
        axes[1].set_title('betweenness_centrality')
        axes[2].set_title('closeness_centrality')
        axes[3].set_title('eigenvector_centrality')
        axes[4].set_title('average_shortest_path')
        axes[5].set_title('degree')
        return figure, axes

    figure, axes = prepare_figure()
    clustering_map = get_map(axes[0])

    def transform_series(series: pd.Series, new_max):
        series = series - series.min()
        series = series / series.max() * new_max
        return series
    lon_span = np.linspace(0, clustering_map.urcrnrx, GRID_SIZE[0])
    lat_span = np.linspace(0, clustering_map.urcrnry, GRID_SIZE[1])
    X, Y = np.meshgrid(lon_span, lat_span)
    triangles = tri.Triangulation(
        transform_series(df_series[0]['lon'], clustering_map.urcrnrx),
        transform_series(df_series[0]['lat'], clustering_map.urcrnry),
    )

    for i, df in enumerate(df_series):
    # def animate_func(i):
        df = df_series[i]
        interpolator = tri.LinearTriInterpolator(triangles, df['clustering'])
        clustering = interpolator(X, Y)
        interpolator = tri.LinearTriInterpolator(triangles, df['betweenness_centrality'])
        betweenness_centrality = interpolator(X, Y)
        interpolator = tri.LinearTriInterpolator(triangles, df['closeness_centrality'])
        closeness_centrality = interpolator(X, Y)
        interpolator = tri.LinearTriInterpolator(triangles, df['eigenvector_centrality'])
        eigenvector_centrality = interpolator(X, Y)
        interpolator = tri.LinearTriInterpolator(triangles, df['average_shortest_path'])
        average_shortest_path = interpolator(X, Y)
        interpolator = tri.LinearTriInterpolator(triangles, df['degree'])
        degree = interpolator(X, Y)

        clustering_map = get_map(axes[0])
        bc_map = get_map(axes[1])
        cc_map = get_map(axes[2])
        ec_map = get_map(axes[3])
        asp_map = get_map(axes[4])
        degree_map = get_map(axes[5])
        clustering_map.contourf(X, Y, clustering)
        bc_map.contourf(X, Y, betweenness_centrality)
        cc_map.contourf(X, Y, closeness_centrality)
        ec_map.contourf(X, Y, eigenvector_centrality)
        asp_map.contourf(X, Y, average_shortest_path)
        degree_map.contourf(X, Y, degree)

        # for axis in axes:
        #     plt.colorbar(ax=axis)

        dt = full_df.index[i]
        print(f'{dt.strftime("%B")} {dt.year}')
        plt.tight_layout()
        figure.set_size_inches(32, 18)
        filename = f'images/metric_heatmap_{dt.strftime("%B")}_{dt.year}.png'
        plt.savefig(filename, bbox_inches='tight')
        print(f'Saved to file {filename}')
        # plt.show()
        figure, axes = prepare_figure()

    # anim_secs = len(df_series)
    # anim = FuncAnimation(
    #     plt.gcf(),
    #     animate_func,
    #     frames = anim_secs * ANIM_FPS,
    #     interval = 1000 / ANIM_FPS,
    #     blit=True,
    # )
    # anim.save('test_anim.mp4', fps=anim_fps)
    # plt.tight_layout()
    # plt.show()

main()