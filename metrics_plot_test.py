import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

OUTPUTS_DIR = 'data/outputs'
# DATA_FILE = f'{OUTPUTS_DIR}/spatial_metrics_drop_90_thr_2p8.pkl'
DATA_FILE = f'{OUTPUTS_DIR}/spatial_metrics_drop_50_ed_0p025.pkl'
GRID_SIZE = (50, 50)
ANIM_FPS = 1
SAVE_FIG = 0

# spatial_metrics = ['eccentricity', 'average_shortest_path', 'degree',
#     'degree_centrality', 'eigenvector_centrality', 'betweenness_centrality',
#     'closeness_centrality', 'clustering']
spatial_metrics = ['clustering', 'betweenness_centrality', 'closeness_centrality',
    'eigenvector_centrality', 'average_shortest_path', 'degree']

def prepare_df_series(df: pd.DataFrame):
    df_series = []
    for i in range(len(df)):
        df_row = df.iloc[i].unstack()
        df_row[['lat', 'lon']] = df_row.index.tolist()
        for m in spatial_metrics:
            df_row[f'norm_{m}'] = df_row[m] / df_row[m].max()
        columns_to_select = ['lat', 'lon', *spatial_metrics,
            *[f'norm_{m}' for m in spatial_metrics]]
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
    mx, my = clustering_map(df_series[0]['lon'], df_series[0]['lat'])

    for i, df in enumerate(df_series):
        df = df_series[i]
        size_func = lambda series: [150 * n**1 for n in series]
        cmap = 'RdYlGn_r'
        for j, m in enumerate(spatial_metrics):
            _ = get_map(axes[j])
            axes[j].scatter(mx, my, c=df[f'norm_{m}'], cmap=cmap,
                s=size_func(df[f'norm_{m}']))
            if not SAVE_FIG:
                norm = mpl.colors.Normalize(vmin=df[m].min(), vmax=df[m].max())
                plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[j])

        dt = full_df.index[i]
        print(f'{dt.strftime("%B")} {dt.year}')
        plt.tight_layout()
        figure.set_size_inches(32, 18)
        if SAVE_FIG:
            filename = f'images/metric_plot_{dt.strftime("%B")}_{dt.year}.png'
            plt.savefig(filename, bbox_inches='tight')
            print(f'Saved to file {filename}')
        else:
            plt.show()
        figure, axes = prepare_figure()

main()