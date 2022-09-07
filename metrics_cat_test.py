import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, animation
from helpers import get_map
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

OUTPUTS_DIR = 'data/outputs'
SUMMARY_DATA_FILE = f'{OUTPUTS_DIR}/spatial_metrics_drop_50_thr_2p8.pkl'
# SUMMARY_DATA_FILE = f'{OUTPUTS_DIR}/spatial_metrics_drop_90_ed_0p005.pkl'
# SUMMARY_DATA_FILE = f'{OUTPUTS_DIR}/spatial_metrics_drop_0_thr_2p8.pkl' # currently only 1 time step
SEASONAL_DATA_FILE = f'{OUTPUTS_DIR}/seasonal_metrics_drop_50_thr_2p8.pkl'
ANIM_FILE = f'{OUTPUTS_DIR}/anim_data.pkl'

FIXED_CATEGORIES = 5
# MAX_CATEGORIES = 11
MAX_CATEGORIES = None

SHOW_SUMMARY = 1
SHOW_SEASONS = 0
SHOW_ANIMATION = 0

spatial_metrics = ['eccentricity', 'average_shortest_path', 'degree',
    'degree_centrality', 'eigenvector_centrality', 'clustering']

colours = ['grey', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', 'white', '#bcbd22', '#17becf', 'black']
cmap = lambda num_categories: colors.ListedColormap([colours[i] for i in range(num_categories + 1)])
bounds = lambda num_categories: [x - 0.5 for x in range(num_categories + 2)]
seasons = ['summer', 'autumn', 'winter', 'spring']

lon_start = 110
lon_end = 155
lat_start = -45
lat_end = -10

def prepare_averaged_df(df: pd.DataFrame):
    df = df.mean().unstack()
    df.index.name = 'lat_lon'
    return df[spatial_metrics]

def prepare_df_series(df: pd.DataFrame):
    df_series = []
    for i in range(len(df)):
        df_row = df.iloc[i].unstack()
        df_row.index.name = 'lat_lon'
        df_series.append(df_row[spatial_metrics])
    return df_series

def kmeans_optimise(df: pd.DataFrame):
    train_data = np.array(df)
    sil_scores_dict = {}
    kmeans_dict = {}
    for n in range(2, MAX_CATEGORIES + 1):
        kmeans_dict[n] = KMeans(n_clusters=n, random_state=0).fit(train_data)
        sil_scores_dict[n] = silhouette_score(train_data, kmeans_dict[n].labels_)
        # Incentivise more categorisations
        # TODO: Investigate this more
        # sil_scores_dict[n] += np.sqrt(1 + n/75)
    from pprint import pprint
    pprint(sil_scores_dict)
    optimal_num_categories = max(sil_scores_dict, key=sil_scores_dict.get)
    return optimal_num_categories, kmeans_dict[optimal_num_categories]


def calculate_map_df(df: pd.DataFrame, kmeans):
    df['category'] = [x + 1 for x in kmeans.labels_]
    lons = np.linspace(lon_start + 0.5, lon_end - 0.5, int(lon_end - lon_start))
    lats = np.linspace(lat_start + 0.25, lat_end - 0.25, int((lat_end - lat_start) / 0.5))
    map_grid_data = pd.DataFrame(pd.DataFrame(0, columns=lons, index=lats).stack())
    map_grid_data = map_grid_data.reset_index().rename(
        columns={'level_0': 'lat', 'level_1': 'lon'})
    df = df.reset_index().rename(columns={'level_0': 'lat_lon'})
    df[['lat', 'lon']] = pd.DataFrame(df['lat_lon'].tolist())
    map_df = pd.merge(map_grid_data, df, on=['lat', 'lon'], how='left')
    map_df = map_df.reset_index().pivot(index='lat', columns='lon', values='category')
    map_df = map_df.replace(np.nan, 0)
    return map_df

def main():
    figure, axis = plt.subplots()
    df: pd.DataFrame = pd.read_pickle(SUMMARY_DATA_FILE)

    if SHOW_SUMMARY:
        averaged_df = prepare_averaged_df(df)
        if MAX_CATEGORIES:
            num_categories, kmeans = kmeans_optimise(averaged_df)
        else:
            num_categories = FIXED_CATEGORIES
            train_data = np.array(averaged_df)
            kmeans = KMeans(n_clusters=FIXED_CATEGORIES, random_state=0).fit(train_data)
        map_df = calculate_map_df(averaged_df, kmeans)
        summary_map = get_map(axis)
        _cmap = cmap(num_categories)
        _bounds = bounds(num_categories)
        norm = colors.BoundaryNorm(_bounds, _cmap.N)
        img = axis.imshow(map_df, cmap=_cmap, norm=norm, origin='lower',
            extent=(0, summary_map.urcrnrx, 0, summary_map.urcrnry))
        plt.colorbar(img, cmap=cmap, norm=norm, boundaries=_bounds,
            ticks=list(range(num_categories + 1)))
        plt.show()

    if SHOW_SEASONS:
        seasonal_metrics_dict = pd.read_pickle(SEASONAL_DATA_FILE)
        figure = plt.figure()
        for i, season in enumerate(seasons, start=1):
            axis = figure.add_subplot(2, 2, i)
            season_map = get_map(axis)
            df = pd.DataFrame.from_dict(seasonal_metrics_dict[season])
            averaged_df = df.T.reset_index()
            for m in spatial_metrics:
                averaged_df[m] = averaged_df[m].fillna(0).apply(lambda series: np.average(series))
                averaged_df[m] = averaged_df[m] / averaged_df[m].max()
            averaged_df['lat_lon'] = averaged_df.apply(lambda row: (row.level_0, row.level_1), axis=1)
            averaged_df = averaged_df.set_index(['lat_lon'])
            averaged_df = averaged_df[spatial_metrics]
            # TODO: use this instead once seasonal output has been updated
            # averaged_df = prepare_averaged_df(df)
            if MAX_CATEGORIES:
                num_categories, kmeans = kmeans_optimise(averaged_df)
            else:
                num_categories = FIXED_CATEGORIES
                train_data = np.array(averaged_df)
                kmeans = KMeans(n_clusters=FIXED_CATEGORIES, random_state=0).fit(train_data)
            map_df = calculate_map_df(averaged_df, kmeans)
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

    if SHOW_ANIMATION:
        anim_fps = 8
        anim_secs = 30
        summary_map = get_map(axis)

        # map_dfs = []
        # df_series = prepare_df_series(df)
        # for i, _df in enumerate(df_series):
        #     if i % 10 == 0:
        #         print(f'{i} / {len(df_series)}')
        #     train_data = np.array(_df)
        #     kmeans = KMeans(n_clusters=FIXED_CATEGORIES, random_state=0).fit(train_data)
        #     map_dfs.append(calculate_map_df(_df, kmeans))
        # with open(ANIM_FILE, 'wb') as f:
        #     pickle.dump(map_dfs, f)

        with open(ANIM_FILE, 'rb') as f:
            map_dfs = pickle.load(f)

        _cmap = cmap(FIXED_CATEGORIES)
        _bounds = bounds(FIXED_CATEGORIES)
        norm = colors.BoundaryNorm(_bounds, _cmap.N)
        img = axis.imshow(map_dfs[0], cmap=_cmap, norm=norm, origin='lower',
            extent=(0, summary_map.urcrnrx, 0, summary_map.urcrnry))

        # curr_pos = 0
        # def key_event(e):
        #     nonlocal curr_pos
        #     if e.key == 'right':
        #         curr_pos += 1
        #     else:
        #         curr_pos -= 1
        #     if curr_pos < 0:
        #         curr_pos = 0
        #     elif curr_pos > anim_fps * anim_secs - 1:
        #         curr_pos = anim_fps * anim_secs - 1
        #     print(curr_pos)
        #     axis.cla()
        #     img = axis.imshow(map_dfs[0], cmap=_cmap, norm=norm, origin='lower',
        #         extent=(0, summary_map.urcrnrx, 0, summary_map.urcrnry))
        #     img.set_array(map_dfs[curr_pos])
        #     dt = df.index[curr_pos]
        #     axis.set_title(f'{dt.strftime("%B")} {dt.year}')
        #     figure.canvas.draw()
        # figure.canvas.mpl_connect('key_press_event', key_event)

        def animate_func(i):
            img.set_array(map_dfs[i])
            dt = df.index[i]
            axis.set_title(f'{dt.strftime("%B")} {dt.year}')
            return [img]

        anim = animation.FuncAnimation(
            figure,
            animate_func,
            frames = anim_secs * anim_fps,
            interval = 1000 / anim_fps,
        )
        # anim.save('test_anim.mp4', fps=anim_fps)
        plt.show()

main()