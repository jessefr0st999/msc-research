from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pickle
import os
import time

import pandas as pd
import numpy as np
from geopy.distance import geodesic
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import folium
import folium.plugins
from selenium import webdriver

from helpers import *


## Constants

# TODO: Convert these to command line flags
DROP_PCT = 80
SAVE_PRECIPITATION = 1
SAVE_LINKS = 1
SAVE_METRICS = 0
SAVE_MAP_IMAGES = 0
CALCUALTE_METRICS = 0
PLOT_GRAPHS = 1
ANALYSED_DT_LIMIT = 1
# LINK_STR_GEO_PENALTY = 1/20000
LINK_STR_GEO_PENALTY = 0
LINK_STR_THRESHOLD = 2.8
EDGE_DENSITY = 0.005
# EDGE_DENSITY = None
AVG_LOOKBACK_MONTHS = 12
LAG_MONTHS = 6

DATA_DIR = 'data/precipitation'
OUTPUTS_DIR = 'data/outputs'
DATA_FILE = f'{DATA_DIR}/FusedData.csv'
LOCATIONS_FILE = f'{DATA_DIR}/Fused.Locations.csv'
PREC_DATA_FILE = f'{DATA_DIR}/dataframe_drop_{DROP_PCT}.pkl'
if EDGE_DENSITY:
    TIME_METRICS_DATA_FILE = f'{OUTPUTS_DIR}/time_metrics_drop_{DROP_PCT}_ed_' + \
        f'{str(EDGE_DENSITY).replace(".", "p")}.pkl'
    SPATIAL_METRICS_DATA_FILE = f'{OUTPUTS_DIR}/spatial_metrics_drop_{DROP_PCT}_ed_' + \
        f'{str(EDGE_DENSITY).replace(".", "p")}.pkl'
    SEASONAL_METRICS_DATA_FILE = f'{OUTPUTS_DIR}/seasonal_metrics_drop_{DROP_PCT}_ed_' + \
        f'{str(EDGE_DENSITY).replace(".", "p")}.pkl'
else:
    TIME_METRICS_DATA_FILE = f'{OUTPUTS_DIR}/time_metrics_drop_{DROP_PCT}_thr_' + \
        f'{str(LINK_STR_THRESHOLD).replace(".", "p")}.pkl'
    SPATIAL_METRICS_DATA_FILE = f'{OUTPUTS_DIR}/spatial_metrics_drop_{DROP_PCT}_thr_' + \
        f'{str(LINK_STR_THRESHOLD).replace(".", "p")}.pkl'
    SEASONAL_METRICS_DATA_FILE = f'{OUTPUTS_DIR}/seasonal_metrics_drop_{DROP_PCT}_thr_' + \
        f'{str(LINK_STR_THRESHOLD).replace(".", "p")}.pkl'

YEARS = range(2000, 2022)
MONTHS = range(1, 13)


## Helper functions

def build_link_str_df(df: pd.DataFrame, start_time=None):
    link_str_df = pd.DataFrame(index=[df['lat'], df['lon']], columns=[df['lat'], df['lon']])
    df = df.reset_index().set_index(['lat', 'lon'])
    for i, idx1 in enumerate(df.index):
        if start_time and i % 25 == 0:
            print(f'{i} / {len(df.index)} points correlated; time elapsed: {datetime.now() - start_time}')
        for j, idx2 in enumerate(df.index[i + 1:]):
            seq_1 = df.at[idx1, 'prec_seq']
            seq_2 = df.at[idx2, 'prec_seq']
            # Calculate covariance matrix from both unlagged and lagged sequences
            seq_1_list = [seq_1[LAG_MONTHS :]]
            seq_2_list = [seq_2[LAG_MONTHS :]]
            for k in range(-1, -1 - LAG_MONTHS, -1):
                seq_1_offset = seq_1[k + LAG_MONTHS : k]
                seq_2_offset = seq_2[k + LAG_MONTHS : k]
                seq_1_list.append(seq_1_offset)
                seq_2_list.append(seq_2_offset)
            # seq_list contains [seq_1_L0, seq_1_L1, ..., seq_2_L0, seq_2_L1, ...]
            seq_list = [*seq_1_list, *seq_2_list]
            # Covariance matrix will have 2 * (LAG_MONTHS + 1) rows and columns
            cov_mat = np.corrcoef(seq_list)
            # Extract coefficients corresponding to
            # (seq_1_L0, seq_2_L0), (seq_1_L0, seq_2_L1), (seq_1_L0, seq_2_L2), ...
            # and (seq_2_L0, seq_1_L1), (seq_2_L0, seq_1_L2), ...
            corrs = np.abs([*cov_mat[LAG_MONTHS + 1 :, 0],
                *cov_mat[LAG_MONTHS + 1, 1 : LAG_MONTHS + 1]])
            # Calculate link strength from correlations as documented
            link_str = (np.max(corrs) - np.mean(corrs)) / np.std(corrs)
            geodesic_km = geodesic(idx1, idx2).km
            link_str_df.at[idx1, idx2] = link_str - LINK_STR_GEO_PENALTY * geodesic_km
    # link_str_df is upper triangular
    link_str_df = link_str_df.add(link_str_df.T, fill_value=0).fillna(0)
    return link_str_df

def create_graph(adjacency: pd.DataFrame):
    graph = nx.from_numpy_matrix(adjacency.values)
    graph = nx.relabel_nodes(graph, dict(enumerate(adjacency.columns)))
    return graph

def plot_graph(graph: nx.Graph, adjacency: pd.DataFrame, lons, lats):
    if graph is None:
        graph = create_graph(adjacency)
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
    map_x, map_y = _map(lons, lats)
    pos = {}
    for i, elem in enumerate(adjacency.index):
        pos[elem] = (map_x[i], map_y[i])
    nx.draw_networkx_nodes(G=graph, pos=pos, nodelist=graph.nodes(),
        node_color='r', alpha=0.8,
        node_size=[adjacency[location].sum() for location in graph.nodes()])
    nx.draw_networkx_edges(G=graph, pos=pos, edge_color='g',
        alpha=0.2, arrows=False)
    _map.drawcountries(linewidth=3)
    _map.drawstates(linewidth=0.2)
    _map.drawcoastlines(linewidth=3)
    plt.tight_layout()
    plt.savefig('./map_1.png', format='png', dpi=300)
    plt.show()

def calculate_network_metrics(graph):
    shortest_paths, eccentricities = shortest_path_and_eccentricity(graph)
    return {
        'average_degree': average_degree(graph),
        'transitivity': transitivity(graph),
        'eigenvector_centrality': eigenvector_centrality(graph),
        'coreness': coreness(graph),
        'average_degree': average_degree(graph),
        'global_average_link_distance': global_average_link_distance(graph),
        'modularity': modularity(graph),
        'shortest_path': np.average(list(shortest_paths.values())),
        'eccentricity': np.average(list(eccentricities.values())),
    }

def prepare_prec_df(input_df):
    input_df.columns = pd.to_datetime(input_df.columns, format='D%Y.%m')
    df_locations = pd.read_csv(LOCATIONS_FILE)
    df = pd.concat([df_locations, input_df], axis=1)

    # Remove a random subset of locations from dataframe for quicker testing
    if DROP_PCT > 0:
        np.random.seed(10)
        _floor = lambda n: int(n // 1)
        drop_indices = np.random.choice(df.index, _floor(0.01 * DROP_PCT * len(df)), replace=False)
        df = df.drop(drop_indices)

    df = df.set_index(['Lat', 'Lon'])
    def seq_func(row: pd.Series):
        def func(start_date, end_date, r: pd.Series):
            sequence = r[(r.index > start_date) & (r.index <= end_date)]
            return list(sequence) if len(sequence) == AVG_LOOKBACK_MONTHS + LAG_MONTHS else None
        vec_func = np.vectorize(func, excluded=['r'])
        start_dates = [d - relativedelta(months=AVG_LOOKBACK_MONTHS + LAG_MONTHS) for d in row.index]
        end_dates = [d for d in row.index]
        _row = vec_func(start_date=start_dates, end_date=end_dates, r=row)
        return pd.Series(_row, index=row.index)
    start = datetime.now()
    print('Constructing sequences...')
    df = df.apply(seq_func, axis=1)
    print(f'Sequences constructed; time elapsed: {datetime.now() - start}')
    df = df.stack().reset_index()
    df = df.rename(columns={'level_2': 'date', 0: 'prec_seq', 'Lat': 'lat', 'Lon': 'lon'})
    df = df.set_index('date')
    return df


## Workflow

def main():
    if Path(PREC_DATA_FILE).is_file():
        print(f'Reading precipitation data from pickle file {PREC_DATA_FILE}')
        df: pd.DataFrame = pd.read_pickle(PREC_DATA_FILE)
    else:
        print('Reading precipitation data from raw files')
        df = prepare_prec_df(pd.read_csv(DATA_FILE))
        if SAVE_PRECIPITATION:
            print(f'Saving precipitation data to pickle file {PREC_DATA_FILE}')
            df.to_pickle(PREC_DATA_FILE)

    graphs = []
    time_metrics_list = []
    graph_times = []
    analysed_dt_count = 0
    for y in YEARS:
        for m in MONTHS:
            if ANALYSED_DT_LIMIT and analysed_dt_count == ANALYSED_DT_LIMIT:
                break
            dt = datetime(y, m, 1)
            try:
                location_df = df.loc[dt]
                # Skip unless the sequence based on the specified lookback time is available
                if location_df['prec_seq'].isnull().values.any():
                    location_df = None
            except KeyError:
                location_df = None
            if location_df is not None:
                analysed_dt_count += 1
                date_summary = f'{dt.year}, {dt.strftime("%b")}'
                month_str = str(dt.month) if dt.month >= 10 else f'0{dt.month}'
                links_file = f'{DATA_DIR}/link_str_drop_{DROP_PCT}_{dt.year}_{month_str}'
                if LINK_STR_GEO_PENALTY:
                    links_file += f'_geo_pen_{str(int(1 / LINK_STR_GEO_PENALTY))}'
                links_file += '.pkl'
                if Path(links_file).is_file():
                    print(f'{date_summary}: reading link strength data from pickle file {links_file}')
                    link_str_df: pd.DataFrame = pd.read_pickle(links_file)
                else:
                    print(f'\n{date_summary}: calculating link strength data...')
                    start = datetime.now()
                    link_str_df = build_link_str_df(location_df, start)
                    print(f'{date_summary}: correlations and link strengths calculated; time elapsed: {datetime.now() - start}')
                    if SAVE_LINKS:
                        print(f'{date_summary}: saving link strength data to pickle file {links_file}')
                        link_str_df.to_pickle(links_file)

                adjacency = pd.DataFrame(0, columns=link_str_df.columns, index=link_str_df.index)
                if EDGE_DENSITY:
                    threshold = np.quantile(link_str_df, 1 - EDGE_DENSITY)
                else:
                    threshold = LINK_STR_THRESHOLD
                adjacency[link_str_df >= threshold] = 1
                graph = create_graph(adjacency)
                graphs.append(graph)
                graph_times.append(dt)
                if PLOT_GRAPHS:
                    plot_graph(graph, adjacency, location_df['lon'], location_df['lat'])
                if not CALCUALTE_METRICS:
                    continue
                if Path(TIME_METRICS_DATA_FILE).is_file():
                    continue
                print(f'{date_summary}: calculating graph metrics...')
                start = datetime.now()
                graph_metrics = calculate_network_metrics(graph)
                print(f'{date_summary}: graph metrics calculated; time elapsed: {datetime.now() - start}')
                time_metrics_list.append({
                    'dt': dt,
                    'graph_metrics': graph_metrics,
                    'link_metrics': {
                        'average_link_strength': np.average(link_str_df),
                    },
                })

    if not CALCUALTE_METRICS:
        return

    # Output for time metrics
    if SAVE_METRICS:
        print(f'Saving time series metrics to pickle file {TIME_METRICS_DATA_FILE}')
        with open(TIME_METRICS_DATA_FILE, 'wb') as f:
            pickle.dump(time_metrics_list, f)

    # Perform spatial analysis by averaging across timesteps
    lats = []
    lons = []
    spatial_metrics = ['eccentricity', 'average_shortest_path', 'degree',
        'degree_centrality', 'eigenvector_centrality', 'clustering']
    spatial_metrics_dict = {}
    seasons = ['summer', 'autumn', 'winter', 'spring']
    seasonal_metrics_dict = {season: {} for season in seasons}
    for node in graphs[0]:
        spatial_metrics_dict[node] = {m: [] for m in spatial_metrics}
        for season in seasons:
            seasonal_metrics_dict[season][node] = {m: [] for m in spatial_metrics}
        lats.append(node[0])
        lons.append(node[1])

    # Input for spatial metrics
    if Path(SPATIAL_METRICS_DATA_FILE).is_file() and Path(SEASONAL_METRICS_DATA_FILE).is_file():
        print('Reading spatial metrics from pickle files '
            f'{SPATIAL_METRICS_DATA_FILE} and {SEASONAL_METRICS_DATA_FILE}')
        spatial_metrics_df = pd.read_pickle(SPATIAL_METRICS_DATA_FILE)
        with open(SEASONAL_METRICS_DATA_FILE, 'rb') as f:
            seasonal_metrics_dict = pickle.load(f)
    else:
        start = datetime.now()
        spatial_metrics_array_full = []
        for g, dt in zip(graphs, graph_times):
            # Type hints
            g: nx.Graph = g
            dt: datetime = dt
            date_summary = f'{dt.year}, {dt.strftime("%b")}'
            print(f'{date_summary}: calculating spatial metrics...')
            if dt.month in [12, 1, 2]:
                season = 'summer'
            elif dt.month in [3, 4, 5]:
                season = 'autumn'
            elif dt.month in [6, 7, 8]:
                season = 'winter'
            else:
                season = 'spring'
            average_shortest_path, eccentricity = shortest_path_and_eccentricity(graph)
            clustering = nx.clustering(g)
            try:
                eigenvector_centrality = nx.eigenvector_centrality(g)
            except nx.exception.PowerIterationFailedConvergence:
                eigenvector_centrality = {k: np.nan for k in g.nodes}
            degree_centrality = nx.degree_centrality(g)
            def set_metrics(_dict):
                _dict['eccentricity'].append(eccentricity[node])
                _dict['average_shortest_path'].append(average_shortest_path[node])
                _dict['degree'].append(g.degree[node])
                _dict['degree_centrality'].append(degree_centrality[node])
                _dict['eigenvector_centrality'].append(eigenvector_centrality[node])
                _dict['clustering'].append(clustering[node])
                return _dict
            spatial_metrics_array = []
            # TODO: rework seasonal output to be like summary output
            for node in graphs[0]:
                spatial_metrics_dict[node] = set_metrics(spatial_metrics_dict[node])
                seasonal_metrics_dict[season][node] = set_metrics(seasonal_metrics_dict[season][node])
                spatial_metrics_array.extend([
                    eccentricity[node],
                    average_shortest_path[node],
                    g.degree[node],
                    degree_centrality[node],
                    eigenvector_centrality[node],
                    clustering[node],
                ])
            spatial_metrics_array_full.append(spatial_metrics_array)
            index = pd.MultiIndex.from_product([list(graphs[0]), spatial_metrics])
        spatial_metrics_df = pd.DataFrame(spatial_metrics_array_full, index=graph_times, columns=index)

        print(f'Spatial metrics calculated; time elapsed: {datetime.now() - start}')

    # Output for spatial metrics
    if SAVE_METRICS and not Path(SPATIAL_METRICS_DATA_FILE).is_file():
        print(f'Saving spatial metrics to pickle file {SPATIAL_METRICS_DATA_FILE}')
        spatial_metrics_df.to_pickle(SPATIAL_METRICS_DATA_FILE)
    if SAVE_METRICS and not Path(SEASONAL_METRICS_DATA_FILE).is_file():
        print(f'Saving seasonal spatial metrics to pickle file {SEASONAL_METRICS_DATA_FILE}')
        with open(SEASONAL_METRICS_DATA_FILE, 'wb') as f:
            pickle.dump(seasonal_metrics_dict, f)

    # Construct HTML files with spatial metric plots
    for m in spatial_metrics:
        # Lower than average latitude to include Tasmania
        map = folium.Map(location=[np.average(lats) - 3, np.average(lons)], zoom_start=5)
        values = [np.average(v[m]) for v in spatial_metrics_dict.values()]
        heatmap = folium.plugins.HeatMap(
            list(zip(lats, lons, values)),
            min_opacity=0.3,
            radius=30,
            blur=30,
            max_zoom=1,
        )
        map_file = f'{OUTPUTS_DIR}/{m}_drop_{DROP_PCT}.html'
        heatmap.add_to(map)
        map.save(map_file)
        print(f'Map file {map_file} saved!')
        if SAVE_MAP_IMAGES:
            image_file = f'{OUTPUTS_DIR}/{m}_drop_{DROP_PCT}.png'
            with webdriver.Firefox() as driver:
                driver.get(f'{os.getcwd()}\\{map_file}')
                time.sleep(1)
                driver.save_screenshot(image_file)
            print(f'Image file {image_file} saved!')

if __name__ == '__main__':
    start = datetime.now()
    main()
    print(f'Total time elapsed: {datetime.now() - start}')