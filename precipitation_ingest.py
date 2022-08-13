from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pickle
import os
import time

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import folium
import folium.plugins
from selenium import webdriver

from helpers import *


## Constants

# TODO: Convert these to command line flags
DROP_PCT = 90
SAVE_LINKS = 1
SAVE_CORRS = 0
SAVE_PRECIPITATION = 1
SAVE_METRICS = 1
SAVE_MAP_IMAGES = 0
CREATE_GRAPHS = 1
PLOT_GRAPHS = 0
ANALYSED_DT_LIMIT = 0
LINK_STR_THRESHOLD = 2.8
AVG_LOOKBACK_MONTHS = 12
LAG_MONTHS = 6

DATA_DIR = 'data/precipitation'
OUTPUTS_DIR = 'data/outputs'
DATA_FILE = f'{DATA_DIR}/FusedData.csv'
LOCATIONS_FILE = f'{DATA_DIR}/Fused.Locations.csv'
PREC_DATA_FILE = f'{DATA_DIR}/dataframe_drop_{DROP_PCT}.pkl'
TIME_METRICS_DATA_FILE = f'{OUTPUTS_DIR}/time_metrics_drop_{DROP_PCT}_thr_' + \
    f'{str(LINK_STR_THRESHOLD).replace(".", "p")}.pkl'
SPATIAL_METRICS_DATA_FILE = f'{OUTPUTS_DIR}/spatial_metrics_drop_{DROP_PCT}_thr_' + \
    f'{str(LINK_STR_THRESHOLD).replace(".", "p")}.pkl'

YEARS = range(2000, 2022)
MONTHS = range(1, 13)


## Helper functions

def build_link_str_df(df: pd.DataFrame, start_time=None):
    link_str_df = pd.DataFrame(index=[df['lat'], df['lon']], columns=[df['lat'], df['lon']])
    corrs_df = pd.DataFrame(index=[df['lat'], df['lon']], columns=[df['lat'], df['lon']])
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
            for i in range(-1, -1 - LAG_MONTHS, -1):
                seq_1_offset = seq_1[i + LAG_MONTHS : i]
                seq_2_offset = seq_2[i + LAG_MONTHS : i]
                seq_1_list.append(seq_1_offset)
                seq_2_list.append(seq_2_offset)
            # seq_list contains [seq_1_L0, seq_1_L1, ..., seq_2_L0, seq_2_L1, ...]
            seq_list = [*seq_1_list, *seq_2_list]
            # Covariance matrix will have 2 * (LAG_MONTHS + 1) rows and columns
            cov_mat = np.corrcoef(seq_list)
            # Extract coefficients corresponding to
            # (seq_1_L0, seq_2_L0), (seq_1_L0, seq_2_L1), (seq_1_L0, seq_2_L2), ...
            # and (seq_2_L0, seq_1_L1), (seq_2_L0, seq_1_L2), ...
            corrs = np.abs([*cov_mat[LAG_MONTHS + 1 :, 0], *cov_mat[LAG_MONTHS + 1, 1 : LAG_MONTHS + 1]])
            if SAVE_CORRS:
                corrs_df.at[idx1, idx2] = corrs
            # Calculate link strength from correlations as documented
            link_str = (np.max(corrs) - np.mean(corrs)) / np.std(corrs)
            link_str_df.at[idx1, idx2] = link_str
    # link_str_df is upper triangular
    link_str_df = link_str_df.add(link_str_df.T, fill_value=0).fillna(0)
    return link_str_df, corrs_df

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
    mx, my = _map(lons, lats)
    pos = {}
    for i, elem in enumerate(adjacency.index):
        pos[elem] = (mx[i], my[i])
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
                links_file = f'{DATA_DIR}/link_str_drop_{DROP_PCT}_{dt.year}_{month_str}.pkl'
                corrs_file = f'{DATA_DIR}/corrs_drop_{DROP_PCT}_{dt.year}_{month_str}.pkl'
                # TODO: Allow reading in correlations file
                if Path(links_file).is_file():
                    print(f'{date_summary}: reading link strength data from pickle file {links_file}')
                    link_str_df: pd.DataFrame = pd.read_pickle(links_file)
                else:
                    print(f'\n{date_summary}: calculating link strength data...')
                    start = datetime.now()
                    link_str_df, corrs_df = build_link_str_df(location_df, start)
                    print(f'{date_summary}: correlations and link strengths calculated; time elapsed: {datetime.now() - start}')
                    if SAVE_LINKS:
                        print(f'{date_summary}: saving link strength data to pickle file {links_file}')
                        link_str_df.to_pickle(links_file)
                    if SAVE_CORRS:
                        print(f'{date_summary}: saving correlation data to pickle file {corrs_file}')
                        corrs_df.to_pickle(corrs_file)

                if not CREATE_GRAPHS:
                    continue
                adjacency = pd.DataFrame(0, columns=link_str_df.columns, index=link_str_df.index)
                adjacency[link_str_df >= LINK_STR_THRESHOLD] = 1
                graph = create_graph(adjacency)
                graphs.append(graph)
                if PLOT_GRAPHS:
                    plot_graph(graph, adjacency, location_df['lon'], location_df['lat'])
                graph_times.append(dt)
                if Path(TIME_METRICS_DATA_FILE).is_file():
                    continue
                print(f'{date_summary}: calculating graph metrics...')
                start = datetime.now()
                graph_metrics = calculate_network_metrics(graph)
                print(f'{date_summary}: graph metrics calculated; time elapsed: {datetime.now() - start}')
                time_metrics_list.append({
                    'graph_metrics': graph_metrics,
                    'link_metrics': {
                        'average_link_strength': np.average(link_str_df),
                    },
                })

    if not CREATE_GRAPHS:
        return

    # Input/output for time metrics
    if Path(TIME_METRICS_DATA_FILE).is_file():
        print(f'Reading time series metrics from pickle file {TIME_METRICS_DATA_FILE}')
        with open(TIME_METRICS_DATA_FILE, 'rb') as f:
            time_metrics_list = pickle.load(f)
    elif SAVE_METRICS:
        print(f'Saving time series metrics to pickle file {TIME_METRICS_DATA_FILE}')
        with open(TIME_METRICS_DATA_FILE, 'wb') as f:
            pickle.dump(time_metrics_list, f)

    # Perform spatial analysis by averaging across timesteps
    lats = []
    lons = []
    spatial_metrics = ['eccentricity', 'average_shortest_path', 'degree',
        'degree_centrality', 'eigenvector_centrality', 'clustering']
    spatial_metrics_dict = {}
    for node in graphs[0]:
        spatial_metrics_dict[node] = {m: [] for m in spatial_metrics}
        lats.append(node[0])
        lons.append(node[1])

    # Input for spatial metrics
    if Path(SPATIAL_METRICS_DATA_FILE).is_file():
        print(f'Reading spatial metrics from pickle file {SPATIAL_METRICS_DATA_FILE}')
        with open(SPATIAL_METRICS_DATA_FILE, 'rb') as f:
            spatial_metrics_dict = pickle.load(f)
    else:
        print('Calculating spatial metrics...')
        start = datetime.now()
        for g in graphs:
            g: nx.Graph = g # Type hint
            average_shortest_path, eccentricity = shortest_path_and_eccentricity(graph)
            clustering = nx.clustering(g)
            eigenvector_centrality = nx.eigenvector_centrality(g)
            degree_centrality = nx.degree_centrality(g)
            for node in spatial_metrics_dict:
                spatial_metrics_dict[node]['degree'].append(g.degree[node])
                spatial_metrics_dict[node]['average_shortest_path'].append(average_shortest_path[node])
                spatial_metrics_dict[node]['eccentricity'].append(eccentricity[node])
                spatial_metrics_dict[node]['clustering'].append(clustering[node])
                spatial_metrics_dict[node]['eigenvector_centrality'].append(eigenvector_centrality[node])
                spatial_metrics_dict[node]['degree_centrality'].append(degree_centrality[node])
        print(f'Spatial metrics calculated; time elapsed: {datetime.now() - start}')

    # Output for spatial metrics
    if SAVE_METRICS and not Path(SPATIAL_METRICS_DATA_FILE).is_file():
        print(f'Saving spatial metrics to pickle file {SPATIAL_METRICS_DATA_FILE}')
        with open(SPATIAL_METRICS_DATA_FILE, 'wb') as f:
            pickle.dump(spatial_metrics_dict, f)

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

    # Construct figures for time series metrics
    figure, axes = plt.subplots(4, 2)
    axes[0, 0].set_title('Average degree')
    axes[0, 0].plot(graph_times, [l['graph_metrics']['average_degree'] \
        for l in time_metrics_list], '-b')
    axes[1, 0].set_title('Coreness')
    axes[1, 0].plot(graph_times, [l['graph_metrics']['coreness'] \
        for l in time_metrics_list], '-g')
    axes[2, 0].set_title('Modularity')
    axes[2, 0].plot(graph_times, [l['graph_metrics']['modularity'] \
        for l in time_metrics_list], '-r')
    axes[3, 0].set_title('Transitivity')
    axes[3, 0].plot(graph_times, [l['graph_metrics']['transitivity'] \
        for l in time_metrics_list], '-m')
    axes[0, 1].set_title('Link strength')
    axes[0, 1].plot(graph_times, [l['link_metrics']['average_link_strength'] \
        for l in time_metrics_list], '-k')
    axes[1, 1].set_title('Eigenvector centrality')
    axes[1, 1].plot(graph_times, [l['graph_metrics']['eigenvector_centrality'] \
        for l in time_metrics_list], '-y')
    axes[2, 1].set_title('Shortest path')
    axes[2, 1].plot(graph_times, [l['graph_metrics']['shortest_path'] \
        for l in time_metrics_list], '-', color='tab:orange')
    axes[3, 1].set_title('Eccentricity')
    axes[3, 1].plot(graph_times, [l['graph_metrics']['eccentricity'] \
        for l in time_metrics_list], '-', color='tab:cyan')
    plt.savefig(f'{OUTPUTS_DIR}/graph_plots_drop_{DROP_PCT}.png')
    plt.show()

if __name__ == '__main__':
    start = datetime.now()
    main()
    print(f'Total time elapsed: {datetime.now() - start}')