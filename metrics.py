from datetime import datetime
import argparse

import pandas as pd
import numpy as np
import networkx as nx

from helpers import read_link_str_df, link_str_to_adjacency

YEARS = list(range(2000, 2022 + 1))
MONTHS = list(range(1, 13))
DATA_DIR = 'data/precipitation'
DATA_FILE = f'{DATA_DIR}/FusedData.csv'
LOCATIONS_FILE = f'{DATA_DIR}/Fused.Locations.csv'

DECADE_END_DATES = [datetime(2011, 4, 1), datetime(2022, 3, 1)]
WHOLE_GRAPH_METRICS = [
    'asyn_lpa_partitions',
    'asyn_lpa_modularity',
    'average_betweenness_centrality',
    'average_closeness_centrality',
    'average_coreness',
    'average_degree',
    'average_eccentricity',
    'average_eigenvector_centrality',
    'average_link_strength',
    'average_shortest_path',
    'global_average_link_distance',
    'greedy_modularity_modularity',
    'greedy_modularity_partitions',
    'louvain_modularity',
    'louvain_partitions',
    'transitivity',
]

def average_degree(graph: nx.Graph):
    return 2 * len(graph.edges) / len(graph.nodes)

# TODO: Make this faster
def shortest_path_and_eccentricity(graph: nx.Graph):
    shortest_path_lengths = nx.shortest_path_length(graph)
    ecc_by_node = {}
    average_spl_by_node = {}
    for source_node, target_nodes in shortest_path_lengths:
        ecc_by_node[source_node] = np.max(list(target_nodes.values()))
        for point, spl in target_nodes.items():
            if spl == 0:
                target_nodes[point] = len(graph.nodes)
        average_spl_by_node[source_node] = np.average(list(target_nodes.values()))
    return average_spl_by_node, ecc_by_node

def partitions(graph: nx.Graph):
    lcc_graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
    return {
        'louvain': [p for p in \
             nx.algorithms.community.louvain_communities(lcc_graph)],
        'greedy_modularity': [p for p in \
            nx.algorithms.community.greedy_modularity_communities(lcc_graph)],
        'asyn_lpa': [p for p in \
            nx.algorithms.community.asyn_lpa_communities(lcc_graph, seed=0)],
        # This one causes the script to hang for an ~1000 edge graph
        # 'girvan_newman': [p for p in \
        #     nx.algorithms.community.girvan_newman(lcc_graph)],
    }

def modularity(graph: nx.Graph, communities):
    return nx.algorithms.community.modularity(graph, communities=communities)

# TODO: implement
def global_average_link_distance(graph: nx.Graph):
    return None

def create_graph(adjacency: pd.DataFrame):
    graph = nx.from_numpy_array(adjacency.values)
    graph = nx.relabel_nodes(graph, dict(enumerate(adjacency.columns)))
    return graph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', default='data')
    parser.add_argument('--edge_density', type=float, default=0.005)
    parser.add_argument('--link_str_threshold', type=float, default=None)
    parser.add_argument('--month', type=int, default=None)
    parser.add_argument('--data_dir', default='data/precipitation')
    parser.add_argument('--fmt', default='csv')
    parser.add_argument('--plot_world', action='store_true', default=False)
    parser.add_argument('--link_str_file_tag', default='corr_alm_60_lag_0')
    args = parser.parse_args()

    df = pd.read_csv(DATA_FILE)
    df.columns = pd.to_datetime(df.columns, format='D%Y.%m')
    locations_df = pd.read_csv(LOCATIONS_FILE).set_index(['Lat', 'Lon'])
    whole_graph_metrics_df = pd.DataFrame(np.nan, columns=WHOLE_GRAPH_METRICS,
        index=DECADE_END_DATES if 'decadal' in args.link_str_file_tag \
            else df.columns)
    create_empty_df = lambda: pd.DataFrame(np.nan, columns=locations_df.index,
        index=DECADE_END_DATES if 'decadal' in args.link_str_file_tag \
            else df.columns)
    coreness_df = create_empty_df()
    degree_df = create_empty_df()
    eccentricity_df = create_empty_df()
    shortest_path_df = create_empty_df()
    betweenness_centrality_df = create_empty_df()
    closeness_centrality_df = create_empty_df()
    eigenvector_centrality_df = create_empty_df()

    def calculate_metrics(dt, links_file):
        try:
            link_str_df = read_link_str_df(links_file)
        except FileNotFoundError:
            return
        date_summary = f'{dt.year}, {dt.strftime("%b")}'
        print(f'{date_summary}: reading link strength data from CSV file {links_file}')
        adjacency = link_str_to_adjacency(link_str_df, args.edge_density,
            args.link_str_threshold)
        graph = create_graph(adjacency)

        start = datetime.now()
        print(f'{date_summary}: calculating graph metrics...')
        # General
        shortest_paths, eccentricities = shortest_path_and_eccentricity(graph)
        coreness = nx.core_number(graph)
        coreness_df.loc[dt] = coreness
        degree_df.loc[dt] = dict(graph.degree)
        eccentricity_df.loc[dt] = eccentricities
        shortest_path_df.loc[dt] = shortest_paths
        whole_graph_metrics_df.loc[dt, 'average_coreness'] = np.mean(list(coreness.values()))
        whole_graph_metrics_df.loc[dt, 'average_degree'] = average_degree(graph)
        whole_graph_metrics_df.loc[dt, 'average_eccentricity'] = np.mean(list(eccentricities.values()))
        whole_graph_metrics_df.loc[dt, 'average_link_strength'] = np.average(link_str_df)
        whole_graph_metrics_df.loc[dt, 'average_shortest_path'] = np.mean(list(shortest_paths.values()))
        whole_graph_metrics_df.loc[dt, 'global_average_link_distance'] = global_average_link_distance(graph)
        whole_graph_metrics_df.loc[dt, 'transitivity'] = nx.transitivity(graph)
        # Partitions/modularity
        _partitions = partitions(graph)
        lcc_graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
        whole_graph_metrics_df.loc[dt, 'asyn_lpa_partitions'] = len(_partitions['asyn_lpa'])
        whole_graph_metrics_df.loc[dt, 'asyn_lpa_modularity'] = modularity(lcc_graph,
            _partitions['asyn_lpa'])
        whole_graph_metrics_df.loc[dt, 'greedy_modularity_partitions'] = len(_partitions['greedy_modularity'])
        whole_graph_metrics_df.loc[dt, 'greedy_modularity_modularity'] = modularity(lcc_graph,
            _partitions['greedy_modularity'])
        whole_graph_metrics_df.loc[dt, 'louvain_partitions'] = len(_partitions['louvain'])
        whole_graph_metrics_df.loc[dt, 'louvain_modularity'] = modularity(lcc_graph,
            _partitions['louvain'])
        # Centrality
        bc = nx.betweenness_centrality(graph)
        betweenness_centrality_df.loc[dt] = bc
        whole_graph_metrics_df.loc[dt, 'average_betweenness_centrality'] = np.mean(list(bc.values()))
        cc = nx.closeness_centrality(graph)
        closeness_centrality_df.loc[dt] = cc
        whole_graph_metrics_df.loc[dt, 'average_closeness_centrality'] = np.mean(list(cc.values()))
        try:
            ec = nx.eigenvector_centrality(graph)
            eigenvector_centrality_df.loc[dt] = ec
            whole_graph_metrics_df.loc[dt, 'average_eigenvector_centrality'] = np.mean(list(ec.values()))
        except nx.exception.PowerIterationFailedConvergence:
            print(f'PowerIterationFailedConvergence exception for {date_summary} eigenvector centrality')
        partition_sizes = {k: len(v) for k, v in _partitions.items()}
        print(f'{date_summary}: graph partitions: {partition_sizes}')
        print(f'{date_summary}: graph metrics calculated; time elapsed: {datetime.now() - start}')
        
    if 'decadal' in args.link_str_file_tag:
        links_file_d1 = f'{DATA_DIR}/link_str_{args.link_str_file_tag}_d1.csv'
        calculate_metrics(DECADE_END_DATES[0], links_file_d1)
        links_file_d2 = f'{DATA_DIR}/link_str_{args.link_str_file_tag}_d2.csv'
        calculate_metrics(DECADE_END_DATES[1], links_file_d2)
    else:
        for y in YEARS:
            if args.month:
                dt = datetime(y, args.month, 1)
                links_file = (f'{DATA_DIR}/link_str_{args.link_str_file_tag}'
                    f'_m{dt.strftime("%m_%Y")}.csv')
                calculate_metrics(dt, links_file)
            else:
                for m in MONTHS:
                    dt = datetime(y, m, 1)
                    links_file = (f'{DATA_DIR}/link_str_{args.link_str_file_tag}'
                        f'_{dt.strftime("%Y_%m")}.csv')
                    calculate_metrics(dt, links_file)

    graph_file_tag = f'ed_{str(args.edge_density).replace(".", "p")}' \
        if args.edge_density \
        else f'thr_{str(args.link_str_threshold).replace(".", "p")}'
    metrics_file_base = f'metrics_{args.link_str_file_tag}_{graph_file_tag}'
    metrics_file_base += f'_m{dt.strftime("%m")}' if args.month else ''
    print(f'Saving metrics of graphs to pickle files with base {metrics_file_base}')
    whole_graph_metrics_df.to_pickle(f'{args.output_folder}/{metrics_file_base}_whole.pkl')
    coreness_df.to_pickle(f'{args.output_folder}/{metrics_file_base}_cor.pkl')
    degree_df.to_pickle(f'{args.output_folder}/{metrics_file_base}_deg.pkl')
    eccentricity_df.to_pickle(f'{args.output_folder}/{metrics_file_base}_ecc.pkl')
    shortest_path_df.to_pickle(f'{args.output_folder}/{metrics_file_base}_sp.pkl')
    betweenness_centrality_df.to_pickle(f'{args.output_folder}/{metrics_file_base}_b_cent.pkl')
    closeness_centrality_df.to_pickle(f'{args.output_folder}/{metrics_file_base}_c_cent.pkl')
    eigenvector_centrality_df.to_pickle(f'{args.output_folder}/{metrics_file_base}_e_cent.pkl')

if __name__ == '__main__':
    start = datetime.now()
    main()
    print(f'Total time elapsed: {datetime.now() - start}')