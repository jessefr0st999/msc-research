from datetime import datetime
import pickle
import argparse

import pandas as pd
import numpy as np
import networkx as nx

from helpers import *

YEARS = list(range(2000, 2022 + 1))
MONTHS = list(range(1, 13))
DATA_DIR = 'data/precipitation'
DATA_FILE = f'{DATA_DIR}/FusedData.csv'
LOCATIONS_FILE = f'{DATA_DIR}/Fused.Locations.csv'

def create_graph(adjacency: pd.DataFrame):
    graph = nx.from_numpy_array(adjacency.values)
    graph = nx.relabel_nodes(graph, dict(enumerate(adjacency.columns)))
    return graph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', default='outputs')
    parser.add_argument('--edge_density', type=float, default=0.005)
    parser.add_argument('--link_str_threshold', type=float, default=None)
    parser.add_argument('--link_str_file_tag', default='alm_60_lag_0')
    args = parser.parse_args()

    def calculate_network_metrics(graph):
        # shortest_paths, eccentricities = shortest_path_and_eccentricity(graph)
        _partitions = partitions(graph)
        lcc_graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
        return {
            'average_degree': average_degree(graph),
            'transitivity': transitivity(graph),
            'coreness': coreness(graph),
            # 'global_average_link_distance': global_average_link_distance(graph),
            # 'shortest_path': np.average(list(shortest_paths.values())),
            # 'eccentricity': np.average(list(eccentricities.values())),
            # Centrality
            'eigenvector_centrality': nx.eigenvector_centrality(graph),
            'betweenness_centrality': nx.betweenness_centrality(graph),
            'closeness_centrality': nx.closeness_centrality(graph),
            # Partitions/modularity
            'louvain_partitions': len(_partitions['louvain']),
            'louvain_modularity': modularity(lcc_graph, _partitions['louvain']),
            'greedy_modularity_partitions': len(_partitions['greedy_modularity']),
            'greedy_modularity_modularity': modularity(lcc_graph, _partitions['greedy_modularity']),
            'asyn_lpa_partitions': len(_partitions['asyn_lpa']),
            'asyn_lpa_modularity': modularity(lcc_graph, _partitions['asyn_lpa']),
        }, _partitions

    metrics_list = []
    for y in YEARS:
        for m in MONTHS:
            dt = datetime(y, m, 1)
            links_file = f'{DATA_DIR}/link_str_{args.link_str_file_tag}_{dt.strftime("%Y_%m")}.pkl'
            try:
                link_str_df: pd.DataFrame = pd.read_pickle(links_file)
            except FileNotFoundError:
                continue
            date_summary = f'{dt.year}, {dt.strftime("%b")}'
            print(f'{date_summary}: reading link strength data from pickle file {links_file}')

            adjacency = pd.DataFrame(0, columns=link_str_df.columns, index=link_str_df.index)
            if args.edge_density:
                threshold = np.quantile(link_str_df, 1 - args.edge_density)
                print(f'{date_summary}: fixed edge density {args.edge_density} gives threshold {threshold}')
            else:
                threshold = args.link_str_threshold
            adjacency[link_str_df >= threshold] = 1
            if not args.edge_density:
                _edge_density = np.sum(np.sum(adjacency)) / adjacency.size
                print(f'{date_summary}: fixed threshold {args.link_str_threshold} gives edge density {_edge_density}')
            graph = create_graph(adjacency)

            start = datetime.now()
            print(f'{date_summary}: calculating graph metrics...')
            metrics, _partitions = calculate_network_metrics(graph)
            partition_sizes = {k: len(v) for k, v in _partitions.items()}
            print(f'{date_summary}: graph partitions: {partition_sizes}')
            print(f'{date_summary}: graph metrics calculated; time elapsed: {datetime.now() - start}')
            metrics_list.append({
                'dt': dt,
                **metrics,
                'average_link_strength': np.average(link_str_df),
            })

    graph_file_tag = f'ed_{str(args.edge_density).replace(".", "p")}' if args.edge_density \
        else f'thr_{str(args.link_str_threshold).replace(".", "p")}'
    metrics_file = f'metrics_{args.link_str_file_tag}_{graph_file_tag}.pkl'
    print(f'Saving metrics of graphs to pickle file {metrics_file}')
    metrics_file = f'{args.output_folder}/{metrics_file}'
    with open(metrics_file, 'wb') as f:
        pickle.dump(metrics_list, f)

if __name__ == '__main__':
    start = datetime.now()
    main()
    print(f'Total time elapsed: {datetime.now() - start}')