from datetime import datetime
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

def create_graph(adjacency: pd.DataFrame):
    graph = nx.from_numpy_array(adjacency.values)
    graph = nx.relabel_nodes(graph, dict(enumerate(adjacency.columns)))
    return graph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', default='outputs')
    parser.add_argument('--edge_density', type=float, default=0.005)
    parser.add_argument('--link_str_threshold', type=float, default=None)
    parser.add_argument('--link_str_file_tag', default='corr_alm_60_lag_0')
    args = parser.parse_args()

    prec_df = pd.read_csv(DATA_FILE)
    prec_df.columns = pd.to_datetime(prec_df.columns, format='D%Y.%m')
    locations_df = pd.read_csv(LOCATIONS_FILE).set_index(['Lat', 'Lon'])
    whole_graph_metrics_df = pd.DataFrame(np.nan, index=prec_df.columns,
        columns=WHOLE_GRAPH_METRICS)
    create_empty_df = lambda: pd.DataFrame(np.nan, index=prec_df.columns,
        columns=locations_df.index)
    coreness_df = create_empty_df()
    degree_df = create_empty_df()
    eccentricity_df = create_empty_df()
    shortest_path_df = create_empty_df()
    betweenness_centrality_df = create_empty_df()
    closeness_centrality_df = create_empty_df()
    eigenvector_centrality_df = create_empty_df()
    for y in YEARS:
        for m in MONTHS:
            dt = datetime(y, m, 1)
            links_file = f'{DATA_DIR}/link_str_{args.link_str_file_tag}_{dt.strftime("%Y_%m")}.csv'
            try:
                link_str_df = read_link_str_df(links_file)
            except FileNotFoundError:
                continue
            date_summary = f'{dt.year}, {dt.strftime("%b")}'
            print(f'{date_summary}: reading link strength data from CSV file {links_file}')

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
            
    graph_file_tag = f'ed_{str(args.edge_density).replace(".", "p")}' if args.edge_density \
        else f'thr_{str(args.link_str_threshold).replace(".", "p")}'
    metrics_file_base = f'metrics_{args.link_str_file_tag}_{graph_file_tag}'
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