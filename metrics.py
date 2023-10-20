from datetime import datetime
import argparse

import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import community
from geopy.distance import geodesic, great_circle

from helpers import read_link_str_df, link_str_to_adjacency, rect_to_square

YEARS = list(range(2000, 2022 + 1))
MONTHS = list(range(1, 13))

DECADE_END_DATES = [datetime(2011, 4, 1), datetime(2022, 3, 1)]
WHOLE_GRAPH_METRICS = [
    'average_betweenness_centrality',
    'average_closeness_centrality',
    'average_coreness',
    'average_degree',
    'average_eccentricity',
    'average_eigenvector_centrality',
    'average_link_strength',
    'average_shortest_path',
    'average_local_link_distance',
    'degree_assortativity',
    'fluid_modularity',
    'fluid_partitions',
    'greedy_mod_modularity',
    'greedy_mod_partitions',
    'label_prop_partitions',
    'label_prop_modularity',
    'louvain_modularity',
    'louvain_partitions',
    'transitivity',
]

def average_degree(graph: nx.Graph):
    return 2 * len(graph.edges) / len(graph.nodes)

# NOTE: slow but seemingly difficult to make faster
def shortest_path_and_eccentricity(graph: nx.Graph):
    # Calculate these manually for a network which may not be connected
    shortest_path_lengths = nx.shortest_path_length(graph)
    ecc_by_node = {}
    average_spl_by_node = {}
    # TODO: check results
    for source_node, target_nodes in shortest_path_lengths:
        ecc_by_node[source_node] = np.max(list(target_nodes.values()))
        for point, spl in target_nodes.items():
            if spl == 0:
                target_nodes[point] = np.nan
        average_spl_by_node[source_node] = np.nanmean(list(target_nodes.values()))
        # _values = np.array(list(target_nodes.values()))
        # _values = np.where(_values == 0, _values, np.nan)
        # average_spl_by_node[source_node] = np.nanmean(_values)
        # ecc_by_node[source_node] = np.max(_values)
    return average_spl_by_node, ecc_by_node

# Defined in De Castro Santos as, for each location, the average of the
# normalised geodesic distance of all the location's neighbours
# The global average link distance is further defined, for a given graph,
# as the average of its locations' local link distances.
# Great circle is used instead of geodesic as it is much faster at only
# a small expense of accuracy
def local_link_distance(graph: nx.Graph):
    link_distances = {}
    for node in graph.nodes:
        link_distances[node] = np.average([great_circle(node, x).km \
            for x in graph.neighbors(node)])
    return link_distances

# TODO: save metrics for each graph to a separate file
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', default='data')
    parser.add_argument('--edge_density', '--ed', type=float, default=0.005)
    parser.add_argument('--col_quantile', type=float, default=None)
    parser.add_argument('--link_str_threshold', type=float, default=None)
    parser.add_argument('--month', type=int, default=None)
    parser.add_argument('--data_dir', default='data/precipitation')
    parser.add_argument('--link_str_file_tag', default='corr_alm_60_lag_0')
    parser.add_argument('--num_comm_min', type=int, default=3)
    parser.add_argument('--num_comm_max', type=int, default=10)
    parser.add_argument('--fmt', default='csv')
    args = parser.parse_args()

    # First, read in one of the target link strength files to get the
    # location and time indices for initialising the arrays of metrics
    if 'decadal' in args.link_str_file_tag or 'dms' in args.link_str_file_tag:
        link_str_df = read_link_str_df(
            f'{args.data_dir}/link_str_{args.link_str_file_tag}_d1.{args.fmt}')
    elif args.month:
        link_str_df = read_link_str_df(
            f'{args.data_dir}/link_str_{args.link_str_file_tag}_m03_2022.{args.fmt}')
    else:
        link_str_df = read_link_str_df(
            f'{args.data_dir}/link_str_{args.link_str_file_tag}_2022_03.{args.fmt}')
    whole_graph_metrics_df = pd.DataFrame(columns=WHOLE_GRAPH_METRICS)
    create_empty_df = lambda: pd.DataFrame(
        columns=link_str_df.index if link_str_df.index.equals(link_str_df.columns)
            else [*link_str_df.index.values, *link_str_df.columns.values])
    coreness_df = create_empty_df()
    degree_df = create_empty_df()
    weighted_degree_df = create_empty_df()
    eccentricity_df = create_empty_df()
    shortest_path_df = create_empty_df()
    local_link_distance_df = create_empty_df()
    betweenness_centrality_df = create_empty_df()
    closeness_centrality_df = create_empty_df()
    eigenvector_centrality_df = create_empty_df()

    def calculate_metrics(dt, link_str_df):
        date_summary = f'{dt.year}, {dt.strftime("%b")}'
        adjacency = link_str_to_adjacency(link_str_df, args.edge_density,
            args.link_str_threshold, col_quantile=args.col_quantile)
        graph = nx.from_numpy_array(adjacency.values)
        graph = nx.relabel_nodes(graph, dict(enumerate(adjacency.columns)))
        w_adjacency = rect_to_square(link_str_df)
        w_graph = nx.from_numpy_array(w_adjacency)
        w_graph = nx.relabel_nodes(w_graph, dict(enumerate(adjacency.columns)))
        # Graphs from multiple variables may have self edges
        for g in graph, w_graph:
            g.remove_edges_from(nx.selfloop_edges(g))
        
        start = datetime.now()
        print(f'{date_summary}: calculating graph metrics...')
        # General
        shortest_paths, eccentricities = shortest_path_and_eccentricity(graph)
        coreness = nx.core_number(graph)
        coreness_df.loc[dt] = coreness
        degree_df.loc[dt] = dict(graph.degree)
        weighted_degree_df.loc[dt] = dict(w_graph.degree(weight='weight'))
        eccentricity_df.loc[dt] = eccentricities
        shortest_path_df.loc[dt] = shortest_paths
        local_link_distance_df.loc[dt] = local_link_distance(graph)
        whole_graph_metrics_df.loc[dt, 'average_coreness'] = np.mean(list(coreness.values()))
        whole_graph_metrics_df.loc[dt, 'average_degree'] = average_degree(graph)
        whole_graph_metrics_df.loc[dt, 'average_eccentricity'] = np.mean(list(eccentricities.values()))
        whole_graph_metrics_df.loc[dt, 'average_link_strength'] = np.average(link_str_df)
        whole_graph_metrics_df.loc[dt, 'average_shortest_path'] = np.mean(list(shortest_paths.values()))
        whole_graph_metrics_df.loc[dt, 'average_local_link_distance'] = np.average(local_link_distance_df)
        whole_graph_metrics_df.loc[dt, 'transitivity'] = nx.transitivity(graph)
        whole_graph_metrics_df.loc[dt, 'degree_assortativity'] = nx.degree_assortativity_coefficient(graph)
        # Partitions/modularity
        lcc_graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
        fluid_partitions_all = {}
        fluid_partitions_modularity = {}
        for n in range(args.num_comm_min, args.num_comm_max + 1):
            fluid_partitions_all[n] = [p for p in \
                nx.algorithms.community.asyn_fluidc(lcc_graph, n, seed=0)]
            fluid_partitions_modularity[n] = community.modularity(lcc_graph,
                communities=fluid_partitions_all[n])
        fluid_partitions = fluid_partitions_all[max(fluid_partitions_modularity,
            key=fluid_partitions_modularity.get)]
        greedy_mod_partitions = [p for p in \
            nx.algorithms.community.greedy_modularity_communities(lcc_graph,
            cutoff=args.num_comm_min, best_n=args.num_comm_max)]
        label_prop_partitions = [p for p in \
            nx.algorithms.community.asyn_lpa_communities(lcc_graph, seed=0)]
        louvain_partitions = [p for p in \
            nx.algorithms.community.louvain_communities(lcc_graph, seed=0)]
        whole_graph_metrics_df.loc[dt, 'fluid_partitions'] = len(fluid_partitions)
        whole_graph_metrics_df.loc[dt, 'fluid_modularity'] = community.modularity(lcc_graph,
            fluid_partitions)
        whole_graph_metrics_df.loc[dt, 'greedy_mod_partitions'] = len(greedy_mod_partitions)
        whole_graph_metrics_df.loc[dt, 'greedy_mod_modularity'] = community.modularity(lcc_graph,
            greedy_mod_partitions)
        whole_graph_metrics_df.loc[dt, 'label_prop_partitions'] = len(label_prop_partitions)
        whole_graph_metrics_df.loc[dt, 'label_prop_modularity'] = community.modularity(lcc_graph,
            label_prop_partitions)
        whole_graph_metrics_df.loc[dt, 'louvain_partitions'] = len(louvain_partitions)
        whole_graph_metrics_df.loc[dt, 'louvain_modularity'] = community.modularity(lcc_graph,
            louvain_partitions)
        # Centrality
        bc = nx.betweenness_centrality(graph) # can take a while
        betweenness_centrality_df.loc[dt] = bc
        whole_graph_metrics_df.loc[dt, 'average_betweenness_centrality'] = np.mean(list(bc.values()))
        # betweenness_centrality_df.loc[dt] = {k: 0 for k in eccentricities}
        # whole_graph_metrics_df.loc[dt, 'average_betweenness_centrality'] = 0
        cc = nx.closeness_centrality(graph)
        closeness_centrality_df.loc[dt] = cc
        whole_graph_metrics_df.loc[dt, 'average_closeness_centrality'] = np.mean(list(cc.values()))
        try:
            # NOTE: EC often fails with the default tolerance
            ec = nx.eigenvector_centrality(graph, tol=1e-03)
            eigenvector_centrality_df.loc[dt] = ec
            whole_graph_metrics_df.loc[dt, 'average_eigenvector_centrality'] = np.mean(list(ec.values()))
        except nx.exception.PowerIterationFailedConvergence:
            print(f'PowerIterationFailedConvergence exception for {date_summary} eigenvector centrality')
        print(f'{date_summary}: {len(fluid_partitions)} fluid_partitions')
        print(f'{date_summary}: {len(greedy_mod_partitions)} greedy_mod_partitions')
        print(f'{date_summary}: {len(label_prop_partitions)} label_prop_partitions')
        print(f'{date_summary}: {len(louvain_partitions)} louvain_partitions')
        print(f'{date_summary}: graph metrics calculated; time elapsed: {datetime.now() - start}')
        
    if 'decadal' in args.link_str_file_tag or 'dms' in args.link_str_file_tag:
        calculate_metrics(DECADE_END_DATES[0], read_link_str_df(
            f'{args.data_dir}/link_str_{args.link_str_file_tag}_d1.{args.fmt}'))
        calculate_metrics(DECADE_END_DATES[1], read_link_str_df(
            f'{args.data_dir}/link_str_{args.link_str_file_tag}_d2.{args.fmt}'))
    elif args.month:
        for y in YEARS:
            dt = datetime(y, args.month, 1)
            links_file = (f'{args.data_dir}/link_str_{args.link_str_file_tag}'
                f'_m{dt.strftime("%m_%Y")}.{args.fmt}')
            try:
                link_str_df = read_link_str_df(links_file)
            except FileNotFoundError:
                continue
            calculate_metrics(dt, link_str_df)
    else:
        for y in YEARS:
            for m in MONTHS:
                dt = datetime(y, m, 1)
                links_file = (f'{args.data_dir}/link_str_{args.link_str_file_tag}'
                    f'_{dt.strftime("%Y_%m")}.{args.fmt}')
                try:
                    link_str_df = read_link_str_df(links_file)
                except FileNotFoundError:
                    continue
                calculate_metrics(dt, link_str_df)

    graph_file_tag = f'ed_{str(args.edge_density).replace(".", "p")}' \
        if args.edge_density \
        else f'thr_{str(args.link_str_threshold).replace(".", "p")}'
    if args.col_quantile:
        graph_file_tag += f'_cq_{str(args.col_quantile).replace(".", "p")}'
    metrics_file_base = f'metrics_{args.link_str_file_tag}_{graph_file_tag}'
    metrics_file_base += f'_m{dt.strftime("%m")}' if args.month else ''
    print(f'Saving metrics of graphs to pickle files with base {metrics_file_base}')
    whole_graph_metrics_df.to_pickle(f'{args.output_folder}/{metrics_file_base}_whole.pkl')
    coreness_df.to_pickle(f'{args.output_folder}/{metrics_file_base}_cor.pkl')
    degree_df.to_pickle(f'{args.output_folder}/{metrics_file_base}_deg.pkl')
    weighted_degree_df.to_pickle(f'{args.output_folder}/{metrics_file_base}_wdeg.pkl')
    eccentricity_df.to_pickle(f'{args.output_folder}/{metrics_file_base}_ecc.pkl')
    shortest_path_df.to_pickle(f'{args.output_folder}/{metrics_file_base}_sp.pkl')
    local_link_distance_df.to_pickle(f'{args.output_folder}/{metrics_file_base}_lld.pkl')
    betweenness_centrality_df.to_pickle(f'{args.output_folder}/{metrics_file_base}_b_cent.pkl')
    closeness_centrality_df.to_pickle(f'{args.output_folder}/{metrics_file_base}_c_cent.pkl')
    eigenvector_centrality_df.to_pickle(f'{args.output_folder}/{metrics_file_base}_e_cent.pkl')

if __name__ == '__main__':
    start = datetime.now()
    main()
    print(f'Total time elapsed: {datetime.now() - start}')