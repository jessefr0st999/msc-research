import networkx as nx
import numpy as np
from networkx.algorithms import community
from mpl_toolkits.basemap import Basemap

def average_degree(graph: nx.Graph):
    return 2 * len(graph.edges) / len(graph.nodes)

def transitivity(graph: nx.Graph):
    return nx.transitivity(graph)

def eigenvector_centrality(graph):
    try:
        centrality = nx.eigenvector_centrality(graph)
    except nx.exception.PowerIterationFailedConvergence:
        return np.nan
    return sum(centrality.values()) / len(graph.nodes)

def betweenness_centrality(graph):
    centrality = nx.betweenness_centrality(graph)
    return sum(centrality.values()) / len(graph.nodes)

def closeness_centrality(graph):
    centrality = nx.closeness_centrality(graph)
    return sum(centrality.values()) / len(graph.nodes)

def coreness(graph: nx.Graph):
    k_indices = nx.core_number(graph).values()
    return sum(k_indices) / len(graph.nodes)

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
        'louvain': [p for p in community.louvain_communities(lcc_graph)],
        'greedy_modularity': [p for p in community.greedy_modularity_communities(lcc_graph)],
        'asyn_lpa': [p for p in community.asyn_lpa_communities(lcc_graph, seed=0)],
        # This one causes the script to hang for an ~1000 edge graph
        # 'girvan_newman': [p for p in community.girvan_newman(lcc_graph)],
    }

def modularity(graph: nx.Graph, communities):
    return community.modularity(graph, communities=communities)

# TODO: implement
def global_average_link_distance(graph: nx.Graph):
    return None

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