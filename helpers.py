import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs

def average_degree(graph: nx.Graph):
    return 2 * len(graph.edges) / len(graph.nodes)

# TODO: Check if equal to triangle density
def transitivity(graph: nx.Graph):
    return nx.transitivity(graph)

# TODO: Fix
def eigenvector_centrality(adjacency):
    max_eval, max_evec = eigs(adjacency.astype('d'), k=1, which='LM')
    return np.average(np.real(max_evec))

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

def modularity(graph: nx.Graph):
    partitions = [p for p in nx.algorithms.community.louvain_partitions(graph)]
    return nx.algorithms.community.modularity(graph, communities=partitions[0])

# TODO: implement
def global_average_link_distance(graph: nx.Graph):
    return None