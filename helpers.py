import networkx as nx
from scipy.sparse.linalg import eigs

def average_degree(graph: nx.Graph):
    return 2 * len(graph.edges) / len(graph.nodes)

def transitivity(graph: nx.Graph):
    return nx.transitivity(graph)

# TODO: update this
def eigenvector_centrality(graph: nx.Graph):
    adjacency = nx.adjacency_matrix(graph)
    max_eval, max_evec = eigs(adjacency.asfptype(), k=1, which='LM')
    return max_evec[0]

def coreness(graph: nx.Graph):
    k_indices = nx.core_number(graph).values()
    return sum(k_indices) / len(graph.nodes)

def average_shortest_path(graph: nx.Graph):
    try:
        length = nx.average_shortest_path_length(graph)
    except nx.exception.NetworkXError:
        length = None
    return length

def eccentricity(graph: nx.Graph):
    try:
        ecc = nx.eccentricity(graph).values()
        avg_ecc = sum(ecc) / len(graph.nodes)
    except nx.exception.NetworkXError:
        avg_ecc = None
    return avg_ecc

# TODO
def global_average_link_distance(graph: nx.Graph):
    return None

def modularity(graph: nx.Graph):
    partitions = [p for p in nx.algorithms.community.louvain_partitions(graph)]
    return nx.algorithms.community.modularity(graph, communities=partitions[0])