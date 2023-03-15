import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import community
from fractions import Fraction

partition_colours = ['red', 'cyan', 'green', 'orange', 'yellow', 'pink', 'white', 'grey']
n_comm_min = 2
n_comm_max = 6

# Modularity example
graph = nx.from_edgelist([
    (1, 2), (1, 3), (2, 4), (3, 4),
    (3, 5), (5, 6), (5, 7),
    (6, 7), (5, 8), (7, 8),
])
# BC and CC example
# graph = nx.from_edgelist([
#     (1, 2), (1, 3), (2, 4), (3, 4),
#     (3, 5), (4, 5), (5, 6), (5, 7),
#     (6, 7),
# ])
# Degree, coreness and eccentricity example
# graph = nx.from_edgelist([
#     (1, 2), (2, 4), (3, 4),
#     (3, 5), (4, 5), (5, 6), (5, 7),
#     (6, 7), (6, 8), (7, 8), (6, 9), (7, 9), (8, 9),
#     (7, 10), (9, 10),
# ])

# Examples for March 2023 presentation
# graph = nx.random_geometric_graph(30, 0.35, seed=1)

# asyn_lpa_communities and asyn_fluidc example
# graph = nx.random_geometric_graph(12, 0.5, seed=5)

# greedy_modularity_communities example
# graph = nx.random_geometric_graph(30, 0.5, seed=5)

bc = nx.betweenness_centrality(graph, normalized=True)
cc = nx.closeness_centrality(graph)
# ec = nx.eigenvector_centrality(graph)
coreness = nx.core_number(graph)
ecc = nx.eccentricity(graph)
degree = nx.degree(graph)

# partitions = [p for p in next(community.girvan_newman(graph))]
partitions = [p for p in community.asyn_lpa_communities(graph, seed=0)]
# partitions = [p for p in community.greedy_modularity_communities(graph,
#     cutoff=n_comm_min, best_n=n_comm_max)]
# partitions = [p for p in community.louvain_communities(graph, seed=1)]
# fluid_partitions = {}
# fluid_partitions_modularity = {}
# for n in range(n_comm_min, n_comm_max + 1):
#     fluid_partitions[n] = [p for p in community.asyn_fluidc(graph, n, seed=1)]
#     fluid_partitions_modularity[n] = nx.algorithms.community.modularity(
#         graph, communities=fluid_partitions[n])
# partitions = fluid_partitions[max(fluid_partitions_modularity,
#     key=fluid_partitions_modularity.get)]

# Optimal
# partitions = [
#     {1, 2, 3, 4},
#     {8, 5, 6, 7}
# ]
# Sub-optimal
# partitions = [
#     {1, 2, 3, 4, 5},
#     {8, 6, 7}
# ]
# One community
# partitions = [
#     {1, 2, 3, 4, 8, 5, 6, 7}
# ]
# Each node in a community
# partitions = [
#     {1}, {2}, {3}, {4}, {8}, {5}, {6}, {7}
# ]
# Poor communities
# partitions = [
#     {1, 6}, {2, 3, 7}, {5}, {4, 8}
# ]
# Pseudo-random
# partitions = [
#     {2, 6}, {1, 3, 7}, {4}, {5, 8}
# ]

print(f'Number of communities: {len(partitions)}')
print(f'Modularity of communities: {community.modularity(graph, communities=partitions)}')

node_colours = [None for _ in graph.nodes]
for i, p in enumerate(partitions):
    for node in p:
        node_index = list(graph.nodes).index(node)
        node_colours[node_index] = partition_colours[i]

labels = {i: i for i in graph.nodes}
# labels = {i: str(Fraction(bc[i]).limit_denominator()) for i in graph.nodes}
# labels = {i: f'{i}\n{str(Fraction(bc[i]).limit_denominator())}\n{str(Fraction(cc[i]).limit_denominator())}' for i in graph.nodes}
# labels = {i: round(bc[i], 3) for i in graph.nodes}
# labels = {i: f'{i}, {coreness[i]}' for i in graph.nodes}
# labels = {i: degree[i] for i in graph.nodes}
# labels = {i: f'{i}, {round(cc[i], 3)}' for i in graph.nodes}

options = {'edgecolors': 'tab:gray', 'node_size': 700, 'alpha': 0.9}
pos = nx.spring_layout(graph, seed=0)
# nx.draw_networkx_nodes(graph, pos, node_color='orange', **options)
nx.draw_networkx_nodes(graph, pos, node_color=node_colours, **options)
nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
nx.draw_networkx_labels(graph, pos, labels)

plt.tight_layout()
plt.axis('off')
plt.show()

# Weighted degree example
# graph = nx.graphraph()
# graph.add_node(1, pos=(1,1))
# graph.add_node(2, pos=(2,2))
# graph.add_node(3, pos=(1,0))
# graph.add_node(4, pos=(0,1))
# graph.add_edge(1, 2, weight=0.5)
# graph.add_edge(1, 3, weight=2)
# graph.add_edge(2, 3, weight=1)
# graph.add_edge(1, 4, weight=1)

# pos = nx.get_node_attributes(graph, 'pos')
# edge_labels = nx.get_edge_attributes(graph, 'weight')
# degree = nx.degree(graph, weight='weight')
# labels = {i: degree[i] for i in graph.nodes}

# options = {'edgecolors': 'tab:gray', 'node_size': 700, 'alpha': 0.9}
# nx.draw_networkx_nodes(graph, pos, node_color='orange', **options)
# nx.draw_networkx_nodes(graph, pos, node_color=node_colours, **options)
# nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
# nx.draw_networkx_labels(graph, pos, labels)
# nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
# plt.show()