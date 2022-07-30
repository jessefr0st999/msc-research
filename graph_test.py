import networkx as nx
import matplotlib.pyplot as plt

graph = nx.Graph()

graph.add_weighted_edges_from([
    ('a', 'b', 5),
    ('b', 'c', 3),
    ('a', 'c', 1),
    ('c', 'd', 1),
])

# for g in graph.edges:
#     print(g)

subax1 = plt.subplot(121)
nx.draw(graph)
subax2 = plt.subplot(122)
nx.draw(graph, pos=nx.circular_layout(graph), node_color='r', edge_color='b')
plt.show()