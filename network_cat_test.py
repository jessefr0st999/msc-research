import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from networkx.algorithms import community
from mpl_toolkits.basemap import Basemap

LINK_STR_THRESHOLD = 3.1

def main():
    prec_file = 'data/precipitation/dataframe_drop_50.pkl'
    prec_df = pd.read_pickle(prec_file)
    location_df = prec_df.loc[datetime(2001, 9, 1)]
    link_str_file = 'data/precipitation/link_str_drop_50_2001_09.pkl'
    link_str_df = pd.read_pickle(link_str_file)
    adjacency = pd.DataFrame(0, columns=link_str_df.columns, index=link_str_df.index)
    adjacency[link_str_df >= LINK_STR_THRESHOLD] = 1
    graph = nx.from_numpy_matrix(adjacency.values)
    graph = nx.relabel_nodes(graph, dict(enumerate(adjacency.columns)))

    # partitions = [p for p in community.kernighan_lin_bisection(graph)]
    # partitions = [p for p in community.louvain_partitions(graph, resolution=1)]
    lcc_graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
    partitions = [p for p in community.asyn_fluidc(lcc_graph, k=4)]
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    node_colours = []
    for n in lcc_graph.nodes:
        for i, p in enumerate(partitions):
            if n in p:
                node_colours.append(colours[i])
                break

    graph_map = Basemap(
        projection='merc',
        llcrnrlon=110,
        llcrnrlat=-45,
        urcrnrlon=155,
        urcrnrlat=-10,
        lat_ts=0,
        resolution='l',
        suppress_ticks=True,
    )
    mx, my = graph_map(location_df['lon'], location_df['lat'])
    pos = {}
    for i, elem in enumerate(adjacency.index):
        pos[elem] = (mx[i], my[i])
    nx.draw_networkx_nodes(G=lcc_graph, pos=pos, nodelist=lcc_graph.nodes(),
        node_color=node_colours, alpha=0.8,
        node_size=[30 + 2*adjacency[location].sum() for location in lcc_graph.nodes()])
    nx.draw_networkx_edges(G=lcc_graph, pos=pos, edge_color='gray',
        alpha=0.2, arrows=False)
    graph_map.drawcountries(linewidth=3)
    graph_map.drawstates(linewidth=0.2)
    graph_map.drawcoastlines(linewidth=3)
    plt.tight_layout()
    plt.show()

main()