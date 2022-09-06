import argparse
from datetime import datetime

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms import community
from mpl_toolkits.basemap import Basemap

# threshold = 3.1
EDGE_DENSITY = 0.025

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2006)
    args = parser.parse_args()

    prec_file = 'data/precipitation/dataframe_drop_50.pkl'
    prec_df = pd.read_pickle(prec_file)
    location_df = prec_df.loc[datetime(args.year, 1, 1)]
    link_str_file = f'data/precipitation/link_str_drop_50_{args.year}_01.pkl'
    link_str_df = pd.read_pickle(link_str_file)
    threshold = np.quantile(link_str_df, 1 - EDGE_DENSITY)
    print(f'Fixed edge density {EDGE_DENSITY} gives threshold {threshold}')
    adjacency = pd.DataFrame(0, columns=link_str_df.columns, index=link_str_df.index)
    adjacency[link_str_df >= threshold] = 1
    graph = nx.from_numpy_matrix(adjacency.values)
    graph = nx.relabel_nodes(graph, dict(enumerate(adjacency.columns)))

    lcc_graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
    lv_partitions = [p for p in community.louvain_communities(lcc_graph)]
    gm_partitions = [p for p in community.greedy_modularity_communities(lcc_graph)]
    # al_partitions = [p for p in community.asyn_lpa_communities(lcc_graph, seed=0)]
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    lv_node_colours = []
    gm_node_colours = []
    for n in lcc_graph.nodes:
        for i, p in enumerate(lv_partitions):
            if n in p:
                lv_node_colours.append(colours[i % len(colours)])
                break
        for i, p in enumerate(gm_partitions):
            if n in p:
                gm_node_colours.append(colours[i % len(colours)])
                break

    def get_map(axis):
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

    figure, axes = plt.subplots(1, 2)
    axes[0].set_title('lv_partitions')
    axes[1].set_title('gm_partitions')
    lv_map = get_map(axes[0])
    gm_map = get_map(axes[1])
    mx, my = lv_map(location_df['lon'], location_df['lat'])
    pos = {}
    for i, elem in enumerate(adjacency.index):
        pos[elem] = (mx[i], my[i])
    nx.draw_networkx_nodes(G=lcc_graph, pos=pos, nodelist=lcc_graph.nodes(),
        node_color=lv_node_colours, alpha=0.8, ax=axes[0],
        node_size=[40 + 3*adjacency[location].sum() for location in lcc_graph.nodes()])
    nx.draw_networkx_nodes(G=lcc_graph, pos=pos, nodelist=lcc_graph.nodes(),
        node_color=gm_node_colours, alpha=0.8, ax=axes[1],
        node_size=[40 + 3*adjacency[location].sum() for location in lcc_graph.nodes()])
    nx.draw_networkx_edges(G=lcc_graph, pos=pos, edge_color='gray',
        alpha=0.2, arrows=False, ax=axes[0])
    nx.draw_networkx_edges(G=lcc_graph, pos=pos, edge_color='gray',
        alpha=0.2, arrows=False, ax=axes[1])
    plt.tight_layout()
    figure.set_size_inches(32, 18)
    filename = f'images/communities_January_{args.year}.png'
    plt.savefig(filename, bbox_inches='tight')
    print(f'Saved to file {filename}')
    # plt.show()

main()