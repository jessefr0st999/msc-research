import argparse
from datetime import datetime

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms import community
from helpers import get_map

DATA_DIR = 'data/precipitation'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prec_file', default='dataframe_drop_0_alm_12_lag_0.pkl')
    parser.add_argument('--link_str_file', default='link_str_drop_0_alm_12_lag_0_2006_01.pkl')
    parser.add_argument('--edge_density', type=float, default=0.005)
    parser.add_argument('--num_af_communities', type=int, default=6)
    args = parser.parse_args()

    prec_df = pd.read_pickle(f'{DATA_DIR}/{args.prec_file}')
    link_str_df = pd.read_pickle(f'{DATA_DIR}/{args.link_str_file}')
    date_part = args.link_str_file.split('lag_')[1].split('.pkl')[0]
    _, year, month = date_part.split('_')
    location_df = prec_df.loc[datetime(int(year), int(month), 1)]
    threshold = np.quantile(link_str_df, 1 - args.edge_density)
    print(f'Fixed edge density {args.edge_density} gives threshold {threshold}')
    adjacency = pd.DataFrame(0, columns=link_str_df.columns, index=link_str_df.index)
    adjacency[link_str_df >= threshold] = 1
    graph = nx.from_numpy_matrix(adjacency.values)
    graph = nx.relabel_nodes(graph, dict(enumerate(adjacency.columns)))

    lcc_graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
    lv_partitions = [p for p in community.louvain_communities(lcc_graph)]
    gm_partitions = [p for p in community.greedy_modularity_communities(lcc_graph)]
    af_partitions = [p for p in community.asyn_fluidc(lcc_graph, args.num_af_communities)]
    # al_partitions = [p for p in community.asyn_lpa_communities(lcc_graph, seed=0)]
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    lv_node_colours = []
    gm_node_colours = []
    af_node_colours = []
    for n in lcc_graph.nodes:
        for i, p in enumerate(lv_partitions):
            if n in p:
                lv_node_colours.append(colours[i % len(colours)])
                break
        for i, p in enumerate(gm_partitions):
            if n in p:
                gm_node_colours.append(colours[i % len(colours)])
                break
        for i, p in enumerate(af_partitions):
            if n in p:
                af_node_colours.append(colours[i % len(colours)])
                break

    figure, axes = plt.subplots(1, 3)
    axes = axes.flatten()
    axes[0].set_title('lv_partitions')
    axes[1].set_title('gm_partitions')
    axes[2].set_title(f'af_partitions, {args.num_af_communities} communities')
    lv_map = get_map(axes[0])
    gm_map = get_map(axes[1])
    af_map = get_map(axes[2])
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
    nx.draw_networkx_nodes(G=lcc_graph, pos=pos, nodelist=lcc_graph.nodes(),
        node_color=af_node_colours, alpha=0.8, ax=axes[2],
        node_size=[40 + 3*adjacency[location].sum() for location in lcc_graph.nodes()])
    nx.draw_networkx_edges(G=lcc_graph, pos=pos, edge_color='gray',
        alpha=0.2, arrows=False, ax=axes[0])
    nx.draw_networkx_edges(G=lcc_graph, pos=pos, edge_color='gray',
        alpha=0.2, arrows=False, ax=axes[1])
    nx.draw_networkx_edges(G=lcc_graph, pos=pos, edge_color='gray',
        alpha=0.2, arrows=False, ax=axes[2])
    plt.tight_layout()
    figure.set_size_inches(32, 18)
    filename = (f'images/communities_ed_{str(args.edge_density).replace(".", "p")}'
        f'_{args.link_str_file.split("link_str_")[1]}.png')
    plt.savefig(filename, bbox_inches='tight')
    print(f'Saved to file {filename}')
    # plt.show()

main()