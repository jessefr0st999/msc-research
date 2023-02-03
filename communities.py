import argparse
from datetime import datetime

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms import community

from helpers import read_link_str_df, configure_plots, get_map

DATA_DIR = 'data/precipitation'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prec_seq_file', default='prec_seq_alm_60_lag_0.pkl')
    parser.add_argument('--link_str_file', default='link_str_alm_60_lag_0_2022_03.csv')
    parser.add_argument('--output_folder', default=None)
    parser.add_argument('--edge_density', type=float, default=0.005)
    parser.add_argument('--num_af_communities', type=int, default=6)
    args = parser.parse_args()
    label_size, font_size, show_or_save = configure_plots(args)

    prec_seq_df = pd.read_pickle(f'{DATA_DIR}/{args.prec_seq_file}')
    link_str_df = read_link_str_df(f'{DATA_DIR}/{args.link_str_file}')
    try:
        if 'decadal' in args.link_str_file:
            year, month = (2011, 4) if 'd1' in args.link_str_file \
                else (2022, 3)
        else:
            date_part = args.link_str_file.split('lag_')[1].split('.csv')[0]
            _, year, month = date_part.split('_')
    except IndexError:
        # Month-only link strength files
        year = int(args.link_str_file.split('_')[-1][:4])
        month = int(args.link_str_file.split('_')[-2][-2:])
    dt = datetime(int(year), int(month), 1)
    dt_prec_seq_df = prec_seq_df.loc[dt]
    threshold = np.quantile(link_str_df, 1 - args.edge_density)
    print(f'Fixed edge density {args.edge_density} gives threshold {threshold}')
    adjacency = pd.DataFrame(0, columns=link_str_df.columns, index=link_str_df.index)
    adjacency[link_str_df >= threshold] = 1
    graph = nx.from_numpy_array(adjacency.values)
    graph = nx.relabel_nodes(graph, dict(enumerate(adjacency.columns)))

    lcc_graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
    lv_partitions = [p for p in community.louvain_communities(lcc_graph)]
    gm_partitions = [p for p in community.greedy_modularity_communities(lcc_graph)]
    # NOTE: below algorithms are random, hence why a seed is specified
    af_partitions = [p for p in community.asyn_fluidc(lcc_graph, args.num_af_communities, seed=0)]
    al_partitions = [p for p in community.asyn_lpa_communities(lcc_graph, seed=0)]
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 'black', '#444']

    # Calculate colours of each node based on community
    lv_node_colours = []
    gm_node_colours = []
    af_node_colours = []
    al_node_colours = []
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
        for i, p in enumerate(al_partitions):
            if n in p:
                al_node_colours.append(colours[i % len(colours)])
                break

    figure, axes = plt.subplots(2, 2, layout='compressed')
    axes = axes.flatten()
    axes[0].set_title(f'{dt.strftime("%b %Y")}: lv_partitions, {len(lv_partitions)} communities')
    axes[1].set_title(f'{dt.strftime("%b %Y")}: gm_partitions, {len(gm_partitions)} communities')
    axes[2].set_title(f'{dt.strftime("%b %Y")}: af_partitions, {args.num_af_communities} communities')
    axes[3].set_title(f'{dt.strftime("%b %Y")}: al_partitions, {len(al_partitions)} communities')
    lv_map = get_map(axes[0])
    gm_map = get_map(axes[1])
    af_map = get_map(axes[2])
    al_map = get_map(axes[3])
    mx, my = lv_map(dt_prec_seq_df['lon'], dt_prec_seq_df['lat'])
    pos = {}
    for i, elem in enumerate(adjacency.index):
        pos[elem] = (mx[i], my[i])
    for (axis, colours) in zip (axes, [lv_node_colours, gm_node_colours,
            af_node_colours, al_node_colours]):
        nx.draw_networkx_nodes(G=lcc_graph, pos=pos, nodelist=lcc_graph.nodes(),
            node_color=colours, alpha=0.8, ax=axis, node_size=80)
        nx.draw_networkx_edges(G=lcc_graph, pos=pos, edge_color='gray',
            alpha=0.2, arrows=False, ax=axis)
    filename = (f'communities_ed_{str(args.edge_density).replace(".", "p")}'
        f'_{args.link_str_file.split("link_str_")[1].split(".csv")[0]}.png')
    show_or_save(figure, filename)

main()