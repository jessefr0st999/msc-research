import argparse
from datetime import datetime

import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community

from helpers import read_link_str_df, configure_plots, get_map, \
    link_str_to_adjacency, file_region_type

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--link_str_file', default='link_str_corr_alm_60_lag_0_2022_03.csv')
    parser.add_argument('--data_dir', default='data/precipitation')
    parser.add_argument('--output_folder', default=None)
    parser.add_argument('--edge_density', '--ed', type=float, default=0.005)
    parser.add_argument('--link_str_threshold', type=float, default=None)
    parser.add_argument('--num_comm_min', '--min', type=int, default=3)
    parser.add_argument('--num_comm_max', '--max', type=int, default=6)
    args = parser.parse_args()
    label_size, font_size, show_or_save = configure_plots(args)
    
    map_region = file_region_type(args.link_str_file)
    link_str_df = read_link_str_df(f'{args.data_dir}/{args.link_str_file}')
    try:
        if 'decadal' in args.link_str_file:
            date = datetime(2011, 4, 1) if 'd1' in args.link_str_file \
                else datetime(2022, 3, 1)
            year = date.year
            month = date.month
        else:
            date_part = args.link_str_file.split('lag_')[1].split('.csv')[0]
            _, year, month = date_part.split('_')
    except IndexError:
        # Month-only link strength files
        year = int(args.link_str_file.split('_')[-1][:4])
        month = int(args.link_str_file.split('_')[-2][-2:])
    dt = datetime(int(year), int(month), 1)
    adjacency = link_str_to_adjacency(link_str_df, args.edge_density,
        args.link_str_threshold)
    graph = nx.from_numpy_array(adjacency.values)
    graph = nx.relabel_nodes(graph, dict(enumerate(adjacency.columns)))
    lcc_graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()

    # Girvan-Newman is very slow
    # girvan_newman_partitions = [p for p in next(community.girvan_newman(lcc_graph))]
    greedy_mod_partitions = [p for p in community.greedy_modularity_communities(lcc_graph,
        cutoff=args.num_comm_min, best_n=args.num_comm_max)]
    # NOTE: below algorithms are random, hence why a seed is specified
    fluid_partitions_all = {}
    fluid_partitions_modularity = {}
    for n in range(args.num_comm_min, args.num_comm_max + 1):
        fluid_partitions_all[n] = [p for p in community.asyn_fluidc(lcc_graph, n, seed=0)]
        fluid_partitions_modularity[n] = community.modularity(lcc_graph,
            communities=fluid_partitions_all[n])
    fluid_partitions = fluid_partitions_all[max(fluid_partitions_modularity,
        key=fluid_partitions_modularity.get)]
    label_prop_partitions = [p for p in community.asyn_lpa_communities(lcc_graph, seed=0)]
    louvain_mod_partitions = [p for p in community.louvain_communities(lcc_graph, seed=0)]

    # Calculate colours of each node based on community
    colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 'black', '#444']
    lv_node_colours = []
    gm_node_colours = []
    fl_node_colours = []
    lp_node_colours = []
    # gn_node_colours = []
    for n in lcc_graph.nodes:
        for i, p in enumerate(greedy_mod_partitions):
            if n in p:
                gm_node_colours.append(colours[i % len(colours)])
                break
        for i, p in enumerate(label_prop_partitions):
            if n in p:
                lp_node_colours.append(colours[i % len(colours)])
                break
        for i, p in enumerate(louvain_mod_partitions):
            if n in p:
                lv_node_colours.append(colours[i % len(colours)])
                break
        for i, p in enumerate(fluid_partitions):
            if n in p:
                fl_node_colours.append(colours[i % len(colours)])
                break
        # for i, p in enumerate(girvan_newman_partitions):
        #     if n in p:
        #         gn_node_colours.append(colours[i % len(colours)])
        #         break

    figure, axes = plt.subplots(2, 2, layout='compressed')
    axes = axes.flatten()
    axes[0].set_title(f'{dt.strftime("%b %Y")}: greedy_mod_partitions, {len(greedy_mod_partitions)} communities')
    axes[1].set_title(f'{dt.strftime("%b %Y")}: fluid_partitions, {len(fluid_partitions)} communities')
    axes[2].set_title(f'{dt.strftime("%b %Y")}: louvain_mod_partitions, {len(louvain_mod_partitions)} communities')
    axes[3].set_title(f'{dt.strftime("%b %Y")}: label_prop_partitions, {len(label_prop_partitions)} communities')
    # axes[4].set_title(f'{dt.strftime("%b %Y")}: girvan_newman_partitions, {len(girvan_newman_partitions)} communities')
    gm_map = get_map(axes[0], region=map_region)
    fl_map = get_map(axes[1], region=map_region)
    lv_map = get_map(axes[2], region=map_region)
    lp_map = get_map(axes[3], region=map_region)
    # gn_map = get_map(axes[4], region=map_region)
    lats, lons = zip(*adjacency.columns)
    map_x, map_y = lv_map(lons, lats)
    pos = {}
    for i, elem in enumerate(adjacency.index):
        pos[elem] = (map_x[i], map_y[i])
    for (axis, colours) in zip(axes, [gm_node_colours, fl_node_colours,
            lv_node_colours, lp_node_colours]):
        nx.draw_networkx_nodes(G=lcc_graph, pos=pos, nodelist=lcc_graph.nodes(),
            node_color=colours, alpha=0.8, ax=axis,
            node_size=400 if 'geo_agg' in args.link_str_file else 80)
        nx.draw_networkx_edges(G=lcc_graph, pos=pos, edge_color='gray',
            alpha=0.2, arrows=False, ax=axis,
            width=2 if 'geo_agg' in args.link_str_file else 1)
    filename = (f'communities_ed_{str(args.edge_density).replace(".", "p")}'
        f'_{args.link_str_file.split("link_str_")[1].split(".csv")[0]}.png')
    show_or_save(figure, filename)

if __name__ == '__main__':
    main()