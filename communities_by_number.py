import argparse
from datetime import datetime
from math import ceil

import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community

from helpers import read_link_str_df, configure_plots, get_map, \
    link_str_to_adjacency, file_region_type

COLOURS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 'black', '#444']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--link_str_file', default='link_str_corr_alm_60_lag_0_2022_03.csv')
    parser.add_argument('--data_dir', default='data/precipitation')
    parser.add_argument('--output_folder', default=None)
    parser.add_argument('--edge_density', '--ed', type=float, default=0.005)
    parser.add_argument('--link_str_threshold', type=float, default=None)
    parser.add_argument('--num_comm_min', type=int, default=3)
    parser.add_argument('--num_comm_max', type=int, default=10)
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
    lats, lons = zip(*adjacency.columns)
    graph = nx.from_numpy_array(adjacency.values)
    graph = nx.relabel_nodes(graph, dict(enumerate(adjacency.columns)))
    lcc_graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()

    comm_range = range(args.num_comm_min, args.num_comm_max + 1)
    greedy_mod_partitions = {}
    greedy_mod_modularity = {}
    fluid_partitions = {}
    fluid_modularity = {}
    greedy_mod_colours = {n: [] for n in comm_range}
    fluid_colours = {n: [] for n in comm_range}
    for n in comm_range:
        fluid_partitions[n] = [p for p in community.asyn_fluidc(lcc_graph, n, seed=0)]
        fluid_modularity[n] = community.modularity(lcc_graph,
            communities=fluid_partitions[n])
        greedy_mod_partitions[n] = [p for p in community.greedy_modularity_communities(lcc_graph,
            cutoff=n, best_n=n)]
        greedy_mod_modularity[n] = community.modularity(lcc_graph,
            communities=greedy_mod_partitions[n])
        for node in lcc_graph.nodes:
            for i, p in enumerate(greedy_mod_partitions[n]):
                if node in p:
                    greedy_mod_colours[n].append(COLOURS[i % len(COLOURS)])
                    break
            for i, p in enumerate(fluid_partitions[n]):
                if node in p:
                    fluid_colours[n].append(COLOURS[i % len(COLOURS)])
                    break

    for method, node_colours, modularities in [
        ('greedy_mod', greedy_mod_colours, greedy_mod_modularity),
        ('fluid', fluid_colours, fluid_modularity),
    ]:
        figure, axes = plt.subplots(2, ceil(len(comm_range) / 2), layout='compressed')
        axes = iter(axes.flatten())
        for n in comm_range:
            axis = next(axes)
            _map = get_map(axis, region=map_region)
            map_x, map_y = _map(lons, lats)
            pos = {}
            for j, elem in enumerate(adjacency.index):
                pos[elem] = (map_x[j], map_y[j])
            axis.set_title(f'{dt.strftime("%b %Y")}: {method} {n}, '
                f'mod = {round(modularities[n], 3)}')
            nx.draw_networkx_nodes(G=lcc_graph, pos=pos, nodelist=lcc_graph.nodes(),
                node_color=node_colours[n], alpha=0.8, ax=axis, node_size=80)
            nx.draw_networkx_edges(G=lcc_graph, pos=pos, edge_color='gray',
                alpha=0.2, arrows=False, ax=axis, width=1)
        filename = (f'communities_{comm_range[0]}_{comm_range[-1]}'
            f'_{method}_ed_{str(args.edge_density).replace(".", "p")}'
            f'_{args.link_str_file.split("link_str_")[1].split(".csv")[0]}.png')
        show_or_save(figure, filename)

if __name__ == '__main__':
    main()