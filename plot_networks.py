from datetime import datetime
from dateutil.relativedelta import relativedelta
import argparse

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from helpers import get_map, configure_plots, read_link_str_df, link_str_to_adjacency

YEARS = list(range(2000, 2022 + 1))
MONTHS = list(range(1, 13))
OUTPUTS_DIR = 'data/outputs'
LOCATIONS_FILE = f'data/precipitation/Fused.Locations.csv'

def network_map(axis, link_str_df, edge_density=None, threshold=None,
        plot_world=False, lag_bool_df=None):
    adjacency = link_str_to_adjacency(link_str_df, edge_density,
        threshold, lag_bool_df)
    _map = get_map(axis, aus=not plot_world)
    lats, lons = zip(*adjacency.columns)
    map_x, map_y = _map(lons, lats)
    graph = nx.from_numpy_array(adjacency.values)
    graph = nx.relabel_nodes(graph, dict(enumerate(adjacency.columns)))
    pos = {}
    for i, elem in enumerate(adjacency.index):
        pos[elem] = (map_x[i], map_y[i])
    node_sizes = [adjacency[location].sum() / 5 for location in graph.nodes()]
    # node_sizes = [3 for _ in graph.nodes()]
    nx.draw_networkx_nodes(ax=axis, G=graph, pos=pos, nodelist=graph.nodes(),
        node_color='r', alpha=0.8, node_size=node_sizes)
    nx.draw_networkx_edges(ax=axis, G=graph, pos=pos, edge_color='g',
        alpha=0.2, arrows=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', default=None)
    parser.add_argument('--edge_density', type=float, default=0.005)
    parser.add_argument('--link_str_threshold', type=float, default=None)
    parser.add_argument('--start_year', type=int, default=2021)
    parser.add_argument('--start_month', type=int, default=4)
    parser.add_argument('--month', type=int, default=None)
    parser.add_argument('--data_dir', default='data/precipitation')
    parser.add_argument('--fmt', default='csv')
    parser.add_argument('--plot_world', action='store_true', default=False)
    parser.add_argument('--last_dt', action='store_true', default=False)
    parser.add_argument('--link_str_file_tag', default='corr_alm_60_lag_0')
    args = parser.parse_args()
    label_size, font_size, show_or_save = configure_plots(args)

    graph_file_tag = f'ed_{str(args.edge_density).replace(".", "p")}' \
        if args.edge_density \
        else f'thr_{str(args.link_str_threshold).replace(".", "p")}'
    network_map_kw = {
        'plot_world': args.plot_world,
        'edge_density': args.edge_density,
        'threshold': args.link_str_threshold,
    }
    if 'decadal' in args.link_str_file_tag:
        figure, axes = plt.subplots(2, 1, layout='compressed')
        axes = iter(axes.flatten())

        links_file = f'{args.data_dir}/link_str_{args.link_str_file_tag}_d1'
        link_str_df = read_link_str_df(f'{links_file}.{args.fmt}')
        axis = next(axes)
        network_map(axis, link_str_df, **network_map_kw)
        axis.set_title('Decade 1')

        links_file = f'{args.data_dir}/link_str_{args.link_str_file_tag}_d2'
        link_str_df = read_link_str_df(f'{links_file}.{args.fmt}')
        axis = next(axes)
        network_map(axis, link_str_df, **network_map_kw)
        axis.set_title('Decade 2')

        figure_title = f'networks_{args.link_str_file_tag}_{graph_file_tag}.png'
        show_or_save(figure, figure_title)
    elif args.last_dt:
        figure, axis = plt.subplots(1, layout='compressed')
        dt = datetime(2022, 3, 1)
        links_file = (f'{args.data_dir}/link_str_{args.link_str_file_tag}'
            f'_{dt.strftime("%Y_%m")}')
        link_str_df = read_link_str_df(f'{links_file}.{args.fmt}')
        network_map(axis, link_str_df, **network_map_kw)
        axis.set_title(dt.strftime('%Y %b'))
        figure_title = (f'networks_{args.link_str_file_tag}_{graph_file_tag}'
            f'_{dt.strftime("%Y_%m")}.png')
        print(figure_title)
        show_or_save(figure, figure_title)
    else:
        figure, axes = plt.subplots(3, 4, layout='compressed')
        axes = iter(axes.flatten())
        start_dt = datetime(args.start_year, args.month or args.start_month, 1)
        end_dt = start_dt + relativedelta(years=11) if args.month else \
            start_dt + relativedelta(months=11)
        for y in YEARS:
            if args.month:
                dt = datetime(y, args.month, 1)
                if dt < start_dt or dt > end_dt:
                    continue
                links_file = (f'{args.data_dir}/link_str_{args.link_str_file_tag}'
                    f'_m{dt.strftime("%m_%Y")}')
                link_str_df = read_link_str_df(f'{links_file}.{args.fmt}')
                axis = next(axes)
                network_map(axis, link_str_df, **network_map_kw)
                axis.set_title(dt.strftime('%Y %b'))
            else:
                for m in MONTHS:
                    dt = datetime(y, m, 1)
                    if dt < start_dt or dt > end_dt:
                        continue
                    links_file = (f'{args.data_dir}/link_str_{args.link_str_file_tag}'
                        f'_{dt.strftime("%Y_%m")}')
                    link_str_df = read_link_str_df(f'{links_file}.{args.fmt}')
                    axis = next(axes)
                    network_map(axis, link_str_df, **network_map_kw)
            figure_title = f'networks_{args.link_str_file_tag}_{graph_file_tag}'
            figure_title += f'_m{start_dt.strftime("%m_%Y")}_{end_dt.strftime("%Y")}.png' \
                if args.month else \
                f'_{start_dt.strftime("%Y_%m")}_{end_dt.strftime("%Y_%m")}.png'
        show_or_save(figure, figure_title)

if __name__ == '__main__':
    main()