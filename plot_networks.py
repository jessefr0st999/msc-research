from datetime import datetime
from dateutil.relativedelta import relativedelta
import argparse

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from helpers import *

YEARS = list(range(2000, 2022 + 1))
MONTHS = list(range(1, 13))
DATA_DIR = 'data/precipitation'
OUTPUTS_DIR = 'data/outputs'
DATA_FILE = f'{DATA_DIR}/FusedData.csv'
LOCATIONS_FILE = f'{DATA_DIR}/Fused.Locations.csv'

def create_graph(adjacency: pd.DataFrame):
    graph = nx.from_numpy_array(adjacency.values)
    graph = nx.relabel_nodes(graph, dict(enumerate(adjacency.columns)))
    return graph

def get_map(axis=None):
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
    _map.drawcountries(linewidth=1)
    _map.drawstates(linewidth=0.2)
    _map.drawcoastlines(linewidth=1)
    return _map

def network_map(axis, map_x, map_y, adjacency: pd.DataFrame, title):
    graph = create_graph(adjacency)
    pos = {}
    for i, elem in enumerate(adjacency.index):
        pos[elem] = (map_x[i], map_y[i])
    node_sizes = [adjacency[location].sum() / 5 for location in graph.nodes()]
    nx.draw_networkx_nodes(ax=axis, G=graph, pos=pos, nodelist=graph.nodes(),
        node_color='r', alpha=0.8, node_size=node_sizes)
    nx.draw_networkx_edges(ax=axis, G=graph, pos=pos, edge_color='g',
        alpha=0.2, arrows=False)
    if title:
        axis.set_title(title)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', default=None)
    parser.add_argument('--edge_density', type=float, default=0.005)
    parser.add_argument('--link_str_threshold', type=float, default=None)
    parser.add_argument('--start_year', type=int, default=2021)
    parser.add_argument('--start_month', type=int, default=4)
    parser.add_argument('--link_str_file_tag', default='alm_60_lag_0')
    args = parser.parse_args()

    label_size = 20 if args.output_folder else 10
    font_size = 20 if args.output_folder else 10
    mpl.rcParams['xtick.labelsize'] = label_size
    mpl.rcParams['ytick.labelsize'] = label_size
    mpl.rcParams.update({'font.size': font_size})
    def show_or_save(figure, filename):
        if args.output_folder:
            figure.set_size_inches(32, 18)
            plt.savefig(f'{args.output_folder}/{filename}', bbox_inches='tight')
            print(f'Plot saved to file {args.output_folder}/{filename}!')
        else:
            plt.show()

    locations_df = pd.read_csv(LOCATIONS_FILE)
    lats = locations_df['Lat']
    lons = locations_df['Lon']

    start_dt = datetime(args.start_year, args.start_month, 1)
    end_dt = start_dt + relativedelta(months=11)
    figure, axes = plt.subplots(3, 4, layout='compressed')
    axes = iter(axes.flatten())
    for y in YEARS:
        for m in MONTHS:
            dt = datetime(y, m, 1)
            if dt < start_dt or dt > end_dt:
                continue
            date_summary = f'{dt.year}, {dt.strftime("%b")}'
            links_file = f'{DATA_DIR}/link_str_{args.link_str_file_tag}_{dt.strftime("%Y_%m")}.pkl'
            print(f'{date_summary}: reading link strength data from pickle file {links_file}')
            link_str_df: pd.DataFrame = pd.read_pickle(links_file)

            adjacency = pd.DataFrame(0, columns=link_str_df.columns, index=link_str_df.index)
            if args.edge_density:
                threshold = np.quantile(link_str_df, 1 - args.edge_density)
                print(f'{date_summary}: fixed edge density {args.edge_density} gives threshold {threshold}')
            else:
                threshold = args.link_str_threshold
            adjacency[link_str_df >= threshold] = 1
            if not args.edge_density:
                _edge_density = np.sum(np.sum(adjacency)) / adjacency.size
                print(f'{date_summary}: fixed threshold {args.link_str_threshold} gives edge density {_edge_density}')
           
            axis = next(axes)
            _map = get_map(axis)
            map_x, map_y = _map(lons, lats)
            network_map(axis, map_x, map_y, adjacency, date_summary)
    graph_file_tag = f'ed_{str(args.edge_density).replace(".", "p")}' if args.edge_density \
        else f'thr_{str(args.link_str_threshold).replace(".", "p")}'
    figure_title = (f'networks_{args.link_str_file_tag}_{graph_file_tag}'
        f'_{start_dt.strftime("%Y_%m")}_{end_dt.strftime("%Y_%m")}.png')
    show_or_save(figure, figure_title)

if __name__ == '__main__':
    main()