import argparse
from math import ceil

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from helpers import configure_plots, read_link_str_df, file_region_type
from plot_networks import network_map

YEARS = list(range(2000, 2022 + 1))
MONTHS = list(range(1, 13))
OUTPUTS_DIR = 'data/outputs'
LOCATIONS_FILE = f'data/precipitation/Fused.Locations.csv'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', default=None)
    parser.add_argument('--edge_density', '--ed', type=float, default=0.005)
    parser.add_argument('--link_str_threshold', type=float, default=None)
    parser.add_argument('--data_dir', default='data/precipitation')
    parser.add_argument('--link_str_file', default='link_str_corr_alm_60_lag_6_2022_03')
    parser.add_argument('--fmt', default='csv')
    args = parser.parse_args()
    label_size, font_size, show_or_save = configure_plots(args)

    link_str_df = read_link_str_df(f'{args.data_dir}/{args.link_str_file}.{args.fmt}')
    max_lag_df = read_link_str_df(f'{args.data_dir}/{args.link_str_file}_max_lags.pkl')
    lag_span = range(max_lag_df.min().min(), max_lag_df.max().max() + 1)

    link_str_file_tag = args.link_str_file.split('link_str_')[1]
    network_map_kw = {
        'map_region': file_region_type(link_str_file_tag),
        'edge_density': args.edge_density,
        'threshold': args.link_str_threshold,
    }
    figure, axes = plt.subplots(4, ceil(len(lag_span) / 4), layout='compressed')
    axes = iter(axes.flatten())
    for lag in lag_span:
        axis = next(axes)
        network_map_kw['lag_bool_df'] = max_lag_df.applymap(
            lambda x: 1 if x == lag else 0)
        network_map(axis, link_str_df, **network_map_kw)
        axis.set_title(f'lag = {lag}')

    graph_file_tag = f'ed_{str(args.edge_density).replace(".", "p")}' \
        if args.edge_density \
        else f'thr_{str(args.link_str_threshold).replace(".", "p")}'
    figure_title = f'networks_by_max_lag_{link_str_file_tag}_{graph_file_tag}.png'
    show_or_save(figure, figure_title)

if __name__ == '__main__':
    main()