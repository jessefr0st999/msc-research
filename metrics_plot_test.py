import argparse
from datetime import datetime

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from helpers import get_map

DATA_DIR = 'data/precipitation'

metric_names = [
    'betweenness_centrality',
    'closeness_centrality',
    'eigenvector_centrality',
]
cmap = 'RdYlGn_r'

def size_func(series):
    series_norm = series / np.max(series)
    return [50 * n for n in series_norm]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prec_file', default='dataframe_alm_60_lag_0.pkl')
    parser.add_argument('--link_str_file', default='link_str_alm_60_lag_0_2006_01.pkl')
    parser.add_argument('--edge_density', type=float, default=0.005)
    parser.add_argument('--output_folder', default=None)
    args = parser.parse_args()

    prec_df = pd.read_pickle(f'{DATA_DIR}/{args.prec_file}')
    link_str_df = pd.read_pickle(f'{DATA_DIR}/{args.link_str_file}')
    try:
        date_part = args.link_str_file.split('lag_')[1].split('.pkl')[0]
        _, year, month = date_part.split('_')
    except IndexError:
        # Month-only link strength files
        year = int(args.link_str_file.split('_')[-1][:4])
        month = int(args.link_str_file.split('_')[-2][-2:])
    dt = datetime(int(year), int(month), 1)
    dt_prec_df = prec_df.loc[dt]
    location_prec_df = dt_prec_df.set_index(['lat', 'lon'])
    threshold = np.quantile(link_str_df, 1 - args.edge_density)
    print(f'Fixed edge density {args.edge_density} gives threshold {threshold}')
    adjacency = pd.DataFrame(0, columns=link_str_df.columns, index=link_str_df.index)
    adjacency[link_str_df >= threshold] = 1
    graph = nx.from_numpy_matrix(adjacency.values)
    graph = nx.relabel_nodes(graph, dict(enumerate(adjacency.columns)))

    figure, axes = plt.subplots(1, 3)
    axes = axes.flatten()
    lv_map = get_map(axes[0])
    gm_map = get_map(axes[1])
    af_map = get_map(axes[2])
    mx, my = lv_map(dt_prec_df['lon'], dt_prec_df['lat'])
    pos = {}
    for i, elem in enumerate(adjacency.index):
        pos[elem] = (mx[i], my[i])

    bc = [v for v in nx.betweenness_centrality(graph).values()]
    axes[0].set_title(f'{dt.strftime("%b")} {year}: betweenness_centrality', fontsize=20)
    axes[0].scatter(mx, my, c=bc, cmap=cmap,
        s=size_func(bc))

    cc = [v for v in nx.closeness_centrality(graph).values()]
    axes[1].set_title(f'{dt.strftime("%b")} {year}: closeness_centrality', fontsize=20)
    axes[1].scatter(mx, my, c=cc, cmap=cmap,
        s=size_func(cc))

    axes[2].set_title(f'{dt.strftime("%b")} {year}: eigenvector_centrality', fontsize=20)
    try:
        ec = [v for v in nx.eigenvector_centrality(graph).values()]
        axes[2].scatter(mx, my, c=ec, cmap=cmap,
            s=size_func(ec))
    except nx.exception.PowerIterationFailedConvergence:
        pass

    plt.tight_layout()
    figure.set_size_inches(32, 18)
    if args.output_folder:
        filename = (f'{args.output_folder}/communities_ed_{str(args.edge_density).replace(".", "p")}'
            f'_{args.link_str_file.split("link_str_")[1]}.png')
        plt.savefig(filename, bbox_inches='tight')
        print(f'Saved to file {filename}')
    else:
        plt.show()

main()