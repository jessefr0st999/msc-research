import argparse
from datetime import datetime

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from helpers import get_map

DATA_DIR = 'data/precipitation'
OUTPUTS_DIR = 'data/outputs'
LOCATIONS_FILE = f'{DATA_DIR}/Fused.Locations.csv'

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
    parser.add_argument('--event_sync_file', default='event_sync_colq_0p95_dfq_0p95_tau_12_2022_03_lm_120.csv')
    parser.add_argument('--edge_density', type=float, default=0.005)
    parser.add_argument('--output_folder', default=None)
    args = parser.parse_args()

    event_sync_df = pd.read_csv(f'{OUTPUTS_DIR}/{args.event_sync_file}',
        index_col=[0, 1], header=[0, 1])
    sym_event_array = np.array(event_sync_df) + np.array(event_sync_df).T
    sym_event_df = pd.DataFrame(sym_event_array, columns=event_sync_df.columns,
        index=event_sync_df.columns).fillna(0)
    threshold = np.quantile(sym_event_df, 1 - args.edge_density)
    print(f'Fixed edge density {args.edge_density} gives threshold {threshold}')
    adjacency = pd.DataFrame(0, columns=event_sync_df.columns,
        index=event_sync_df.columns)
    adjacency[sym_event_df > threshold] = 1
    graph = nx.from_numpy_matrix(adjacency.values)
    graph = nx.relabel_nodes(graph, dict(enumerate(adjacency.columns)))

    figure, axes = plt.subplots(1, 3)
    axes = axes.flatten()
    lv_map = get_map(axes[0])
    gm_map = get_map(axes[1])
    af_map = get_map(axes[2])
    locations_df = pd.read_csv(LOCATIONS_FILE)
    mx, my = lv_map(locations_df['Lon'], locations_df['Lat'])
    pos = {}
    for i, elem in enumerate(adjacency.index):
        pos[elem] = (mx[i], my[i])

    bc = [v for v in nx.betweenness_centrality(graph).values()]
    axes[0].set_title(f'betweenness_centrality', fontsize=20)
    axes[0].scatter(mx, my, c=bc, cmap=cmap,
        s=size_func(bc))

    cc = [v for v in nx.closeness_centrality(graph).values()]
    axes[1].set_title(f'closeness_centrality', fontsize=20)
    axes[1].scatter(mx, my, c=cc, cmap=cmap,
        s=size_func(cc))

    axes[2].set_title(f'eigenvector_centrality', fontsize=20)
    try:
        ec = [v for v in nx.eigenvector_centrality(graph).values()]
        axes[2].scatter(mx, my, c=ec, cmap=cmap,
            s=size_func(ec))
    except nx.exception.PowerIterationFailedConvergence:
        pass

    plt.tight_layout()
    if args.output_folder:
        figure.set_size_inches(32, 18)
        filename = (f'{args.output_folder}/communities_ed_{str(args.edge_density).replace(".", "p")}'
            f'_{args.link_str_file.split("link_str_")[1]}.png')
        plt.savefig(filename, bbox_inches='tight')
        print(f'Saved to file {filename}')
    else:
        plt.show()

main()