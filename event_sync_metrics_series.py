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

metric_names = [
    'betweenness_centrality',
    'closeness_centrality',
    'eigenvector_centrality',
]
cmap = 'RdYlGn_r'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--series_p1', default='event_sync_colq_0p95_dfq_0p95_tau_12')
    parser.add_argument('--series_p2', default='lm_120')
    parser.add_argument('--start_year', type=int, default=2010)
    parser.add_argument('--end_year', type=int, default=2022)
    parser.add_argument('--edge_density', type=float, default=0.005)
    parser.add_argument('--output_folder', default=None)
    args = parser.parse_args()

    months = range(1, 12 + 1)
    years = range(args.start_year, args.end_year + 1)
    ad_vec, bc_vec, cc_vec, ec_vec, dt_vec = [], [], [], [], []
    for y in years:
        for m in months:
            month_str = str(m) if m >= 10 else f'0{m}'
            filename = f'{args.series_p1}_{y}_{month_str}_{args.series_p2}.csv'
            try:
                event_sync_df = pd.read_csv(f'{OUTPUTS_DIR}/{filename}',
                    index_col=[0, 1], header=[0, 1])
            except FileNotFoundError:
                continue
            sym_event_array = np.array(event_sync_df) + np.array(event_sync_df).T
            sym_event_df = pd.DataFrame(sym_event_array, columns=event_sync_df.columns,
                index=event_sync_df.columns).fillna(0)
            threshold = np.quantile(sym_event_df, 1 - args.edge_density)
            print(f'{y} {month_str}: Fixed edge density {args.edge_density} gives threshold {threshold}')
            adjacency = pd.DataFrame(0, columns=event_sync_df.columns,
                index=event_sync_df.columns)
            adjacency[sym_event_df > threshold] = 1
            graph = nx.from_numpy_matrix(adjacency.values)
            ad = 2 * len(graph.edges) / len(graph.nodes)
            bc = np.mean([v for v in nx.betweenness_centrality(graph).values()])
            cc = np.mean([v for v in nx.closeness_centrality(graph).values()])
            try:
                ec_values = [v for v in nx.eigenvector_centrality(graph).values()]
                ec = np.mean(ec_values)
            except nx.exception.PowerIterationFailedConvergence:
                ec = 0
            ad_vec.append(ad)
            bc_vec.append(bc)
            cc_vec.append(cc)
            ec_vec.append(ec)
            dt_vec.append(datetime(y, m, 1))

    figure, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    axes[0].plot(dt_vec, bc_vec, '-')
    axes[0].set_title(f'betweenness_centrality', fontsize=20)
    axes[1].plot(dt_vec, cc_vec, '-')
    axes[1].set_title(f'closeness_centrality', fontsize=20)
    axes[2].plot(dt_vec, ec_vec, '-')
    axes[2].set_title(f'eigenvector_centrality', fontsize=20)
    axes[3].plot(dt_vec, ad_vec, '-')
    axes[3].set_title(f'average_degree', fontsize=20)
    plt.tight_layout()
    if args.output_folder:
        figure.set_size_inches(32, 18)
        filename = (f'{args.output_folder}/metric_series_{args.series_p1}_{args.series_p2}'
            f'_ed_{str(args.edge_density).replace(".", "p")}.png')
        plt.savefig(filename, bbox_inches='tight')
        print(f'Saved to file {filename}')
    else:
        plt.show()

main()