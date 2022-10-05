import pickle
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DROP_PCT = 50
METRICS_DIR = 'data/outputs'
# TIME_METRICS_DATA_FILE = f'{METRICS_DIR}/time_metrics_drop_90_thr_2p8.pkl'
# TIME_METRICS_DATA_FILE = f'{METRICS_DIR}/time_metrics_drop_90_ed_0p005.pkl'
TIME_METRICS_DATA_FILE = f'{METRICS_DIR}/time_metrics_drop_50_ed_0p025.pkl'
AGG_TYPE = None
AGG_CUTOFF = 5
EMA_HALF_LIFE = 2

# metrics = ['average_degree', 'coreness', 'transitivity', 'shortest_path', 'eccentricity',
#     'eigenvector_centrality', 'betweenness_centrality', 'closeness_centrality',
#     'louvain_partitions', 'greedy_modularity_partitions', 'asyn_lpa_partitions',
#     'louvain_modularity', 'greedy_modularity_modularity', 'asyn_lpa_modularity',
#     'average_link_strength']

metrics = [
    'shortest_path', 'eigenvector_centrality',
    'betweenness_centrality', 'closeness_centrality',
    'louvain_partitions', 'louvain_modularity',
    'greedy_modularity_partitions', 'greedy_modularity_modularity',
    'asyn_lpa_partitions', 'asyn_lpa_modularity'
]

def main():
    with open(TIME_METRICS_DATA_FILE, 'rb') as f:
        time_metrics_list = pickle.load(f)
    df = pd.DataFrame.from_dict([{**m['graph_metrics'], **m['link_metrics'], 'dt': m['dt']} \
        for m in time_metrics_list]).set_index('dt')

    if AGG_TYPE == 'simple_ma':
        for m in metrics:
            if AGG_CUTOFF:
                df[f'{m}_agg'] = df[m].rolling(AGG_CUTOFF).mean()
            else:
                # Cumulative average
                df[f'{m}_agg'] = df[m].expanding().mean()
    elif AGG_TYPE == 'linear_ma':
        weights = np.arange(1, AGG_CUTOFF + 1)
        for m in metrics:
            df[f'{m}_agg'] = df[m].rolling(AGG_CUTOFF).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True).to_list()
    elif AGG_TYPE == 'exp_ma':
        for m in metrics:
            df[f'{m}_agg'] = df[m].ewm(halflife=EMA_HALF_LIFE).mean()
    else:
        # Just plot the value of the metric at each individual timestamp
        for m in metrics:
            df[f'{m}_agg'] = df[m]

    # Construct figures for time series metrics
    figure, axes = plt.subplots(math.ceil(len(metrics) / 2), 2)
    axes = axes.flatten()
    graph_times =  [m['dt'] for m in time_metrics_list]
    colours = ['grey', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#bcbd22', '#17becf', 'black']
    for i, m in enumerate(metrics):
        colour = colours[i % len(colours)]
        axes[i].plot(graph_times, df[f'{m}_agg'],'-', color=colour)
        axes[i].set_title(m)
    figure.set_size_inches(32, 18)
    filename = f'images/graph_plots_drop_{DROP_PCT}.png'
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

main()