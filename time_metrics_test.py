import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DROP_PCT = 50
LINK_STR_THRESHOLD = 2.8
OUTPUTS_DIR = 'data/outputs'
TIME_METRICS_DATA_FILE = f'{OUTPUTS_DIR}/time_metrics_drop_{DROP_PCT}_thr_' + \
    f'{str(LINK_STR_THRESHOLD).replace(".", "p")}.pkl'
AGG_TYPE = 'raw'
AGG_CUTOFF = 15
EMA_HALF_LIFE = 3

metrics = ['average_degree', 'coreness', 'modularity', 'transitivity',
    'average_link_strength', 'eigenvector_centrality', 'shortest_path', 'eccentricity']

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
    figure, axes = plt.subplots(4, 2)
    graph_times =  [m['dt'] for m in time_metrics_list]
    axes[0, 0].set_title('Average degree')
    axes[0, 0].plot(graph_times, df['average_degree_agg'], '-b')
    axes[1, 0].set_title('Coreness')
    axes[1, 0].plot(graph_times, df['coreness_agg'], '-g')
    axes[2, 0].set_title('Modularity')
    axes[2, 0].plot(graph_times, df['modularity_agg'], '-r')
    axes[3, 0].set_title('Transitivity')
    axes[3, 0].plot(graph_times, df['transitivity_agg'], '-m')
    axes[0, 1].set_title('Link strength')
    axes[0, 1].plot(graph_times, df['average_link_strength_agg'], '-k')
    axes[1, 1].set_title('Eigenvector centrality')
    axes[1, 1].plot(graph_times, df['eigenvector_centrality_agg'], '-y')
    axes[2, 1].set_title('Shortest path')
    axes[2, 1].plot(graph_times, df['shortest_path_agg'], '-', color='tab:orange')
    axes[3, 1].set_title('Eccentricity')
    axes[3, 1].plot(graph_times, df['eccentricity_agg'], '-', color='tab:cyan')
    plt.savefig(f'{OUTPUTS_DIR}/graph_plots_drop_{DROP_PCT}.png')
    plt.show()

main()