import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from helpers import configure_plots

METRICS_DIR = 'data/metrics'

def aggregate_series(series, agg_type, agg_cutoff, ema_half_life):
    if agg_type == 'simple_ma':
        if agg_cutoff:
            agg_series = series.rolling(agg_cutoff).mean()
        else:
            # Cumulative average
            agg_series = series.expanding().mean()
    elif agg_type == 'linear_ma':
        weights = np.arange(1, agg_cutoff + 1)
        agg_series = series.rolling(agg_cutoff).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True).to_list()
    elif agg_type == 'exp_ma':
        agg_series = series.ewm(halflife=ema_half_life).mean()
    else:
        # Just return the value of the metric at each individual timestamp
        agg_series = series
    return agg_series

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics_file_base', default='metrics_corr_alm_60_lag_0_ed_0p005')
    parser.add_argument('--output_folder', default=None)
    args = parser.parse_args()
    label_size, font_size, show_or_save = configure_plots(args)

    metrics_df = pd.read_pickle(f'{METRICS_DIR}/{args.metrics_file_base}_whole.pkl')
    fig_num = 0
    for i, metric in enumerate(sorted(metrics_df.columns)):
        if i % 4 == 0:
            figure, axes = plt.subplots(2, 2, layout='compressed')
            axes = iter(axes.flatten())
            fig_num += 1
        axis = next(axes)
        axis.plot(metrics_df.index.values, metrics_df[metric], '-')
        axis.set_title(metric)
        if i % 4 == 3:
            show_or_save(figure, f'{args.metrics_file_base}_series_{fig_num}.png')
    show_or_save(figure, f'{args.metrics_file_base}_series_{fig_num}.png')

    figure, axes = plt.subplots(4, 5, layout='compressed')
    axes = iter(axes.flatten())
    for i, metric in enumerate(sorted(metrics_df.columns)):
        axis = next(axes)
        axis.plot(metrics_df.index.values, metrics_df[metric], '-')
        axis.set_title(metric)
    show_or_save(figure, f'{args.metrics_file_base}_series_all.png')

if __name__ == '__main__':
    main()