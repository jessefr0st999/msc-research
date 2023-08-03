from dateutil.relativedelta import relativedelta
import argparse
from math import floor

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
np.random.seed(0)
import pymannkendall as mk

from helpers import get_map, scatter_map, prepare_df, configure_plots

FUSED_DAILY_PATH = 'data/fused_upsampled'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=None)
    args = parser.parse_args()
    
    loc_list = []
    prec_series_lists = []
    # prec_series_lists receives data sequentially by location
    # Select random locations from the fused dataset
    prec_df, _, _ = prepare_df('data/precipitation', 'FusedData.csv', 'prec')
    num_samples = args.num_samples if args.num_samples else 1391
    _prec_df = prec_df.T.sample(num_samples)
    for loc, row in _prec_df.iterrows():
        loc_list.append(loc)
        def yearly_average(months=None):
            if months:
                filtered = row.loc[[dt.month in months for dt in row.index]]
                # For summer, consider December as part of the next year
                if months[0] == 12:
                    filtered.index = [pd.to_datetime(dt) + relativedelta(months=1) \
                        for dt in filtered.index.values]
            else:
                filtered = row
            return filtered.groupby(pd.Grouper(freq='1Y')).mean()
        prec_series_lists.append((
            yearly_average(),
            yearly_average([12, 1, 2]),
            yearly_average([3, 4, 5]),
            yearly_average([6, 7, 8]),
            yearly_average([9, 10, 11]),
        ))

    trend_list_keys = ['yearly', 'summer', 'autumn', 'winter', 'spring']
    trend_lists = {key: [] for key in trend_list_keys}
    for i, series_list in enumerate(prec_series_lists):
        if i % 100 == 0:
            print(i)
        for key, series in zip(trend_list_keys, series_list):
            test_result = mk.original_test(series)
            trend_pos = 1 if test_result.h and test_result.s > 0 else 0
            trend_neg = 1 if test_result.h and test_result.s < 0 else 0
            trend_lists[key].append([
                test_result.p,
                test_result.s,
                test_result.s if test_result.h else None,
                trend_pos,
                trend_neg,
            ])
    trend_dfs = {key: pd.DataFrame(trend_lists[key], index=loc_list,
        columns=['p-value', 'score', 'score (trend only)', 'trend_pos', 'trend_neg']) \
            for key in trend_list_keys}

    lats, lons = list(zip(*loc_list))
    figure, axes = plt.subplots(3, 5, layout='compressed')
    axes = iter(axes.T.flatten())
    for key in trend_list_keys:
        trend_df = trend_dfs[key]
        print(f'{key}: positive trend at {trend_df["trend_pos"].sum()} / 1391 locations')
        print(f'{key}: negative trend at {trend_df["trend_neg"].sum()} / 1391 locations')
        for series_name in trend_df.columns[:3]:
            series = trend_df[series_name]
            axis = next(axes)
            _map = get_map(axis)
            mx, my = _map(lons, lats)
            if series_name == 'p-value':
                _min = series.min()
                _max = series.max()
                cmap = 'inferno_r'
            else:
                _min = -np.max(np.abs(series))
                _max = np.max(np.abs(series))
                cmap = 'Spectral_r'
            scatter_map(axis, mx, my, series, size_func=lambda x: 10,
                cb_min= _min, cb_max=_max, cmap=cmap)
            axis.set_title(f'{key} {series_name}')
    plt.show()

    figure, axes = plt.subplots(2, 3, layout='compressed')
    axes = iter(axes.T.flatten())
    for key in trend_list_keys:
        axis = next(axes)
        _map = get_map(axis)
        series = trend_dfs[key]['score']
        scatter_map(axis, mx, my, series, size_func=lambda x: 5,
            show_cb=False, cmap='Spectral_r')
        series = trend_dfs[key]['score (trend only)']
        scatter_map(axis, mx, my, series, size_func=lambda x: 30,
            cb_min=-np.max(np.abs(series)), cb_max=np.max(np.abs(series)),
            cmap='Spectral_r')
        axis.set_title(f'{key} score')
        if key == 'yearly':
            next(axes).axis('off')
    plt.show()
    
if __name__ == '__main__':
    main()