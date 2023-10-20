from dateutil.relativedelta import relativedelta
import argparse
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
from pymannkendall import original_test, seasonal_test, sens_slope
from scipy.stats import norm

from helpers import get_map, scatter_map, prepare_df

FUSED_DAILY_PATH = 'data/fused_upsampled'

def z_score(s, var_s):
    if s > 0:
        z = (s - 1)/np.sqrt(var_s)
    elif s == 0:
        z = 0
    elif s < 0:
        z = (s + 1)/np.sqrt(var_s)
    return z

def p_value(z, alpha):
    # Two-tailed test
    p = 2 * (1 - norm.cdf(abs(z)))  
    h = abs(z) > norm.ppf(1 - alpha / 2)
    return p, h

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=None)
    args = parser.parse_args()
    
    loc_list = []
    prec_series_lists = []
    # prec_series_lists receives data sequentially by location
    # Select random locations from the fused dataset
    prec_df, _, _ = prepare_df('data/precipitation', 'FusedData.csv', 'prec')
    for loc, row in prec_df.T.iterrows():
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
        prec_series_lists.append(tuple(yearly_average([m + 1]) for m in range(12)))
        # prec_series_lists.append((
        #     yearly_average(),
        #     yearly_average([12, 1, 2]),
        #     yearly_average([3, 4, 5]),
        #     yearly_average([6, 7, 8]),
        #     yearly_average([9, 10, 11]),
        # ))
    lats, lons = list(zip(*loc_list))

    # trend_list_keys = ['yearly', 'summer', 'autumn', 'winter', 'spring']
    trend_lists = {m: [] for m in range(1, 13)}
    full_trend_list = []
    for i, series_list in enumerate(prec_series_lists):
        if i % 25 == 0:
            print(i)
        full_trend_score = 0
        full_trend_var = 0
        for m, series in zip(range(1, 13), series_list):
            test_result = original_test(series)
            trend_lists[m].append([
                test_result.p,
                test_result.slope,
                test_result.slope if test_result.slope > 0 else None,
                test_result.slope if test_result.h and test_result.slope > 0 else None,
                test_result.slope if test_result.slope < 0 else None,
                test_result.slope if test_result.h and test_result.slope < 0 else None,
            ])
            full_trend_score += test_result.s
            full_trend_var += test_result.var_s
        # z = z_score(full_trend_score, full_trend_var)
        # p, h = p_value(z, 0.05)
        # slope, intercept = sens_slope(prec_df.iloc[:, i])
        # trend_pos = 1 if h and full_trend_score > 0 else 0
        # trend_neg = 1 if h and full_trend_score < 0 else 0
        # full_trend_list.append([
        #     p,
        #     slope,
        #     slope if h and full_trend_score > 0 else None,
        #     slope if h and full_trend_score < 0 else None,
        # ])
        # NOTE: equivalent to above
        test_result = seasonal_test(prec_df.iloc[:, i], period=12)
        full_trend_list.append([
            test_result.p,
            test_result.slope,
            test_result.slope if test_result.slope > 0 else None,
            test_result.slope if test_result.h and test_result.slope > 0 else None,
            test_result.slope if test_result.slope < 0 else None,
            test_result.slope if test_result.h and test_result.slope < 0 else None,
        ])
    trend_df_columns = ['p_value', 'slope', 'slope_pos', 'slope_pos_sig', 'slope_neg', 'slope_neg_sig']
    trend_dfs = {m: pd.DataFrame(trend_lists[m], index=loc_list, columns=trend_df_columns)
        for m in range(1, 13)}
    full_trend_df = pd.DataFrame(full_trend_list, index=loc_list, columns=trend_df_columns)

    figure, axes = plt.subplots(3, 4, layout='compressed')
    axes = iter(axes.T.flatten())
    for m in range(1, 13):
        trend_df = trend_dfs[m]
        series = trend_df['p_value']
        axis = next(axes)
        _map = get_map(axis)
        mx, my = _map(lons, lats)
        scatter_map(axis, mx, my, series, size_func=lambda x: 8,
            cb_min= series.min(), cb_max=series.max())
        month_str = datetime(2000, m, 1).strftime("%B")
        axis.set_title(f'{month_str} p-value')
        pos_percent = round(100 * np.count_nonzero(~np.isnan(trend_df["slope_pos_sig"])) / 1391, 1)
        neg_percent = round(100 * np.count_nonzero(~np.isnan(trend_df["slope_neg_sig"])) / 1391, 1)
        # print(f'{month_str}: positive trend at {pos_percent}% of locations')
        # print(f'{month_str}: negative trend at {neg_percent}% of locations')
        print(f'{month_str} & {pos_percent}\\% & {neg_percent}\\% \\\\')

    figure, axes = plt.subplots(3, 4, layout='compressed')
    axes = iter(axes.flatten())
    for m in range(1, 13):
        axis = next(axes)
        _map = get_map(axis)
        mx, my = _map(lons, lats)
        series = trend_dfs[m]['slope_pos']
        scatter_map(axis, mx, my, series,
            cb_min=0, cb_max=np.max(series),
            show_cb=False, cmap='jet', size_func=lambda x: 2)
        scatter_map(axis, mx, my, trend_dfs[m]['slope_pos_sig'],
            cb_min=0, cb_max=np.max(series),
            cmap='jet', size_func=lambda x: 30,)
        month_str = datetime(2000, m, 1).strftime("%B")
        axis.set_title(f'{month_str} slope (positive)')

    figure, axes = plt.subplots(3, 4, layout='compressed')
    axes = iter(axes.flatten())
    for m in range(1, 13):
        axis = next(axes)
        _map = get_map(axis)
        mx, my = _map(lons, lats)
        series = trend_dfs[m]['slope_neg']
        scatter_map(axis, mx, my, series,
            cb_min=np.min(series), cb_max=0,
            show_cb=False, cmap='jet_r', size_func=lambda x: 2)
        scatter_map(axis, mx, my, trend_dfs[m]['slope_neg_sig'],
            cb_min=np.min(series), cb_max=0,
            cmap='jet_r', size_func=lambda x: 30,)
        month_str = datetime(2000, m, 1).strftime("%B")
        axis.set_title(f'{month_str} slope (negative)')

    pos_percent = round(100 * np.count_nonzero(~np.isnan(full_trend_df["slope_pos_sig"])) / 1391, 1)
    neg_percent = round(100 * np.count_nonzero(~np.isnan(full_trend_df["slope_neg_sig"])) / 1391, 1)
    # print(f'Positive trend at {pos_percent}% of locations')
    # print(f'Negative trend at {neg_percent}% of locations')
    print(f'Full series & {pos_percent}\\% & {neg_percent}\\%')
    figure, axes = plt.subplots(1, 3, layout='compressed')
    axes = iter(axes.flatten())
    axis = next(axes)
    _map = get_map(axis)
    series = full_trend_df['p_value']
    scatter_map(axis, mx, my, series, size_func=lambda x: 30,
        cb_min= series.min(), cb_max=series.max())
    axis.set_title('Full seasonal Kendall p-value')

    axis = next(axes)
    _map = get_map(axis)
    series = full_trend_df['slope_pos']
    scatter_map(axis, mx, my, series, size_func=lambda x: 2,
        cb_min=0, cb_max=np.max(series),
        show_cb=False, cmap='jet')
    scatter_map(axis, mx, my, full_trend_df['slope_pos_sig'],
        cb_min=0, cb_max=np.max(series),
        cmap='jet', size_func=lambda x: 30)
    axis.set_title('Full seasonal Kendall slope (positive)')
    
    axis = next(axes)
    _map = get_map(axis)
    series = full_trend_df['slope_neg']
    scatter_map(axis, mx, my, series, size_func=lambda x: 2,
        cb_min=np.min(series), cb_max=0,
        show_cb=False, cmap='jet_r')
    scatter_map(axis, mx, my, full_trend_df['slope_neg_sig'],
        cb_min=np.min(series), cb_max=0,
        cmap='jet_r', size_func=lambda x: 30)
    axis.set_title('Full seasonal Kendall slope (negative)')

    plt.show()
    
if __name__ == '__main__':
    main()