import os
import ast
import argparse
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import ranksums, wilcoxon as signed_rank

from helpers import get_map, scatter_map, prepare_df, configure_plots
from unami_2009_helpers import prepare_model_df

PERIOD = 24 * 365
BOM_DAILY_PATH = 'data_unfused/bom_daily'
FUSED_DAILY_PATH = 'data/fused_upsampled'
NUM_BOM_SAMPLES = 1000
PREC_INC = 0.2

# Plot intensity as calculated in Unami 2009 on a map
def main():
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='fused_daily_nsrp')
    parser.add_argument('--norm_by_mean', action='store_true', default=False)
    args = parser.parse_args()
    
    monthly_prec_df, _, _ = prepare_df('data/precipitation', 'FusedData.csv', 'prec')
    monthly_prec_df.index = pd.DatetimeIndex(monthly_prec_df.index)
    loc_list = []
    prec_series_list = []
    if args.dataset in ['fused_daily', 'fused_daily_nsrp']:
        pathnames = []
        for path in os.scandir(FUSED_DAILY_PATH):
            if args.dataset == 'fused_daily_nsrp' and not path.name.startswith('fused_daily_nsrp'):
                continue
            if args.dataset == 'fused_daily' and not path.name.endswith('it_3000.csv'):
                continue
            pathnames.append(path.name)
        for pathname in pathnames:
            prec_df = pd.read_csv(f'{FUSED_DAILY_PATH}/{pathname}', index_col=0)
            prec_series = pd.Series(prec_df.values[:, 0], index=pd.DatetimeIndex(prec_df.index))
            loc = ast.literal_eval(prec_df.columns[0])
            loc_list.append(loc)
            prec_series_list.append(prec_series)
    elif args.dataset == 'bom_daily':
        # Select random files (each corresponding to a location) from the BOM daily dataset
        info_df = pd.read_csv('bom_info.csv', index_col=0, converters={0: ast.literal_eval})
        filenames = set(info_df.sample(NUM_BOM_SAMPLES)['filename'])
        for path in os.scandir(BOM_DAILY_PATH):
            if not path.is_file() or path.name not in filenames:
                continue
            prec_df = pd.read_csv(f'{BOM_DAILY_PATH}/{path.name}')
            prec_df = prec_df.dropna(subset=['Rain'])
            prec_df.index = pd.DatetimeIndex(prec_df['Date'])
            loc = (-prec_df.iloc[0]['Lat'], prec_df.iloc[0]['Lon'])
            prec_series = pd.Series(prec_df['Rain']).dropna().loc['2000-01-01':]
            prec_series.name = loc
            loc_list.append(loc)
            prec_series_list.append(prec_series)
    lats, lons = list(zip(*loc_list))

    # Wilcoxon rank-sum testing for significant decadal change in...
    # rs_p = np.zeros((12, monthly_prec_df.shape[1]))
    # rs_stat = np.zeros((12, monthly_prec_df.shape[1]))
    # df_d1 = monthly_prec_df.loc[:'2011-03-31', :]
    # df_d2 = monthly_prec_df.loc['2011-04-01':, :]
    # for m in range(12):
    #     m_df_d1 = df_d1.loc[[t.month == m + 1 for t in df_d1.index], :]
    #     m_df_d2 = df_d2.loc[[t.month == m + 1 for t in df_d2.index], :]
    #     for i, loc in enumerate(loc_list):
    #         test = ranksums(m_df_d2[loc], m_df_d1[loc])
    #         rs_p[m, i] = test.pvalue
    #         rs_stat[m, i] = test.statistic

    # figure, axes = plt.subplots(3, 4, layout='compressed')
    # axes = iter(axes.flatten())
    # for m in range(12):
    #     axis = next(axes)
    #     _map = get_map(axis)
    #     mx, my = _map(lons, lats)
    #     scatter_map(axis, mx, my, rs_p[m, :], cb_min=0, cb_max=1,
    #         size_func=lambda x: 15)
    #     axis.set_title(datetime(2000, m + 1, 1).strftime('%B'))
    # plt.show()

    # figure, axes = plt.subplots(3, 4, layout='compressed')
    # axes = iter(axes.flatten())
    # for m in range(12):
    #     axis = next(axes)
    #     _map = get_map(axis)
    #     masked_p_series = np.where(np.array(rs_p[m, :]) <= 0.05, rs_stat[m, :], np.nan)
    #     _max = np.max(np.abs(np.nan_to_num(masked_p_series)))
    #     scatter_map(axis, mx, my, masked_p_series, cb_min=-_max, cb_max=_max,
    #         size_func=lambda x: 15, cmap='PRGn')
    #     axis.set_title(datetime(2000, m + 1, 1).strftime('%B'))
    # plt.show()

    # Wilcoxon signed-rank testing for significant decadal change in...
    sr_p = {
        'prec_mean': [],
        'prec_median': [],
        # 'x_mean': [],
        # 'x_median': [],
    }
    sr_stat = {
        'prec_mean': [],
        'prec_median': [],
    }
    i = 0
    for loc, high_freq_series in zip(loc_list, prec_series_list):
        if i % 25 == 0:
            print(i)
        i += 1
        prec = monthly_prec_df[loc]
        prec_d1 = prec[:'2011-03-31']
        prec_d2 = prec['2011-04-01':]
        # model_d1 = prepare_model_df(high_freq_series[:'2011-03-31'], PREC_INC)
        # model_d1 = model_d1.set_index(model_d1['t'])
        # model_d2 = prepare_model_df(high_freq_series['2011-04-01':], PREC_INC)
        # model_d2 = model_d2.set_index(model_d2['t'])
        prec_mean_d1 = []
        prec_mean_d2 = []
        prec_median_d1 = []
        prec_median_d2 = []
        x_mean_d1 = []
        x_mean_d2 = []
        x_median_d1 = []
        x_median_d2 = []
        for m in range(1, 13):
            m_prec_av = prec.loc[[t.month == m for t in prec.index]].mean() \
                if args.norm_by_mean else 1
            m_prec_d1 = prec_d1.loc[[t.month == m for t in prec_d1.index]]
            m_prec_d2 = prec_d2.loc[[t.month == m for t in prec_d2.index]]
            prec_mean_d1.append(m_prec_d1.mean() / m_prec_av)
            prec_mean_d2.append(m_prec_d2.mean() / m_prec_av)
            prec_median_d1.append(m_prec_d1.median() / m_prec_av)
            prec_median_d2.append(m_prec_d2.median() / m_prec_av)
            # x_d1 = model_d1.loc[[
            #     t.month == m for t in model_d1.index]]['x']
            # x_d2 = model_d2.loc[[
            #     t.month == m for t in model_d2.index]]['x']
            # x_mean_d1.append(x_d1.mean())
            # x_mean_d2.append(x_d2.mean())
            # x_median_d1.append(x_d1.median())
            # x_median_d2.append(x_d2.median())
        mean_test = signed_rank(prec_mean_d1, prec_mean_d2)
        median_test = signed_rank(prec_median_d1, prec_median_d2)
        sr_p['prec_mean'].append(mean_test.pvalue)
        sr_p['prec_median'].append(median_test.pvalue)
        sr_stat['prec_mean'].append(mean_test.statistic)
        sr_stat['prec_median'].append(median_test.statistic)
        # series_to_plot['x_mean'].append(signed_rank(x_mean_d1, x_mean_d2).pvalue)
        # series_to_plot['x_median'].append(signed_rank(x_median_d1, x_median_d2).pvalue)

    figure, axes = plt.subplots(2, 2, layout='compressed')
    axes = iter(axes.flatten())
    for series_name, p_series in sr_p.items():
        axis = next(axes)
        _map = get_map(axis)
        mx, my = _map(lons, lats)
        scatter_map(axis, mx, my, p_series, cb_min=0, cb_max=1,
            size_func=lambda x: 25)
        axis.set_title(series_name)
        axis = next(axes)
        _map = get_map(axis)
        masked_p_series = np.where(np.array(p_series) <= 0.05, sr_stat[series_name], np.nan)
        _max = np.max(np.abs(np.nan_to_num(masked_p_series)))
        scatter_map(axis, mx, my, masked_p_series, cb_min=0, cb_max=_max,
            size_func=lambda x: 25)
        axis.set_title(series_name)
    plt.show()
    
if __name__ == '__main__':
    main()