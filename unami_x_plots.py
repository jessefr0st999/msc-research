import os
import ast
from datetime import datetime
import argparse

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import skew

from helpers import get_map, scatter_map, prepare_df, configure_plots
from unami_2009_helpers import prepare_model_df, calculate_param_func, \
    get_x_domain, deseasonalise_x, detrend_x, calculate_param_coeffs

PERIOD = 24 * 365
BOM_DAILY_PATH = 'data_unfused/bom_daily'
FUSED_DAILY_PATH = 'data/fused_upsampled'
NUM_BOM_SAMPLES = 1000

# Plot intensity as calculated in Unami 2009 on a map
def main():
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='fused_daily_nsrp')
    parser.add_argument('--prec_inc', type=float, default=0.2)
    parser.add_argument('--sar_corrected', action='store_true', default=False)
    args = parser.parse_args()
    
    suffix = 'nsrp' if args.dataset == 'fused_daily_nsrp' else 'orig'
    loc_list = []
    prec_series_list = []
    # prec_series_list receives data sequentially by location
    if args.dataset == 'fused':
        prec_df, _, _ = prepare_df('data/precipitation', 'FusedData.csv', 'prec')
        for loc, row in prec_df.T.iterrows():
            loc_list.append(loc)
            prec_series_list.append(row)
    elif args.dataset in ['fused_daily', 'fused_daily_nsrp']:
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
    series_to_plot = {
        'mean': [],
        'std': [],
        'skew': [],
        'q1': [],
        'median': [],
        'q3': [],
    }
    series_min_max = {
        'mean': [-5, -1],
        'std': [0, 3],
        'skew': [0, 0.15],
        'q1': [-7.5, -1.5],
        'median': [-5, -0.5],
        'q3': [-4, 0],
    }
    # month_min_max = [
    #     [-5.5, 0.5],
    #     [-5.5, 0.5],
    #     [-5.5, 0.5],
    #     [-6.5, -1],
    #     [-6.5, -1],
    #     [-7.5, -1],
    #     [-8, -1],
    #     [-8.5, -1],
    #     [-8.5, -1],
    #     [-8, -1],
    #     [-7.5, -0.5],
    #     [-6.5, 0],
    # ]
    monthly_means = [[] for _ in range(12)]
    monthly_q1 = [[] for _ in range(12)]    
    monthly_medians = [[] for _ in range(12)]    
    trend_list = []
    for s, prec_series in enumerate(prec_series_list):
        if s % 25 == 0:
            print(s)
        loc = loc_list[s]
        model_df = prepare_model_df(prec_series, args.prec_inc)
        series_to_plot['mean'].append(model_df['x'].mean())
        series_to_plot['std'].append(model_df['x'].std())
        series_to_plot['skew'].append(skew(model_df['x']))
        series_to_plot['q1'].append(np.quantile(model_df['x'], 0.25))
        series_to_plot['median'].append(model_df['x'].median())
        series_to_plot['q3'].append(np.quantile(model_df['x'], 0.75))
        model_df = model_df.set_index(model_df['t'])
        for m in range(12):
            # monthly_means[m].append(prec_series.loc[[
            #     t.month == m + 1 for t in prec_series.index]].mean())
            monthly_means[m].append(model_df.loc[[
                t.month == m + 1 for t in model_df.index]]['x'].mean())
            monthly_q1[m].append(np.quantile(model_df.loc[[
                t.month == m + 1 for t in model_df.index]]['x'], 0.25))
            monthly_medians[m].append(np.quantile(model_df.loc[[
                t.month == m + 1 for t in model_df.index]]['x'], 0.5))
                        
        if args.sar_corrected:
            beta_coeffs_df = pd.read_csv(f'beta_coeffs_fused_daily_{suffix}.csv',
                index_col=0, converters={0: ast.literal_eval})
            loc_index = list(beta_coeffs_df.index.values).index(loc)
            beta_hats = {
                'beta': pd.read_csv(f'corrected_beta_coeffs_{suffix}.csv')\
                    .iloc[loc_index, :].values,
                'kappa': pd.read_csv(f'corrected_kappa_coeffs_{suffix}.csv')\
                    .iloc[loc_index, :].values,
                'psi': pd.read_csv(f'corrected_psi_coeffs_{suffix}.csv')\
                    .iloc[loc_index, :].values,
            }
        else:
            beta_hats = calculate_param_coeffs(model_df, PERIOD, shift_zero=True)
        param_func = calculate_param_func(model_df, PERIOD, beta_hats)
        x_deseasonalised = deseasonalise_x(model_df, param_func)
        x_detrended = detrend_x(x_deseasonalised, polynomial=1)
        trend = x_deseasonalised - x_detrended
        trend_list.append(trend.iloc[-1] - trend.iloc[0])
         
    figure, axes = plt.subplots(2, 3, layout='compressed')
    axes = iter(axes.flatten())
    for series_name in series_to_plot:
        series = np.array(series_to_plot[series_name])
        axis = next(axes)
        _map = get_map(axis)
        mx, my = _map(lons, lats)
        axis.set_title(f'{series_name}')
        scatter_map(axis, mx, my, series, size_func=lambda x: 30, cmap='inferno_r',
            cb_min=series_min_max[series_name][0],
            cb_max=series_min_max[series_name][1])
            # cb_min=np.min(series), cb_max=np.max(series))
    plt.show()
    
    figure, axes = plt.subplots(3, 4, layout='compressed')
    axes = iter(axes.flatten())
    for m, series in enumerate(monthly_means):
        axis = next(axes)
        _map = get_map(axis)
        mx, my = _map(lons, lats)
        axis.set_title(f'{datetime(2000, m + 1, 1).strftime("%b")} mean')
        scatter_map(axis, mx, my, series, size_func=lambda x: 12,
            cb_min=-9, cb_max=0, cmap='inferno_r')
            # cb_min=np.min(series), cb_max=np.max(series))
            # cb_min=month_min_max[m][0], cb_max=month_min_max[m][1])
    plt.show()
    
    figure, axes = plt.subplots(3, 4, layout='compressed')
    axes = iter(axes.flatten())
    for m, series in enumerate(monthly_q1):
        axis = next(axes)
        _map = get_map(axis)
        mx, my = _map(lons, lats)
        axis.set_title(f'{datetime(2000, m + 1, 1).strftime("%b")} q1')
        scatter_map(axis, mx, my, series, size_func=lambda x: 12,
            cb_min=np.min(series), cb_max=np.max(series), cmap='inferno_r')
    plt.show()
    
    figure, axes = plt.subplots(3, 4, layout='compressed')
    axes = iter(axes.flatten())
    for m, series in enumerate(monthly_medians):
        axis = next(axes)
        _map = get_map(axis)
        mx, my = _map(lons, lats)
        axis.set_title(f'{datetime(2000, m + 1, 1).strftime("%b")} median')
        scatter_map(axis, mx, my, series, size_func=lambda x: 12,
            cb_min=-10, cb_max=0, cmap='inferno_r')
            # cb_min=np.min(series), cb_max=np.max(series))
    plt.show()
         
    figure, axis = plt.subplots(1, 3, layout='compressed')
    _map = get_map(axis)
    axis.set_title('linear trend')
    series = np.array(trend_list)
    scatter_map(axis, mx, my, series, size_func=lambda x: 15,
        cb_min=-np.abs(series).max(),
        cb_max=np.abs(series).max(), cmap='inferno')
    plt.show()

if __name__ == '__main__':
    main()