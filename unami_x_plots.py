import os
import ast
from datetime import datetime
import argparse

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import skew
np.random.seed(0)

from helpers import get_map, scatter_map, prepare_df, configure_plots
from unami_2009_helpers import prepare_model_df

BOM_DAILY_PATH = 'data_unfused/bom_daily'
FUSED_DAILY_PATH = 'data/fused_upsampled'
NUM_BOM_SAMPLES = 1000

# Plot intensity as calculated in Unami 2009 on a map
def main():
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='fused')
    parser.add_argument('--prec_inc', type=float, default=0.5)
    args = parser.parse_args()
    
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
            if not path.is_file() or (args.num_samples and path.name not in filenames):
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
    monthly_means = [[] for _ in range(12)]
    for s, prec_series in enumerate(prec_series_list):
        if s % 25 == 0:
            print(s)
        model_df = prepare_model_df(prec_series, args.prec_inc)
        series_to_plot['mean'].append(model_df['x'].mean())
        series_to_plot['std'].append(model_df['x'].std())
        series_to_plot['skew'].append(skew(model_df['x']))
        series_to_plot['q1'].append(np.quantile(model_df['x'], 0.25))
        series_to_plot['median'].append(model_df['x'].median())
        series_to_plot['q3'].append(np.quantile(model_df['x'], 0.75))
        model_df = model_df.set_index(model_df['t'])
        for m in range(12):
            monthly_means[m].append(model_df.loc[[
                t.month == m + 1 for t in model_df.index]]['x'].mean())
         
    figure, axes = plt.subplots(2, 3, layout='compressed')
    axes = iter(axes.flatten())
    for series_name in series_to_plot:
        series = np.array(series_to_plot[series_name])
        axis = next(axes)
        _map = get_map(axis)
        mx, my = _map(lons, lats)
        axis.set_title(f'x {series_name}')
        scatter_map(axis, mx, my, series, size_func=lambda x: 15,
            cb_min=series_min_max[series_name][0],
            cb_max=series_min_max[series_name][1])
    plt.show()
    
    figure, axes = plt.subplots(3, 4, layout='compressed')
    axes = iter(axes.flatten())
    for m, series in enumerate(monthly_means):
        axis = next(axes)
        _map = get_map(axis)
        mx, my = _map(lons, lats)
        axis.set_title(f'x {datetime(2000, m + 1, 1).strftime("%b")} mean')
        scatter_map(axis, mx, my, series, size_func=lambda x: 15,
            cb_min=np.min(series), cb_max=np.max(series))
    plt.show()

if __name__ == '__main__':
    main()