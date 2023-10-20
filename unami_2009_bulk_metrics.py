import os
import ast
import pickle
from datetime import datetime, timedelta
import argparse
from math import floor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

from helpers import get_map, scatter_map

PERIOD = 24 * 365
NUM_S_INDICES = 12

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_folder', required=True)
    parser.add_argument('--solution_tol', type=float, default=0.1)
    # Below args should match those used for unami_2009_bulk.py
    parser.add_argument('--x_steps', type=int, default=30)
    parser.add_argument('--t_steps', type=int, default=24)
    parser.add_argument('--s_steps', type=int, default=250)
    parser.add_argument('--delta_s', type=float, default=100)
    parser.add_argument('--decadal', action='store_true', default=False)
    args = parser.parse_args()
            
    n = args.x_steps
    m = args.t_steps
    z = args.s_steps
    delta_t = PERIOD / m
    # Fit the parameters then solve the DE for each prec series
    # Extract parameter values and useful statistics from the DE solution
    # Slice the DE solution at evenly-spaced times throughout the year
    # t_indices = [floor(i * m / 12) for i in range(12)]
    # t_indices = [floor(i * m / 6) for i in range(6)]
    # t_indices = [floor(i * m / 4) for i in range(4)]
    t_indices = [0]
    # t_indices = [floor(3 * m / 12)]
    # Sample with various values of x and s
    x_indices = [
        floor(n / 10),
        floor(n / 5),
        floor(n / 2),
        floor(4*n / 5),
        floor(9*n / 10),
    ]
    # s_indices = [ceil((i + 1)*z / NUM_S_INDICES) for i in range(NUM_S_INDICES)]
    # s_indices = [1, 3, 10, 25, 50, 75]
    s_indices = [5, 10, 25, 50, 100, 175, 250]
    s_mesh = np.linspace(0, z * args.delta_s, z + 1)

    series_to_plot = {key: list() for key in [
        'x_inf', 'x_sup',
        'j1_x', 'conv_s', 'pdf_final',
        # 'x_median_last_s', 'x_mean_last_s', 'x_std_last_s',
        # 't_median_last_s', 't_mean_last_s', 't_std_last_s',
        'mean', 'mean_years', 'q1', 'median', 'q3', 'mode', 'mode_pdf',
    ]}
    # for i in range(NUM_S_INDICES):
    #     series_to_plot[f'j1_s_{i + 1}'] = list()
    #     series_to_plot[f'cdf_{i + 1}'] = list()
    for s in s_indices:
        series_to_plot[f'j1_s_{s}'] = list()
        series_to_plot[f'cdf_s_{s}'] = list()
        series_to_plot[f'x_mean_s_{s}'] = list()
        series_to_plot[f't_mean_s_{s}'] = list()

    if args.decadal:
        read_folder = f'{args.read_folder}_d1'
    else:
        read_folder = args.read_folder
    calculated_locations = []
    for path in os.scandir(f'unami_results/{read_folder}'):
        if path.is_file() and path.name.endswith('npy'):
            calculated_locations.append(path.name.split('_')[2].split('.npy')[0])
    loc_list = [ast.literal_eval(loc_str) for loc_str in calculated_locations]
    per_year_df = pd.read_csv('prec_per_year.csv', index_col=[0, 1])

    series_to_plot_path = f'unami_results/{read_folder}/series_to_plot.pkl'
    if os.path.isfile(series_to_plot_path):
        with open(series_to_plot_path, 'rb') as f:
            series_to_plot = pickle.load(f)
    else:
        for s, loc_str in enumerate(calculated_locations):
            u_array = np.load(f'unami_results/{read_folder}/u_array_{loc_str}.npy')
            print(f'({s + 1} / {len(calculated_locations)}) Calculating metrics for {loc_str}...')
            
            current_series = {key: dict() for key in series_to_plot}
            for s_index in s_indices:
                current_series[f'j1_s_{s_index}'][0] = np.abs(u_array[s_index, :, :])\
                    .sum(axis=(0, 1)) / (n - 1) / m
            current_series['j1_x'][0] = np.median(u_array, axis=2).sum(axis=(0, 1)) / m / (z + 1)
            current_series['conv_s'][0] = None
            for i in range(1, z):
                solution_diff = (np.abs(u_array[i, :, :] - u_array[i - 1, :, :]))\
                    .sum(axis=(0, 1))
                if solution_diff < args.solution_tol:
                    current_series['conv_s'][0] = i * args.delta_s
                    break
            # The mean gives expected value of s such that X_s exits the [x_inf, x_sup] domain
            # This requires solving with high enough s that a steady state probability is reached
            # as well as high enough t such that (u, x) falls into a steady cycle
            # TODO: extract PDF value at the mean and median
            # TODO: extract number of year cycles required for steady state
            for k in x_indices:
                # current_series['t_median_last_s'][k] = np.median(1 - u_array[-1, :, k])
                # current_series['t_mean_last_s'][k] = np.mean(1 - u_array[-1, :, k])
                # current_series['t_std_last_s'][k] = np.std(1 - u_array[-1, :, k])
                for s_index in s_indices:
                    current_series[f't_mean_s_{s_index}'][k] = np.mean(1 - u_array[s_index, :, k])
            for j in t_indices:
                # current_series['x_median_last_s'][j] = np.median(1 - u_array[-1, j, :])
                # current_series['x_mean_last_s'][j] = np.mean(1 - u_array[-1, j, :])
                # current_series['x_std_last_s'][j] = np.std(1 - u_array[-1, j, :])
                for s_index in s_indices:
                    current_series[f'x_mean_s_{s_index}'][j] = np.mean(1 - u_array[s_index, j, :])
                for k in x_indices:
                    cdf = 1 - u_array[:, j, k]
                    pdf = [(cdf[i + 1] - cdf[i]) / args.delta_s for i in range(len(cdf) - 1)]
                    # for _i, s_index in enumerate(s_indices):
                    #     current_series[f'cdf_{_i + 1}'][(j, k)] = cdf[s_index]
                    for s_index in s_indices:
                        current_series[f'cdf_s_{s_index}'][(j, k)] = cdf[s_index]
                    # This non-zero indicates the DE should be solved again but for higher s
                    current_series['pdf_final'][(j, k)] = pdf[-1]
                    # current_series['mean'][(j, k)] = None if cdf[-1] < 0.99 else \
                    #     np.sum([s_mesh[i] * p * args.delta_s for i, p in enumerate(pdf)])
                    current_series['mean'][(j, k)] = None if cdf[-1] == 0 else \
                        np.sum([s_mesh[i] * p * args.delta_s for i, p in enumerate(pdf)]) / cdf[-1]
                    current_series['mean_years'][(j, k)] = None if cdf[-1] == 0 else \
                        current_series['mean'][(j, k)] / per_year_df['d1'].loc[loc_list[s]]
                        # current_series['mean'][(j, k)] / per_year_df['both'].loc[loc_list[s]]
                        # current_series['mean'][(j, k)] / per_year_df['d2'].loc[loc_list[s]]
                    current_series['q1'][(j, k)] = None if cdf[-1] < 0.25 else \
                        np.argwhere(cdf >= 0.25)[0, 0] * args.delta_s
                    current_series['median'][(j, k)] = None if cdf[-1] < 0.5 else \
                        np.argwhere(cdf >= 0.5)[0, 0] * args.delta_s
                    current_series['q3'][(j, k)] = None if cdf[-1] < 0.75 else \
                        np.argwhere(cdf >= 0.75)[0, 0] * args.delta_s
                    current_series['mode'][(j, k)] = np.argmax(pdf) * args.delta_s
                    current_series['mode_pdf'][(j, k)] = np.max(pdf)
            for key in series_to_plot:
                series_to_plot[key].append(current_series[key])
        with open(series_to_plot_path, 'wb') as f:
            pickle.dump(series_to_plot, f)
        exit()

    if args.decadal:
        read_folder = f'{args.read_folder}_d2'
        series_to_plot_path = f'unami_results/{read_folder}/series_to_plot.pkl'
        with open(series_to_plot_path, 'rb') as f:
            series_to_plot_d2 = pickle.load(f)
    cmap = 'RdYlBu_r' if args.decadal else 'inferno_r'
    lats, lons = list(zip(*loc_list))
    def df_min(df):
        return df.min().min()
    def df_max(df):
        return df.max().max()
    def df_q(df):
        return np.nanquantile(np.array(df), 0.975)
    
    for series_name, label, _min, _max in [
        # ('j1_x', '1-norm over median x', lambda x: 0, lambda x: 1),
        ('conv_s', 's taken to converge', lambda x: 0, df_max),
    ]:
        df = pd.DataFrame(series_to_plot[series_name], index=loc_list)
        if args.decadal:
            df = pd.DataFrame(series_to_plot_d2[series_name], index=loc_list) - df
            _min = lambda x: -df_max(np.abs(df))
            _max = lambda x: df_max(np.abs(df))
        figure, axis = plt.subplots(1)
        _map = get_map(axis)
        mx, my = _map(lons, lats)
        axis.set_title(label)
        scatter_map(axis, mx, my, df[0], cb_min=_min(df),
            cb_max=25000, size_func=lambda x: 15, cmap=cmap)
        plt.show()

    s_indices = [10, 25, 50, 100]
    figure, axes = plt.subplots(1, len(s_indices), layout='compressed')
    axes = iter(axes.flatten())
    for s in s_indices:
        series_name = f'j1_s_{s}'
        df = pd.DataFrame(series_to_plot[series_name], index=loc_list)
        if args.decadal:
            df = pd.DataFrame(series_to_plot_d2[series_name], index=loc_list) - df
            _min = lambda x: -df_max(np.abs(df))
            _max = lambda x: df_max(np.abs(df))
        axis = next(axes)
        _map = get_map(axis)
        mx, my = _map(lons, lats)
        axis.set_title(f'1-norm at s = {s * args.delta_s}')
        scatter_map(axis, mx, my, df[0], cb_min=0, cb_max=1,
            size_func=lambda x: 15, cmap=cmap)
    plt.show()

    for series_name, label, _min, _max in [
        # ('x_median_last_s', 'median probability across x at final s', lambda x: 0, lambda x: 1),
        # ('x_mean_last_s', 'mean probability across x at final s', lambda x: 0, lambda x: 1),
        # ('x_std_last_s', 'std probability across x at final s', lambda x: 0, df_max),
        # *[(f'x_mean_s_{s}', f'mean probability across x at s = {s * args.delta_s}',
        #     lambda x: 0, lambda x: 1) for s in s_indices],
    ]:
        # For each t, plot the results
        df = pd.DataFrame(series_to_plot[series_name], index=loc_list)
        if args.decadal:
            df = pd.DataFrame(series_to_plot_d2[series_name], index=loc_list) - df
            _min = lambda x: -df_max(np.abs(df))
            _max = lambda x: df_max(np.abs(df))
        # figure, axes = plt.subplots(3, 4, layout='compressed')
        figure, axes = plt.subplots(2, 3, layout='compressed')
        axes = iter(axes.flatten())
        for i, j in enumerate(df.columns):
            if j not in t_indices:
                continue
            axis = next(axes)
            _map = get_map(axis)
            mx, my = _map(lons, lats)
            scatter_map(axis, mx, my, df[j], cb_min=_min(df),
                cb_max=_max(df), size_func=lambda x: 15, cmap=cmap)
            days = int(j) * delta_t / 24
            date_part = (datetime(2000, 1, 1) + timedelta(days=days)).strftime('%b %d')
            axis.set_title(f'{label} at t = {date_part}')
        plt.show()
        
    for series_name, label, _min, _max in [
        # ('t_median_last_s', 'median probability across t at final s', lambda x: 0, lambda x: 1),
        # ('t_mean_last_s', 'mean probability across t at final s', lambda x: 0, lambda x: 1),
        # ('t_std_last_s', 'std probability across t at final s', lambda x: 0, df_max),
        # *[(f't_mean_s_{s}', f'mean probability across t at s = {s * args.delta_s}',
        #     lambda x: 0, lambda x: 1) for s in s_indices],
    ]:
        # For each x, plot the results
        df = pd.DataFrame(series_to_plot[series_name], index=loc_list)
        if args.decadal:
            df = pd.DataFrame(series_to_plot_d2[series_name], index=loc_list) - df
            _min = lambda x: -df_max(np.abs(df))
            _max = lambda x: df_max(np.abs(df))
        figure, axes = plt.subplots(2, 3, layout='compressed')
        axes = iter(axes.flatten())
        for k in x_indices:
            axis = next(axes)
            _map = get_map(axis)
            mx, my = _map(lons, lats)
            scatter_map(axis, mx, my, df[k], cb_min=_min(df),
                cb_max=_max(df), size_func=lambda x: 15, cmap=cmap)
            axis.set_title(f'{label} at x = x_inf + {k} * delta_x')
        next(axes).axis('off')
        plt.show()
        
    x_indices = [
        # floor(n / 10),
        floor(n / 2),
        # floor(9*n / 10),
    ]
    for series_name, label, _min, _max in [
        # *[(f'cdf_s_{s}', f'cdf at s = {s * args.delta_s}', lambda x: 0, lambda x: 1)
        #     for s in s_indices],
        # ('pdf_final', 'pdf at last s', lambda x: 0, df_max),
        ('mean', 'mean of pdf', lambda x: 0, lambda x: 6500),
        ('mean_years', 'mean of pdf', lambda x: 0, lambda x: 5),
        # ('q1', '0.25 quantile of pdf', lambda x: 0, df_max),
        # ('median', 'median of pdf', lambda x: 0, lambda x: 7500),
        # ('q3', '0.75 quantile of pdf', lambda x: 0, df_max),
        # ('mode', 'mode of pdf', lambda x: 0, lambda x: 1500),
        # ('mode_pdf', 'pdf value at mode', lambda x: 0, df_max),
    ]:
        # For each time and x-value, plot the results
        df = pd.DataFrame(series_to_plot[series_name], index=loc_list)
        if args.decadal:
            df = pd.DataFrame(series_to_plot_d2[series_name], index=loc_list) - df
            # _max = lambda x: df_q(np.abs(df))
            # _max = lambda x: 750
            _max = lambda x: 1
            _min = lambda x: -_max(0)
        for k in x_indices:
            # figure, axes = plt.subplots(3, 4, layout='compressed')
            figure, axes = plt.subplots(1, 4, layout='compressed')
            axes = iter(axes.flatten())
            for i, (j, _k) in enumerate(df.columns):
                if j not in t_indices:
                    continue
                if _k != k:
                    continue
                axis = next(axes)
                _map = get_map(axis)
                mx, my = _map(lons, lats)
                scatter_map(axis, mx, my, df[(j, k)], cb_min=_min(df),
                    cb_max=_max(df), size_func=lambda x: 15, cmap=cmap)
                days = int(j) * delta_t / 24
                date_part = (datetime(2000, 1, 1) + timedelta(days=days)).strftime('%b %d')
                # axis.set_title(f'{label} at t = {date_part}, x = x_inf + {k} * delta_x')
        plt.show()
    
if __name__ == '__main__':
    main()