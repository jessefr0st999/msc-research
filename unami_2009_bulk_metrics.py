import os
import ast
import pickle
from datetime import datetime, timedelta
import argparse
from math import floor

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
np.random.seed(0)

from helpers import get_map, scatter_map, configure_plots

PERIOD = 24 * 365
NUM_S_INDICES = 12

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_folder', default=None, required=True)
    parser.add_argument('--solution_tol', type=float, default=0.01)
    # Below args should match those used for unami_2009_bulk.py
    parser.add_argument('--x_steps', type=int, default=50)
    parser.add_argument('--t_steps', type=int, default=50)
    parser.add_argument('--s_steps', type=int, default=100)
    parser.add_argument('--delta_s', type=float, default=10)
    args = parser.parse_args()
            
    n = args.x_steps
    m = args.t_steps
    z = args.s_steps
    delta_t = PERIOD / m
    # Fit the parameters then solve the DE for each prec series
    # Extract parameter values and useful statistics from the DE solution
    s_mesh = np.linspace(0, z * args.delta_s, z + 1)
    # Slice the DE solution at evenly-spaced times throughout the year
    t_indices = [floor(i * m / 12) for i in range(12)]
    # Sample with various values of x and s
    x_indices = [
        floor(n / 10),
        floor(n / 5),
        floor(n / 2),
        floor(4*n / 5),
        floor(9*n / 10),
    ]
    # s_indices = [ceil((i + 1)*z / NUM_S_INDICES) for i in range(NUM_S_INDICES)]
    # s_indices = [20, 40, 60, 80]
    s_indices = [5, 10, 25, 50, 75]

    series_to_plot = {key: list() for key in [
        'x_inf', 'x_sup', 'trend',
        'j2_x', 'j2_t', 'conv_s', 'pdf_final',
        # 'x_median_last_s', 'x_mean_last_s', 'x_std_last_s',
        # 't_median_last_s', 't_mean_last_s', 't_std_last_s',
        'mean', 'q1', 'median', 'q3', 'mode', 'mode_pdf',
    ]}
    # for i in range(NUM_S_INDICES):
    #     series_to_plot[f'j2_s_{i + 1}'] = list()
    #     series_to_plot[f'cdf_{i + 1}'] = list()
    for s in s_indices:
        series_to_plot[f'j2_s_{s}'] = list()
        series_to_plot[f'cdf_s_{s}'] = list()
        series_to_plot[f'x_mean_s_{s}'] = list()
        series_to_plot[f't_mean_s_{s}'] = list()

    calculated_locations = []
    for path in os.scandir(f'unami_results/{args.read_folder}'):
        if path.is_file() and path.name.endswith('npy'):
            calculated_locations.append(path.name.split('_')[2].split('.npy')[0])
    loc_list = [ast.literal_eval(loc_str) for loc_str in calculated_locations]

    series_to_plot_path = f'unami_results/{args.read_folder}/series_to_plot.pkl'
    if os.path.isfile(series_to_plot_path):
        with open(series_to_plot_path, 'rb') as f:
            series_to_plot = pickle.load(f)
    else:
        for s, loc_str in enumerate(calculated_locations):
            u_array = np.load(f'unami_results/{args.read_folder}/u_array_{loc_str}.npy')
            u_last_cycle = u_array[:, -m:, :]
            print(f'({s + 1} / {len(calculated_locations)}) Calculating metrics for {loc_str}...')
            
            current_series = {key: dict() for key in series_to_plot}
            # for _i, s_index in enumerate(s_indices):
            #     current_series[f'j2_s_{_i + 1}'][0] = (u_last_cycle[s_index, :, :] ** 2)\
            #         .sum(axis=(0, 1)) / (n - 1) / m
            for s_index in s_indices:
                current_series[f'j2_s_{s_index}'][0] = (u_last_cycle[s_index, :, :] ** 2)\
                    .sum(axis=(0, 1)) / (n - 1) / m
            current_series['j2_t'][0] = np.median(u_last_cycle, axis=1).sum(axis=(0, 1)) / (n - 1) / (z + 1)
            current_series['j2_x'][0] = np.median(u_last_cycle, axis=2).sum(axis=(0, 1)) / m / (z + 1)
            for i in range(1, z):
                solution_diff = ((u_array[i, -m:, :] - u_array[i - 1, -m:, :]) ** 2)\
                    .sum(axis=(0, 1))
                if solution_diff < args.solution_tol:
                    break
            current_series['conv_s'][0] = i * args.delta_s
            # The mean gives expected value of s such that X_s exits the [x_inf, x_sup] domain
            # This requires solving with high enough s that a steady state probability is reached
            # as well as high enough t such that (u, x) falls into a steady cycle
            # TODO: extract PDF value at the mean and median
            # TODO: extract number of year cycles required for steady state
            for k in x_indices:
                # current_series['t_median_last_s'][k] = np.median(1 - u_last_cycle[-1, :, k])
                # current_series['t_mean_last_s'][k] = np.mean(1 - u_last_cycle[-1, :, k])
                # current_series['t_std_last_s'][k] = np.std(1 - u_last_cycle[-1, :, k])
                for s_index in s_indices:
                    current_series[f't_mean_s_{s_index}'][k] = np.mean(1 - u_last_cycle[s_index, :, k])
            for j in t_indices:
                # current_series['x_median_last_s'][j] = np.median(1 - u_last_cycle[-1, j, :])
                # current_series['x_mean_last_s'][j] = np.mean(1 - u_last_cycle[-1, j, :])
                # current_series['x_std_last_s'][j] = np.std(1 - u_last_cycle[-1, j, :])
                for s_index in s_indices:
                    current_series[f'x_mean_s_{s_index}'][j] = np.mean(1 - u_last_cycle[s_index, j, :])
                for k in x_indices:
                    cdf = 1 - u_last_cycle[:, j, k]
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
                        np.sum([s_mesh[i] * p * args.delta_s for i, p in enumerate(pdf)])
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
    lats, lons = list(zip(*loc_list))
    def df_min(df):
        return df.min().min()
    def df_max(df):
        return df.max().max()
    
    for series_name, label, _min, _max in [
        # ('j2_s_3', '2-norm over 1/4 max s', lambda x: 0, lambda x: 1),
        # ('j2_s_6', '2-norm over 1/2 max s', lambda x: 0, lambda x: 1),
        # ('j2_s_9', '2-norm over 3/4 max s', lambda x: 0, lambda x: 1),
        # ('j2_s_12', '2-norm over last s', lambda x: 0, lambda x: 1),
        ('j2_s_5', '2-norm at s = 5', lambda x: 0, lambda x: 1),
        ('j2_s_10', '2-norm at s = 10', lambda x: 0, lambda x: 1),
        ('j2_s_25', '2-norm at s = 25', lambda x: 0, lambda x: 1),
        ('j2_s_50', '2-norm at s = 50', lambda x: 0, lambda x: 1),
        ('j2_s_75', '2-norm at s = 75', lambda x: 0, lambda x: 1),
        ('j2_t', '2-norm over median t', lambda x: 0, lambda x: 1),
        ('j2_x', '2-norm over median x', lambda x: 0, lambda x: 1),
        ('conv_s', 's taken to converge', lambda x: 0, df_max),
    ]:
        df = pd.DataFrame(series_to_plot[series_name], index=loc_list)
        figure, axis = plt.subplots(1)
        _map = get_map(axis)
        mx, my = _map(lons, lats)
        axis.set_title(label)
        scatter_map(axis, mx, my, df[0], cb_min=_min(df),
            cb_max=_max(df), size_func=lambda x: 15, cmap='inferno_r')
        plt.show()

    for series_name, label, _min, _max in [
        # ('x_median_last_s', 'median probability across x at final s', lambda x: 0, lambda x: 1),
        # ('x_mean_last_s', 'mean probability across x at final s', lambda x: 0, lambda x: 1),
        # ('x_std_last_s', 'std probability across x at final s', lambda x: 0, df_max),
        ('x_mean_s_5', 'mean probability across x at s = 5', lambda x: 0, lambda x: 1),
        ('x_mean_s_10', 'mean probability across x at s = 10', lambda x: 0, lambda x: 1),
        ('x_mean_s_25', 'mean probability across x at s = 25', lambda x: 0, lambda x: 1),
        ('x_mean_s_50', 'mean probability across x at s = 50', lambda x: 0, lambda x: 1),
        ('x_mean_s_75', 'mean probability across x at s = 75', lambda x: 0, lambda x: 1),
    ]:
        # For each time, plot the results
        df = pd.DataFrame(series_to_plot[series_name], index=loc_list)
        figure, axes = plt.subplots(3, 4, layout='compressed')
        axes = iter(axes.flatten())
        for i, j in enumerate(df.columns):
            axis = next(axes)
            _map = get_map(axis)
            mx, my = _map(lons, lats)
            scatter_map(axis, mx, my, df[j], cb_min=_min(df),
                cb_max=_max(df), size_func=lambda x: 15, cmap='inferno_r')
            days = int(j) * delta_t / 24
            date_part = (datetime(2000, 1, 1) + timedelta(days=days)).strftime('%b %d')
            axis.set_title(f'{label} at t = {date_part}')
        plt.show()
        
    for series_name, label, _min, _max in [
        # ('t_median_last_s', 'median probability across t at final s', lambda x: 0, lambda x: 1),
        # ('t_mean_last_s', 'mean probability across t at final s', lambda x: 0, lambda x: 1),
        # ('t_std_last_s', 'std probability across t at final s', lambda x: 0, df_max),
        ('t_mean_s_5', 'mean probability across t at s = 5', lambda x: 0, lambda x: 1),
        ('t_mean_s_10', 'mean probability across t at s = 10', lambda x: 0, lambda x: 1),
        ('t_mean_s_25', 'mean probability across t at s = 25', lambda x: 0, lambda x: 1),
        ('t_mean_s_50', 'mean probability across t at s = 50', lambda x: 0, lambda x: 1),
        ('t_mean_s_75', 'mean probability across t at s = 75', lambda x: 0, lambda x: 1),
    ]:
        # For each x-value, plot the results
        df = pd.DataFrame(series_to_plot[series_name], index=loc_list)
        figure, axes = plt.subplots(2, 3, layout='compressed')
        axes = iter(axes.flatten())
        for k in x_indices:
            axis = next(axes)
            _map = get_map(axis)
            mx, my = _map(lons, lats)
            scatter_map(axis, mx, my, df[k], cb_min=_min(df),
                cb_max=_max(df), size_func=lambda x: 15, cmap='inferno_r')
            axis.set_title(f'{label} at x = x_inf + {k} * delta_x')
        next(axes).axis('off')
        plt.show()
        
    x_indices = [
        floor(n / 10),
        floor(n / 2),
        floor(9*n / 10),
    ]
    for series_name, label, _min, _max in [
        # ('cdf_3', 'cdf at 1/4 max s', lambda x: 0, lambda x: 1),
        # ('cdf_6', 'cdf at 1/2 max s', lambda x: 0, lambda x: 1),
        # ('cdf_9', 'cdf at 3/4 max s', lambda x: 0, lambda x: 1),
        # ('cdf_12', 'cdf at last s', lambda x: 0, lambda x: 1),
        ('cdf_s_5', 'cdf at s = 5', lambda x: 0, lambda x: 1),
        ('cdf_s_10', 'cdf at s = 10', lambda x: 0, lambda x: 1),
        ('cdf_s_25', 'cdf at s = 25', lambda x: 0, lambda x: 1),
        ('cdf_s_50', 'cdf at s = 50', lambda x: 0, lambda x: 1),
        ('cdf_s_75', 'cdf at s = 75', lambda x: 0, lambda x: 1),
        ('pdf_final', 'pdf at last s', lambda x: 0, df_max),
        ('mean', 'mean of pdf', lambda x: 0, df_max),
        # ('q1', '0.25 quantile of pdf', lambda x: 0, df_max),
        ('median', 'median of pdf', lambda x: 0, df_max),
        # ('q3', '0.75 quantile of pdf', lambda x: 0, df_max),
        ('mode', 'mode of pdf', lambda x: 0, df_max),
        ('mode_pdf', 'pdf value at mode', lambda x: 0, df_max),
    ]:
        # For each time and x-value, plot the results
        df = pd.DataFrame(series_to_plot[series_name], index=loc_list)
        for k in x_indices:
            figure, axes = plt.subplots(3, 4, layout='compressed')
            axes = iter(axes.flatten())
            for i, (j, _k) in enumerate(df.columns):
                if _k != k:
                    continue
                axis = next(axes)
                _map = get_map(axis)
                mx, my = _map(lons, lats)
                scatter_map(axis, mx, my, df[(j, k)], cb_min=_min(df),
                    cb_max=_max(df), size_func=lambda x: 15, cmap='inferno_r')
                days = int(j) * delta_t / 24
                date_part = (datetime(2000, 1, 1) + timedelta(days=days)).strftime('%b %d')
                axis.set_title(f'{label} at t = {date_part}, x = x_inf + {k} * delta_x')
        plt.show()
    
if __name__ == '__main__':
    main()