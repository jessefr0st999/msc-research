import os
import ast
import pickle
from datetime import datetime, timedelta
import argparse
from math import floor, ceil
from threading import Thread

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
np.random.seed(0)

from helpers import get_map, scatter_map, prepare_df, configure_plots
from unami_2009_helpers import prepare_model_df, calculate_param_coeffs, \
    calculate_param_func, build_scheme, plot_results, get_x_domain, \
    deseasonalise_x, detrend_x

PERIOD = 24 * 365
BOM_DAILY_PATH = 'data_unfused/bom_daily'
FUSED_DAILY_PATH = 'data/fused_upsampled'
START_DATE = datetime(2000, 1, 1)
PLOT_LOCATIONS = {
    (-37.75, 145.5): 'Melbourne',
    (-12.75, 131.5): 'Darwin',
    (-17.75, 140.5): 'NW Qld',
    (-28.75, 153.5): 'Lismore',
    (-25.75, 133.5): 'Central Australia',
    (-33.25, 151.5): 'Central Coast',
    (-31.75, 116.5): 'Perth',
    (-17.75, 122.5): 'Broome',
    (-35.25, 138.5): 'Adelaide',
    (-42.75, 147.5): 'Hobart',
    (-27.75, 152.5): 'Brisbane',
    (-16.75, 145.5): 'Cairns',
}
PLOT_LOCATIONS = None
if PLOT_LOCATIONS:
    mpl.rcParams.update({'axes.titlesize': 10})
NUM_S_INDICES = 12


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_folder', default=None)
    parser.add_argument('--read_folder', default=None)
    parser.add_argument('--dataset', default='fused')
    parser.add_argument('--prec_inc', type=float, default=0.5)
    parser.add_argument('--x_steps', type=int, default=50)
    parser.add_argument('--t_steps', type=int, default=50)
    parser.add_argument('--s_steps', type=int, default=100)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--delta_s', type=float, default=10)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--np_bcs', action='store_true', default=False)
    parser.add_argument('--shrink_x_proportion', type=float, default=None)
    parser.add_argument('--shrink_x_quantile', type=float, default=None)
    parser.add_argument('--shrink_x_mixed', action='store_true', default=False)
    parser.add_argument('--year_cycles', type=int, default=1)
    parser.add_argument('--side', default=None)
    parser.add_argument('--sar_corrected', action='store_true', default=False)
    parser.add_argument('--solution_tol', type=float, default=0.01)
    parser.add_argument('--solution_tol_exit', action='store_true', default=False)
    args = parser.parse_args()

    if bool(args.read_folder) == bool(args.save_folder):
        print('Exactly one of args.read_folder and args.save_folder required.')
        exit()
    
    suffix = 'nsrp' if args.dataset == 'fused_daily_nsrp' else 'orig'
    lower_q_df = pd.read_csv(f'x_lower_quantiles_{suffix}.csv', index_col=0)
    upper_q_df = pd.read_csv(f'x_upper_quantiles_{suffix}.csv', index_col=0)
    full_loc_list = []
    prec_series_list = []
    # prec_series_list receives data sequentially by location
    if args.dataset == 'fused':
        # Select random locations from the fused dataset
        prec_df, _, _ = prepare_df('data/precipitation', 'FusedData.csv', 'prec')
        num_samples = args.num_samples if args.num_samples else 1391
        _prec_df = prec_df.T.sample(num_samples)
        for loc, row in _prec_df.iterrows():
            full_loc_list.append(loc)
            prec_series_list.append(row)
    elif args.dataset in ['fused_daily', 'fused_daily_nsrp']:
        num_samples = args.num_samples if args.num_samples else 1391
        pathnames = []
        for path in os.scandir(FUSED_DAILY_PATH):
            if args.dataset == 'fused_daily_nsrp' and not path.name.startswith('fused_daily_nsrp'):
                continue
            if args.dataset == 'fused_daily' and not path.name.endswith('it_3000.csv'):
                continue
            pathnames.append(path.name)
        pathnames = np.random.choice(pathnames, num_samples, replace=False)
        for pathname in pathnames:
            prec_df = pd.read_csv(f'{FUSED_DAILY_PATH}/{pathname}', index_col=0)
            prec_series = pd.Series(prec_df.values[:, 0], index=pd.DatetimeIndex(prec_df.index))
            loc = ast.literal_eval(prec_df.columns[0])
            full_loc_list.append(loc)
            prec_series_list.append(prec_series)
    elif args.dataset == 'bom_daily':
        # Select random files (each corresponding to a location) from the BOM daily dataset
        info_df = pd.read_csv('bom_info.csv', index_col=0, converters={0: ast.literal_eval})
        filenames = set(info_df.sample(args.num_samples)['filename']) if args.num_samples else None
        for i, path in enumerate(os.scandir(BOM_DAILY_PATH)):
            if not path.is_file() or (args.num_samples and path.name not in filenames):
                continue
            prec_df = pd.read_csv(f'{BOM_DAILY_PATH}/{path.name}')
            prec_df = prec_df.dropna(subset=['Rain'])
            prec_df.index = pd.DatetimeIndex(prec_df['Date'])
            loc = (-prec_df.iloc[0]['Lat'], prec_df.iloc[0]['Lon'])
            prec_series = pd.Series(prec_df['Rain']).dropna().loc['2000-01-01':]
            prec_series.name = loc
            full_loc_list.append(loc)
            prec_series_list.append(prec_series)
            
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
    s_indices = [ceil((i + 1)*z / NUM_S_INDICES) for i in range(NUM_S_INDICES)]

    series_to_plot_file = f'unami_results/{args.save_folder}/series_to_plot.pkl'
    if os.path.isfile(series_to_plot_file):
        with open(series_to_plot_file, 'rb') as f:
            series_to_plot = pickle.load(f)
    else:
        series_to_plot = {key: list() for key in [
            'x_inf', 'x_sup', 'trend',
            'j2_x', 'j2_t', 'conv_s', 'pdf_final',
            'x_median_last_s', 'x_mean_last_s', 'x_std_last_s',
            't_median_last_s', 't_mean_last_s', 't_std_last_s',
            'mean', 'q1', 'median', 'q3', 'mode', 'mode_pdf',
        ]}
        for i in range(NUM_S_INDICES):
            series_to_plot[f'j2_s_{i + 1}'] = list()
            series_to_plot[f'cdf_{i + 1}'] = list()

    calculated_locations = []
    target_folder = args.read_folder or args.save_folder
    for path in os.scandir(f'unami_results/{target_folder}'):
        if path.is_file():
            calculated_locations.append(path.name.split('_')[2].split('.npy')[0])
    loc_list = []
    for s, prec_series in enumerate(prec_series_list):
        loc = full_loc_list[s]
        if PLOT_LOCATIONS and loc not in PLOT_LOCATIONS:
            continue
        # If reading, skip if data for a location hasn't already been calculated
        if args.read_folder and str(loc) not in calculated_locations:
            continue
        # If calculating, skip if it has already been calculated
        if args.save_folder and str(loc) in calculated_locations:
            continue
        loc_list.append(loc)
        if args.read_folder:
            continue
        model_df = prepare_model_df(prec_series, args.prec_inc)
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
        x_data = np.array(model_df['x'])
        
        t_mesh = np.linspace(0, args.year_cycles * PERIOD, args.year_cycles * m + 1)
        if args.shrink_x_mixed:
            lower_q = lower_q_df.loc[str(loc)][0]
            upper_q = upper_q_df.loc[str(loc)][0]
        else:
            lower_q, upper_q = None, None
        x_inf, x_sup = get_x_domain(model_df['x'], args.shrink_x_proportion,
            args.shrink_x_quantile, lower_q, upper_q)
        scheme_output = build_scheme(param_func, t_mesh, n, m, args.delta_s,
            delta_t, args.np_bcs, x_inf=x_inf, x_sup=x_sup, side=args.side)
        if args.np_bcs:
            M_mats, G_mats, H_mats = scheme_output
        else:
            A_mat, G_mats = scheme_output
        
        print(f'({s + 1} / {len(prec_series_list)}) Solving linear systems for {str(loc)}...')
        year_cycles = args.year_cycles if args.np_bcs else 1
        x_size = n if args.side else n - 1
        u_array = np.zeros((z + 1, year_cycles * m, x_size))
        u_array[0, :, :] = 1  # BC of 1 at s = 0
        if args.np_bcs:
            u_array[:, 0, :] = 1  # BC of 1 at t = 0
        for i in range(1, z + 1):
            if i > 1 and args.solution_tol_exit:
                solution_diff = ((u_array[i, -m:, :] - u_array[i - 1, -m:, :]) ** 2)\
                    .sum(axis=(0, 1))
                if solution_diff < args.solution_tol:
                    for _i in range(i + 1, z + 1):
                        u_array[_i, :, :] = u_array[i, :, :]
                    break
            if i % 10 == 0:
                print(i, '/', z)
            if args.np_bcs:
                # Solve iteratively for each s and t
                for j in range(1, year_cycles * m):
                    b_vec = None
                    t_index = j
                    while b_vec is None:
                        try:
                            b_vec = G_mats[t_index] @ u_array[i - 1, j, :] + H_mats[t_index] @ u_array[i, j - 1, :]
                            u_array[i, j, :] = spsolve(M_mats[t_index], b_vec)
                        except IndexError:
                            t_index -= m
            else:
                # Solve iteratively for each s
                b_vec = np.zeros((m * x_size, 1))
                for j in range(m):
                    start = j * x_size
                    end = (j + 1) * x_size
                    b_vec[start : end] = G_mats[j] @ u_array[i - 1, j, :].reshape((x_size, 1))
                u_vec = spsolve(A_mat, b_vec)
                for j in range(m):
                    start = j * x_size
                    end = (j + 1) * x_size
                    u_array[i, j, :] = u_vec[start : end]
        # TODO: Implement this when reading from folder
        if args.plot:
            plot_results(u_array, x_data, t_mesh, n, m, z, args.delta_s, delta_t,
                param_func, start_date=model_df['t'].iloc[0] if args.np_bcs else None,
                non_periodic=False, x_inf=x_inf, x_sup=x_sup, side=args.side,
                title=f'{PLOT_LOCATIONS[loc]} {str(loc)}' if PLOT_LOCATIONS else str(loc))
        np.save(f'unami_results/{args.save_folder}/u_array_{str(loc)}.npy', u_array)
        
        current_series = {key: dict() for key in series_to_plot}
        current_series['x_inf'][0] = x_inf
        current_series['x_sup'][0] = x_sup
        x_deseasonalised = deseasonalise_x(model_df, param_func)
        x_detrended = detrend_x(x_deseasonalised, polynomial=1)
        trend = x_deseasonalised - x_detrended
        current_series['trend'][0] = trend.iloc[-1] - trend.iloc[0]
        u_last_cycle = u_array[:, -m:, :]
        for _i, s_index in enumerate(s_indices):
            current_series[f'j2_s_{_i + 1}'][0] = (u_last_cycle[s_index, :, :] ** 2)\
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
            current_series['t_median_last_s'][k] = np.median(1 - u_last_cycle[-1, :, k])
            current_series['t_mean_last_s'][k] = np.mean(1 - u_last_cycle[-1, :, k])
            current_series['t_std_last_s'][k] = np.std(1 - u_last_cycle[-1, :, k])
        for j in t_indices:
            current_series['x_median_last_s'][j] = np.median(1 - u_last_cycle[-1, j, :])
            current_series['x_mean_last_s'][j] = np.mean(1 - u_last_cycle[-1, j, :])
            current_series['x_std_last_s'][j] = np.std(1 - u_last_cycle[-1, j, :])
            for k in x_indices:
                cdf = 1 - u_last_cycle[:, j, k]
                pdf = [(cdf[i + 1] - cdf[i]) / args.delta_s for i in range(len(cdf) - 1)]
                for _i, s_index in enumerate(s_indices):
                    current_series[f'cdf_{_i + 1}'][(j, k)] = cdf[s_index]
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
        # Save the results to the same pickle file on each iteration
        def save_pickle(obj, path):
            with open(path, 'wb') as f:
                pickle.dump(obj, f)
        # Prevent keyboard interrupt from corrupting a pickle save
        thread = Thread(target=save_pickle, args=(series_to_plot,
            f'unami_results/{args.save_folder}/series_to_plot.pkl'))
        thread.start()
        thread.join()
        
    if args.save_folder:
        return
    with open(f'unami_results/{args.read_folder}/series_to_plot.pkl', 'rb') as f:
        series_to_plot = pickle.load(f)

    lats, lons = list(zip(*loc_list))
    def df_min(df):
        return df.min().min()
    def df_max(df):
        return df.max().max()
    for series_name, label, _min, _max in [
        ('x_inf', 'x_inf', df_min, df_max),
        ('x_sup', 'x_sup', df_min, df_max),
        ('trend', 'linear trend', lambda x: -df_max(np.abs(x)), lambda x: df_max(np.abs(x))),
        ('j2_s_3', '2-norm over 1/4 max s', lambda x: 0, lambda x: 1),
        ('j2_s_6', '2-norm over 1/2 max s', lambda x: 0, lambda x: 1),
        ('j2_s_9', '2-norm over 3/4 max s', lambda x: 0, lambda x: 1),
        ('j2_s_12', '2-norm over last s', lambda x: 0, lambda x: 1),
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
            cb_max=_max(df), size_func=lambda x: 15, cmap='RdYlBu_r')
        plt.show()

    for series_name, label, _min, _max in [
        ('x_median_last_s', 'median probability across x at final s', lambda x: 0, lambda x: 1),
        ('x_mean_last_s', 'mean probability across x at final s', lambda x: 0, lambda x: 1),
        ('x_std_last_s', 'std probability across x at final s', lambda x: 0, df_max),
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
        ('t_median_last_s', 'median probability across t at final s', lambda x: 0, lambda x: 1),
        ('t_mean_last_s', 'mean probability across t at final s', lambda x: 0, lambda x: 1),
        ('t_std_last_s', 'std probability across t at final s', lambda x: 0, df_max),
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
        
    for series_name, label, _min, _max in [
        ('cdf_3', 'cdf at 1/4 max s', lambda x: 0, lambda x: 1),
        ('cdf_6', 'cdf at 1/2 max s', lambda x: 0, lambda x: 1),
        ('cdf_9', 'cdf at 3/4 max s', lambda x: 0, lambda x: 1),
        ('cdf_12', 'cdf at last s', lambda x: 0, lambda x: 1),
        ('pdf_final', 'pdf at last s', lambda x: 0, df_max),
        ('mean', 'mean of pdf', lambda x: 0, df_max),
        ('q1', '0.25 quantile of pdf', lambda x: 0, df_max),
        ('median', 'median of pdf', lambda x: 0, df_max),
        ('q3', '0.75 quantile of pdf', lambda x: 0, df_max),
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