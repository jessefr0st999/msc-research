import argparse
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import ast
from math import floor

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

from helpers import prepare_df
from unami_2009_helpers import prepare_model_df, calculate_param_coeffs, \
    calculate_param_func, plot_data, plot_params, \
    build_scheme, plot_results, get_x_domain

np.random.seed(0)
np.set_printoptions(suppress=True)
PERIOD = 24 * 365

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='fused')
    parser.add_argument('--prec_inc', type=float, default=0.5)
    parser.add_argument('--x_steps', type=int, default=30)
    parser.add_argument('--t_steps', type=int, default=24)
    parser.add_argument('--s_steps', type=int, default=100)
    parser.add_argument('--delta_s', type=float, default=10)
    parser.add_argument('--year_cycles', type=int, default=1)
    parser.add_argument('--window_years', type=int, default=5)
    parser.add_argument('--sar_corrected', action='store_true', default=False)
    parser.add_argument('--start_date', default='2000-04-01')
    parser.add_argument('--end_date', default='2022-04-01')
    parser.add_argument('--target_date')
    parser.add_argument('--lat', type=float, required=True)
    parser.add_argument('--lon', type=float, required=True)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--solution_tol', type=float, default=0.1)
    args = parser.parse_args()

    loc = (args.lat, args.lon)
    suffix = 'nsrp' if args.dataset == 'fused_daily_nsrp' else 'orig'
    if args.dataset == 'fused':
        prec_df, _, _ = prepare_df('data/precipitation', 'FusedData.csv', 'prec')
        prec_series = pd.Series(prec_df[loc], index=pd.DatetimeIndex(prec_df.index))
    elif args.dataset == 'fused_daily':
        prec_series = pd.read_csv(f'data/fused_upsampled/fused_daily_'
            f'{loc[0]}_{loc[1]}_it_3000.csv', index_col=0)
        prec_series = pd.Series(prec_series.values[:, 0], index=pd.DatetimeIndex(prec_series.index))
    elif args.dataset == 'fused_daily_nsrp':
        prec_series = pd.read_csv(f'data/fused_upsampled/fused_daily_nsrp_'
            f'{loc[0]}_{loc[1]}.csv', index_col=0)
        prec_series = pd.Series(prec_series.values[:, 0], index=pd.DatetimeIndex(prec_series.index))
    
    n = args.x_steps
    m = args.t_steps
    z = args.s_steps
    delta_t = PERIOD / m

    date_spans = []
    end = datetime.strptime(args.end_date, '%Y-%m-%d')
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date \
        else datetime(2000, 4, 1)
    while end > start_date + relativedelta(years=args.window_years):
        start = end - relativedelta(years=args.window_years)
        date_spans.append((start, end))
        end -= relativedelta(months=2) if args.target_date else relativedelta(years=1)
        
    t_indices = [floor(i * m / 12) for i in range(12)]
    x_indices = [
        floor(n / 10),
        floor(n / 5),
        floor(n / 2),
        floor(4*n / 5),
        floor(9*n / 10),
    ]
    s_indices = [5, 10, 25, 50, 100]
    s_mesh = np.linspace(0, z * args.delta_s, z + 1)
    series_to_plot = {key: list() for key in ['conv_s', 'mean', 'median',
        'std', 'q1', 'q3']}
    for s in s_indices:
        series_to_plot[f'j1_s_{s}'] = list()
        series_to_plot[f'cdf_s_{s}'] = list()

    for start, end in date_spans:
        model_df = prepare_model_df(prec_series[start : end], args.prec_inc)
        x_data = np.array(model_df['x'])
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
        # Just use a quantile-based x-domain since the custom one is fitted using
        # the whole time series
        x_inf, x_sup = get_x_domain(model_df['x'], quantile=0.025)
        if args.plot:
            plot_data(model_df, prec_series, x_inf, x_sup)
            t_mesh = np.arange('2000', '2001', dtype='datetime64[D]')
            plot_params(model_df, t_mesh, param_func, x_inf, x_sup)
        t_mesh = np.linspace(0, args.year_cycles * PERIOD, args.year_cycles * m + 1)
        scheme_output = build_scheme(param_func, t_mesh, n, m, args.delta_s,
            delta_t, non_periodic=True, x_inf=x_inf, x_sup=x_sup)
        M_mats, G_mats, H_mats = scheme_output
        start_time = datetime.now()
        print(f'{start} to {end}: solving linear systems:')
        x_size = n - 1
        u_array = np.zeros((z + 1, args.year_cycles * m, x_size))
        u_array[0, :, :] = 1  # IC of 1 at s = 0
        u_array[:, 0, :] = 1  # IC of 1 at t = 0
        for i in range(1, z + 1):
            if i % 10 == 0:
                print(i, '/', z)
            # Solve iteratively for each s and t
            for y in range(args.year_cycles):
                for j in range(m):
                    if j == 0 and y == 0:
                        continue
                    b_vec = G_mats[j] @ u_array[i - 1, y*m + j, :] + \
                        H_mats[j] @ u_array[i, y*m + j - 1, :]
                    u_array[i, y*m + j, :] = spsolve(M_mats[j], b_vec)
                if y == 0:
                    continue
                yc_norm = (np.abs(u_array[i, (y - 1)*m : y*m, :] - \
                    u_array[i, y*m : (y + 1)*m, :])).sum(axis=(0, 1))
                if yc_norm < 0.1:
                    for _y in range(y + 1, args.year_cycles):
                        u_array[i, _y*m : (_y + 1)*m, :] = u_array[i, y*m : (y + 1)*m, :]
                    break
            # If the solution has converged (within specified tolerance), fill it out for
            # all remaining values of s
            if i > 1:
                solution_diff = (np.abs(u_array[i, -m:, :] - u_array[i - 1, -m:, :]))\
                    .sum(axis=(0, 1))
                j1 = np.abs(u_array[i, -m:, :]).sum(axis=(0, 1)) / (n - 1) / m
                # print(i, round(solution_diff, 2), round(j1, 2), y)
                if solution_diff < 0.1:
                    for _i in range(i + 1, z + 1):
                        u_array[_i, :, :] = u_array[i, :, :]
                    break
        print(f'Solving time: {datetime.now() - start_time}')
        if args.plot:
            plot_results(u_array, x_data, t_mesh, n, m, z, args.delta_s, delta_t,
                param_func, start_date=None, x_inf=x_inf, x_sup=x_sup)
        
        u_array = u_array[:, -m:, :]
        current_series = {key: dict() for key in series_to_plot}
        for s_index in s_indices:
            current_series[f'j1_s_{s_index}'] = np.abs(u_array[s_index, :, :])\
                .sum(axis=(0, 1)) / (n - 1) / m
        current_series['conv_s'] = None
        for i in range(1, z):
            solution_diff = (np.abs(u_array[i, :, :] - u_array[i - 1, :, :]))\
                .sum(axis=(0, 1))
            if solution_diff < args.solution_tol:
                current_series['conv_s'] = i * args.delta_s
                break
        for j in t_indices:
            for k in x_indices:
                cdf = 1 - u_array[:, j, k]
                pdf = [(cdf[i + 1] - cdf[i]) / args.delta_s for i in range(len(cdf) - 1)]
                for s_index in s_indices:
                    current_series[f'cdf_s_{s_index}'][(j, k)] = cdf[s_index]
                mean = np.sum([s_mesh[i] * p * args.delta_s for i, p in enumerate(pdf)])
                std = (np.sum([s_mesh[i]**2 * p * args.delta_s for i, p in enumerate(pdf)]) \
                    - mean**2) ** 0.5
                current_series['mean'][(j, k)] = None if cdf[-1] == 0 else mean
                current_series['std'][(j, k)] = None if cdf[-1] == 0 else std
                current_series['q1'][(j, k)] = None if cdf[-1] < 0.16 else \
                    np.argwhere(cdf >= 0.16)[0, 0] * args.delta_s
                current_series['median'][(j, k)] = None if cdf[-1] < 0.5 else \
                    np.argwhere(cdf >= 0.5)[0, 0] * args.delta_s
                current_series['q3'][(j, k)] = None if cdf[-1] < 0.84 else \
                    np.argwhere(cdf >= 0.84)[0, 0] * args.delta_s
        for key in series_to_plot:
            series_to_plot[key].append(current_series[key])
                
    # Plot mean, median etc. as a time series
    end_dates = [span[1] for span in date_spans]
    figure, axes = plt.subplots(2, 3)
    axes = iter(axes.flatten())
    for series_name in [*[f'j1_s_{s}' for s in s_indices], 'conv_s']:
        axis = next(axes)
        axis.plot(end_dates, series_to_plot[series_name])
        if series_name.startswith('j1_s_'):
            axis.set_ylim([-0.05, 1.05])
        axis.set_title(series_name)
    plt.show()

    # Plot remaining cumulative rainfall until specified target_date, then
    # compare against mean and median at each end_date
    if args.target_date:
        target_date = datetime.strptime(args.target_date, '%Y-%m-%d')
        cum_prec = prec_series[end_dates[-1] : target_date].cumsum()
        figure, axis = plt.subplots(1)
        axis.plot(cum_prec[-1] - cum_prec, color='k')
        for j, end_date in enumerate(end_dates):
            month = end_date.month
            axis.plot(end_date, cum_prec[-1] - cum_prec[end_date], 'ko',
                label='actual' if j == 0 else None)
            mean = series_to_plot['mean'][j][(floor((month - 1) / 12), floor(n / 2))]
            std = series_to_plot['std'][j][(floor((month - 1) / 12), floor(n / 2))]
            q1 = series_to_plot['q1'][j][(floor((month - 1) / 12), floor(n / 2))]
            median = series_to_plot['median'][j][(floor((month - 1) / 12), floor(n / 2))]
            q3 = series_to_plot['q3'][j][(floor((month - 1) / 12), floor(n / 2))]
            axis.errorbar(end_date - timedelta(days=3), mean, fmt='ro',
                label='mean +/- std' if j == 0 else None,
                yerr=0.5 * std)
            axis.errorbar(end_date + timedelta(days=3), median, fmt='bo',
                label='median +/- 0.34 quantile' if j == 0 else None,
                yerr=np.array([median - q1, q3 - median]).reshape((2, 1)))
        axis.legend()
        plt.show()

    std_df = pd.DataFrame(series_to_plot['std'], index=end_dates)
    q1_df = pd.DataFrame(series_to_plot['q1'], index=end_dates)
    q3_df = pd.DataFrame(series_to_plot['q3'], index=end_dates)
    for series_name in ['mean', 'median', *[f'cdf_s_{s}' for s in s_indices]]:
        # For each time, plot the results on a different axes
        # Plot different x-values on the same axis
        df = pd.DataFrame(series_to_plot[series_name], index=end_dates)
        figure, axes = plt.subplots(3, 4, layout='compressed')
        axes = iter(axes.flatten())
        prev_j = None
        for j, k in df.columns:
            if j != prev_j:
                axis = next(axes)
                prev_j = j
            for _k, colour in zip(x_indices, ['r', 'm', 'b', 'c', 'g']):
                if _k != k:
                    continue
                if series_name == 'mean' and _k == floor(n / 2):
                    axis.errorbar(end_dates, df[(j, k)], color=colour,
                        label=f'x = x_inf + {k} * delta_x',
                        yerr=0.5 * std_df[(j, k)])
                elif series_name == 'median' and _k == floor(n / 2):
                    error_size = [df[(j, k)] - q1_df[(j, k)], q3_df[(j, k)] - df[(j, k)]]
                    axis.errorbar(end_dates, df[(j, k)], color=colour,
                        label=f'x = x_inf + {k} * delta_x',
                        yerr=np.array(error_size).reshape((2, len(end_dates))))
                else:
                    axis.plot(end_dates, df[(j, k)], color=colour,
                        label=f'x = x_inf + {k} * delta_x')
                if series_name.startswith('cdf_s_'):
                    axis.set_ylim([-0.05, 1.05])
                days = int(j) * delta_t / 24
                date_part = (datetime(2000, 1, 1) + timedelta(days=days)).strftime('%b %d')
                axis.set_title(f'{series_name} at t = {date_part}')
            if j == t_indices[0]:
                axis.legend()
        plt.show()

if __name__ == '__main__':
    main()