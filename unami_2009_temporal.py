import argparse
from datetime import datetime
from math import floor
import ast

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

from helpers import prepare_df
from unami_2009_helpers import prepare_model_df, calculate_param_coeffs, \
    calculate_param_func, build_scheme, plot_results, get_x_domain

np.random.seed(0)
np.set_printoptions(suppress=True)
YEARS = range(2000, 2022)
FUSED_SERIES_KEY = (-28.75, 153.5) # Lismore
# FUSED_SERIES_KEY = (-17.75, 140.5) # NW Qld
# FUSED_SERIES_KEY = (-25.75, 133.5) # Central Australia
# FUSED_SERIES_KEY = (-37.75, 145.5) # Melbourne
# FUSED_SERIES_KEY = (-12.75, 131.5) # Darwin
# FUSED_SERIES_KEY = (-18.25, 133.5) # Lake Woods (anomaly in fused dataset)
# FUSED_SERIES_KEY = (-33.25, 151.5) # Central Coast
# FUSED_SERIES_KEY = (-17.75, 122.5) # Broome
# FUSED_SERIES_KEY = (-42.75, 147.5) # Hobart
# FUSED_SERIES_KEY = (-16.75, 145.5) # Cairns

# t_data is in hours, so set period as one year accordingly
PERIOD = 24 * 365

# TIME_BLOCKS = [
#     ('2000-04-01', '2011-03-31', 'g'),
#     ('2011-04-01', '2022-03-31', 'r'),
# ]
# TIME_BLOCKS = [
#     ('2000-04-01', '2005-09-30', 'g'),
#     ('2005-10-01', '2011-03-31', 'b'),
#     ('2011-04-01', '2016-09-30', 'm'),
#     ('2016-10-01', '2022-03-31', 'r'),
# ]
# TIME_BLOCKS = [
#     ('2000-04-01', '2004-04-01', 'g'),
#     ('2004-04-01', '2008-04-01', 'b'),
#     ('2008-04-01', '2012-04-01', 'm'),
#     ('2012-04-01', '2016-04-01', 'r'),
#     ('2016-04-01', '2020-04-01', 'k'),
#     ('2020-04-01', '2022-05-01', 'c'),
# ]
TIME_BLOCKS = [
    ('2000-04-01', '2002-04-01', None),
    ('2002-04-01', '2004-04-01', None),
    ('2004-04-01', '2006-04-01', None),
    ('2006-04-01', '2008-04-01', None),
    ('2008-04-01', '2010-04-01', None),
    ('2010-04-01', '2012-04-01', None),
    ('2012-04-01', '2014-04-01', None),
    ('2014-04-01', '2016-04-01', None),
    ('2016-04-01', '2018-04-01', None),
    ('2018-04-01', '2020-04-01', None),
    ('2020-04-01', '2022-05-01', None),
]
# TIME_BLOCKS = [(f'{y}-04-01', f'{y + 1}-04-01', None) for y in range(2000, 2022)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='fused')
    parser.add_argument('--prec_inc', type=float, default=0.5)
    parser.add_argument('--x_steps', type=int, default=50)
    parser.add_argument('--t_steps', type=int, default=100)
    parser.add_argument('--s_steps', type=int, default=100)
    parser.add_argument('--delta_s', type=float, default=10)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--np_bcs', action='store_true', default=False)
    parser.add_argument('--shrink_x_proportion', type=float, default=None)
    parser.add_argument('--shrink_x_quantile', type=float, default=None)
    parser.add_argument('--shrink_x_mixed', action='store_true', default=False)
    parser.add_argument('--year_cycles', type=int, default=1)
    parser.add_argument('--sar_corrected', action='store_true', default=False)
    args = parser.parse_args()

    if args.dataset == 'fused':
        prec_df, _, _ = prepare_df('data/precipitation', 'FusedData.csv', 'prec')
        prec_series_full = pd.Series(prec_df[FUSED_SERIES_KEY], index=pd.DatetimeIndex(prec_df.index))
    elif args.dataset == 'fused_daily':
        prec_df = pd.read_csv(f'data/fused_upsampled/fused_daily_'
            f'{FUSED_SERIES_KEY[0]}_{FUSED_SERIES_KEY[1]}_it_3000.csv', index_col=0)
        prec_series_full = pd.Series(prec_df.values[:, 0], index=pd.DatetimeIndex(prec_df.index))
    elif args.dataset == 'fused_daily_nsrp':
        prec_df = pd.read_csv(f'data/fused_upsampled/fused_daily_nsrp_'
            f'{FUSED_SERIES_KEY[0]}_{FUSED_SERIES_KEY[1]}.csv', index_col=0)
        prec_series_full = pd.Series(prec_df.values[:, 0], index=pd.DatetimeIndex(prec_df.index))
    elif args.dataset == 'test':
        prec_df = pd.read_csv(f'data_unfused/test_data.csv', index_col=0, header=None)
        prec_series_full = pd.Series(prec_df.values[:, 0], index=pd.DatetimeIndex(prec_df.index))
        
    model_dfs = [prepare_model_df(prec_series_full[b[0] : b[1]], args.prec_inc)
        for b in TIME_BLOCKS]
    x_max_all = np.max([df['x'].max() for df in model_dfs])
    x_min_all = np.min([df['x'].min() for df in model_dfs])
    param_funcs = []
    for df in model_dfs:
        if args.sar_corrected:
            df_suffix = 'nsrp' if args.dataset == 'fused_daily_nsrp' else 'orig'
            beta_coeffs_df = pd.read_csv(f'beta_coeffs_fused_daily_{df_suffix}.csv',
                index_col=0, converters={0: ast.literal_eval})
            loc_index = list(beta_coeffs_df.index.values).index(FUSED_SERIES_KEY)
            beta_hats = {
                'beta': pd.read_csv(f'corrected_beta_coeffs_{df_suffix}.csv')\
                    .iloc[loc_index, :].values,
                'kappa': pd.read_csv(f'corrected_kappa_coeffs_{df_suffix}.csv')\
                    .iloc[loc_index, :].values,
                'psi': pd.read_csv(f'corrected_psi_coeffs_{df_suffix}.csv')\
                    .iloc[loc_index, :].values,
            }
        else:
            beta_hats = calculate_param_coeffs(df, PERIOD, shift_zero=True)
        param_funcs.append(calculate_param_func(df, PERIOD, beta_hats))
    x_medians = [df['x'].median() for df in model_dfs]

    year_t_vec = np.arange('2000', '2001', dtype='datetime64[D]')
    blocks_t_vec = [datetime.strptime(b[0], '%Y-%m-%d') for b in TIME_BLOCKS]
    months = [1, 3, 5, 7, 9, 11]
    colours = ['b', 'm', 'r', 'y', 'g', 'c']
    if args.plot:
        figure, axes = plt.subplots(len(TIME_BLOCKS), 1)
        axes = iter(axes.flatten())
        for df in model_dfs:
            x_inf, x_sup = get_x_domain(df['x'], args.shrink_x_proportion,
                args.shrink_x_quantile, FUSED_SERIES_KEY if args.shrink_x_mixed else None)
            x_median = df['x'].median()
            axis = next(axes)
            axis.plot(df['s'], df['x'], 'ko-')
            axis.plot(df['s'], df['x'] * 0 + x_sup, 'c-')
            axis.plot(df['s'], df['x'] * 0 + x_inf, 'c-')
            axis.plot(df['s'], [x_median for _ in df['x']], 'r-')
            axis.set_xlabel('s')
            axis.set_ylabel('x')
            axis.set_ylim([x_min_all, x_max_all])
        plt.show()

        # For each time block, plot resulting parameter curves on the same plot
        figure, axes = plt.subplots(3, 1)
        colour_values = np.linspace(0, 255, len(TIME_BLOCKS)) / 255
        for i, b in enumerate(TIME_BLOCKS):
            param_func = param_funcs[i]
            x_median = x_medians[i]
            colour = b[2] or (colour_values[i], 0, 0)
            axes[0].plot(year_t_vec, [param_func('beta', j * 24) for j in range(len(year_t_vec))],
                color=colour, linestyle='dashed', label=f'{b[0]} to {b[1]}')
            axes[1].plot(year_t_vec, [param_func('kappa', j * 24) for j in range(len(year_t_vec))],
                color=colour, linestyle='dashed')
            axes[2].plot(year_t_vec, [param_func('psi', j * 24, x_median) for j in range(len(year_t_vec))],
                color=colour, linestyle='dashed')
        axes[0].set_xlabel('t')
        axes[0].set_ylabel('beta')
        axes[0].legend()
        axes[1].set_xlabel('t')
        axes[1].set_ylabel('kappa = log(K)')
        axes[2].set_xlabel('t')
        axes[2].set_ylabel('psi = log(v) at x median')
        plt.show()

        figure, axes = plt.subplots(3, 1)
        axes = iter(axes.T.flatten())
        axis = next(axes)
        for month, c in zip(months, colours):
            t = datetime(2000, month, 1)
            hours = 24 * (t - datetime(2000, 1, 1)).days
            month_str = t.strftime('%b')
            axis.plot(blocks_t_vec, [f('beta', hours) for f in param_funcs],
                color=c, linestyle='dashed', label=month_str)
        axis.set_title('beta')
        axis.legend()

        axis = next(axes)
        for month, c in zip(months, colours):
            t = datetime(2000, month, 1)
            hours = 24 * (t - datetime(2000, 1, 1)).days
            month_str = t.strftime('%b')
            axis.plot(blocks_t_vec, [f('kappa', hours) for f in param_funcs],
                color=c, linestyle='dashed', label=month_str)
        axis.set_title('kappa')
        axis.legend()

        axis = next(axes)
        for month, c in zip(months, colours):
            t = datetime(2000, month, 1)
            hours = 24 * (t - datetime(2000, 1, 1)).days
            month_str = t.strftime('%b')
            axis.plot(blocks_t_vec, [f('beta', hours, x) for f, x in zip(param_funcs, x_medians)],
                color=c, linestyle='dashed', label=month_str)
        axis.set_title('psi at x_median')
        axis.legend()
        plt.show()
        
    n = args.x_steps
    m = args.t_steps
    z = args.s_steps
    s_mesh = np.linspace(0, z * args.delta_s, z + 1)
    # TODO: Find a good way to compare DE solutions for different time blocks
    x_indices = [
        floor(n / 10),
        floor(n / 5),
        floor(n / 2),
        floor(9*n / 10),
    ]
    last_cdf = np.zeros((len(TIME_BLOCKS), 12, len(x_indices)))
    mean = np.zeros((len(TIME_BLOCKS), 12, len(x_indices)))
    median = np.zeros((len(TIME_BLOCKS), 12, len(x_indices)))
    mode = np.zeros((len(TIME_BLOCKS), 12, len(x_indices)))
    start_time = datetime.now()
    for block_i, b in enumerate(TIME_BLOCKS):
        x_data = np.array(model_dfs[block_i]['x'])
        x_inf, x_sup = get_x_domain(x_data, args.shrink_x_proportion,
            args.shrink_x_quantile, FUSED_SERIES_KEY if args.shrink_x_mixed else None)
        param_func = param_funcs[block_i]
        t_mesh = np.linspace(0, args.year_cycles * PERIOD, args.year_cycles * m + 1)
        delta_t = PERIOD / m
        scheme_output = build_scheme(param_func, t_mesh, n, m, args.delta_s,
            delta_t, non_periodic=args.np_bcs, x_inf=x_inf, x_sup=x_sup)
        if args.np_bcs:
            M_mats, G_mats, H_mats = scheme_output
        else:
            A_mat, G_mats = scheme_output

        print(f'Time block {b[0]} to {b[1]}: solving linear systems:')
        if args.np_bcs:
            u_array = np.zeros((z + 1, args.year_cycles * m, n - 1))
            u_array[0, :, :] = 1
            u_array[:, 0, :] = 1
            # Solve iteratively for each s and t
            for i in range(1, z + 1):
                if i % 10 == 0:
                    print(i, '/', z)
                for j in range(1, args.year_cycles * m):
                    b_vec = None
                    t_index = j
                    while b_vec is None:
                        try:
                            b_vec = G_mats[t_index] @ u_array[i - 1, j, :] + H_mats[t_index] @ u_array[i, j - 1, :]
                            u_array[i, j, :] = spsolve(M_mats[t_index], b_vec)
                        except IndexError:
                            t_index -= m
        else:
            u_vecs = [np.ones(m * (n - 1))]
            # Solve iteratively for each s
            for i in range(z):
                if i % 10 == 0:
                    print(i, '/', z)
                b_vec = np.zeros((m * (n - 1), 1))
                for j in range(m):
                    start = j*(n - 1)
                    end = start + n - 1
                    b_vec[start : end] = (G_mats[j] @ u_vecs[i][start : end]).reshape((n - 1, 1))
                u_vec = spsolve(A_mat, b_vec)
                u_vecs.append(u_vec)
            u_array = np.stack(u_vecs, axis=0).reshape((z + 1, m, n - 1))
        if args.plot:
            plot_results(u_array, x_data, t_mesh, n, m, z, args.delta_s, delta_t, param_func,
                x_inf=x_inf, x_sup=x_sup)
            exit()
        
        u_last_cycle = u_array[:, -m:, :]
        for j in range(12):
            for k, x_index in enumerate(x_indices):
                t_index = floor(j * m / 12)
                cdf = 1 - u_last_cycle[:, t_index, x_index]
                pdf = [(cdf[i + 1] - cdf[i]) / args.delta_s for i in range(len(cdf) - 1)]
                last_cdf[block_i, j, k] = cdf[-1]
                mean[block_i, j, k] = None if cdf[-1] == 0 else \
                    np.sum([s_mesh[i] * p * args.delta_s / cdf[-1] for i, p in enumerate(pdf)])
                median[block_i, j, k] = None if cdf[-1] < 0.5 \
                    else np.argwhere(cdf >= 0.5)[0, 0] * args.delta_s
                mode[block_i, j, k] = np.argmax(pdf) * args.delta_s
    print(f'Solving time: {datetime.now() - start_time}')
    
    figure, axes = plt.subplots(len(months), 1)
    axes = iter(axes.T.flatten())
    for month in months:
        axis = next(axes)
        for k, x_index in enumerate(x_indices):
            month_str = datetime(2000, month, 1).strftime('%b')
            axis.plot(blocks_t_vec, [last_cdf[i, month - 1, k] for i in range(len(blocks_t_vec))],
                color=colours[k], linestyle='dashed',
                label=f'x = x_inf + {x_index} * delta_x')
        axis.set_title(f'CDF at s = {args.delta_s * z}, t = 1 {month_str}')
        if month == 1:
            axis.legend()
    plt.show()
    
    figure, axes = plt.subplots(len(months), 1)
    axes = iter(axes.T.flatten())
    for month in months:
        axis = next(axes)
        for k, x_index in enumerate(x_indices):
            month_str = datetime(2000, month, 1).strftime('%b')
            axis.plot(blocks_t_vec, [mean[i, month - 1, k] for i in range(len(blocks_t_vec))],
                color=colours[k], linestyle='dashed',
                label=f'x = x_inf + {x_index} * delta_x')
        axis.set_title(f'PDF mean for 1 {month_str}')
        if month == 1:
            axis.legend()
    plt.show()
    
    figure, axes = plt.subplots(len(months), 1)
    axes = iter(axes.T.flatten())
    for month in months:
        axis = next(axes)
        for k, x_index in enumerate(x_indices):
            month_str = datetime(2000, month, 1).strftime('%b')
            axis.plot(blocks_t_vec, [median[i, month - 1, k] for i in range(len(blocks_t_vec))],
                color=colours[k], linestyle='dashed',
                label=f'x = x_inf + {x_index} * delta_x')
        axis.set_title(f'PDF median for 1 {month_str}')
        if month == 1:
            axis.legend()
    plt.show()
    
    figure, axes = plt.subplots(len(months), 1)
    axes = iter(axes.T.flatten())
    for month in months:
        axis = next(axes)
        for k, x_index in enumerate(x_indices):
            month_str = datetime(2000, month, 1).strftime('%b')
            axis.plot(blocks_t_vec, [mode[i, month - 1, k] for i in range(len(blocks_t_vec))],
                color=colours[k], linestyle='dashed',
                label=f'x = x_inf + {x_index} * delta_x')
        axis.set_title(f'PDF mode for 1 {month_str}')
        if month == 1:
            axis.legend()
    plt.show()
    
    # TODO: Repeat the above but across many locations, showing results on a map

if __name__ == '__main__':
    main()