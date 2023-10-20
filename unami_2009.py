import argparse
from datetime import datetime
import ast

import pandas as pd
import numpy as np
from scipy.sparse.linalg import spsolve

from helpers import prepare_df
from unami_2009_helpers import prepare_model_df, calculate_param_coeffs, \
    calculate_param_func, plot_data, plot_params, \
    build_scheme, plot_results, get_x_domain

np.random.seed(0)
np.set_printoptions(suppress=True)
YEARS = range(2000, 2022)
# FUSED_SERIES_KEY = (-28.75, 153.5) # Lismore
FUSED_SERIES_KEY = (-19.25, 146.5) # Townsville
# FUSED_SERIES_KEY = (-17.75, 140.5) # NW Qld
# FUSED_SERIES_KEY = (-25.75, 133.5) # Central Australia
# FUSED_SERIES_KEY = (-26.75, 138.5) # NE SA
# FUSED_SERIES_KEY = (-37.75, 145.5) # Melbourne
# FUSED_SERIES_KEY = (-12.75, 131.5) # Darwin
# FUSED_SERIES_KEY = (-33.25, 151.5) # Central Coast
# FUSED_SERIES_KEY = (-17.75, 122.5) # Broome
# FUSED_SERIES_KEY = (-42.75, 147.5) # Hobart
# FUSED_SERIES_KEY = (-16.75, 145.5) # Cairns
# FUSED_SERIES_KEY = (-35.25, 138.5) # Adelaide
# FUSED_SERIES_KEY = (-27.75, 152.5) # Brisbane
# FUSED_SERIES_KEY = (-31.75, 116.5) # Perth
# FUSED_SERIES_KEY = (-18.25, 133.5) # Lake Woods (anomaly in fused dataset)
# FUSED_SERIES_KEY = (-22.75, 118.5) # NW WA
# FUSED_SERIES_KEY = (-28.75, 141.5) # SW Qld
# BOM_DAILY_FILE = 'BOMDaily086213_rosebud'
# BOM_DAILY_FILE = 'BOMDaily033250_mid_qld_coast'
# BOM_DAILY_FILE = 'BOMDaily009930_albany'
# BOM_DAILY_FILE = 'BOMDaily051043_desert_nsw'
# BOM_DAILY_FILE = 'BOMDaily001026_northern_wa'
# BOM_DAILY_FILE = 'BOMDaily003084_northern_wa'
BOM_DAILY_FILE = 'BOMDaily014163_darwin'

# t_data is in hours, so set period as one year accordingly
PERIOD = 24 * 365

# s is in units of mm
# t is in units of hours
# x is in units of log(mm/h)
# In paper, prec_inc (delta) = 0.2 mm, delta_s = 0.02 mm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='fused')
    parser.add_argument('--prec_inc', type=float, default=0.2)
    parser.add_argument('--x_steps', type=int, default=30)
    parser.add_argument('--t_steps', type=int, default=24)
    parser.add_argument('--s_steps', type=int, default=100)
    parser.add_argument('--delta_s', type=float, default=10)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--deseasonalise', action='store_true', default=False)
    parser.add_argument('--np_bcs', action='store_true', default=False)
    parser.add_argument('--x_proportion', type=float, default=None)
    parser.add_argument('--x_quantile', type=float, default=None)
    parser.add_argument('--x_mixed', action='store_true', default=False)
    parser.add_argument('--decade', type=int, default=None)
    parser.add_argument('--year_cycles', type=int, default=1)
    parser.add_argument('--side', default=None)
    parser.add_argument('--trend_polynomial', type=int, default=None)
    parser.add_argument('--sar_corrected', action='store_true', default=False)
    parser.add_argument('--end_date', default=None)
    parser.add_argument('--lat', type=float, default=None)
    parser.add_argument('--lon', type=float, default=None)
    args = parser.parse_args()

    loc = (args.lat, args.lon) if args.lat and args.lon else FUSED_SERIES_KEY
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
    elif args.dataset == 'bom_daily':
        prec_df = pd.read_csv(f'data_unfused/{BOM_DAILY_FILE}.csv')
        prec_df.index = pd.DatetimeIndex(prec_df['Date'])
        prec_series = pd.Series(prec_df['Rain']).dropna().loc['2000-04-01':]
    elif args.dataset == 'test':
        prec_series = pd.read_csv(f'data_unfused/test_data.csv', index_col=0, header=None)
        prec_series = pd.Series(prec_series.values[:, 0], index=pd.DatetimeIndex(prec_series.index))
    if args.decade == 1:
        prec_series = prec_series[:'2011-03-31']
    elif args.decade == 2:
        prec_series = prec_series['2011-04-01':]
        # prec_series = prec_series['2019-04-01':]
    if args.deseasonalise:
        ds_period = 12 if args.dataset == 'fused' else 365
    else:
        ds_period = None
    if args.end_date:
        prec_series = prec_series[:args.end_date]
    model_df = prepare_model_df(prec_series, args.prec_inc, ds_period)
    
    n = args.x_steps
    m = args.t_steps
    z = args.s_steps
    x_data = np.array(model_df['x'])
    if args.x_mixed:
        lower_q = pd.read_csv(f'x_lower_quantiles_{suffix}.csv', index_col=0)\
            .loc[str(loc)][0]
        upper_q = pd.read_csv(f'x_upper_quantiles_{suffix}.csv', index_col=0)\
            .loc[str(loc)][0]
    else:
        lower_q, upper_q = None, None
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
        x_inf = pd.read_csv(f'corrected_x_inf_{suffix}.csv').iloc[loc_index, 0]
        x_sup = pd.read_csv(f'corrected_x_sup_{suffix}.csv').iloc[loc_index, 0]
    else:
        beta_hats = calculate_param_coeffs(model_df, PERIOD, shift_zero=True)
        x_inf, x_sup = get_x_domain(model_df['x'], args.x_proportion,
            args.x_quantile, lower_q, upper_q)
    param_func = calculate_param_func(model_df, PERIOD, beta_hats,
        trend_polynomial=args.trend_polynomial)
    if args.plot:
        plot_data(model_df, prec_series, x_inf, x_sup)
        t_mesh = np.arange('2000', '2001', dtype='datetime64[D]')
        plot_params(model_df, t_mesh, param_func, x_inf, x_sup)
        
    year_cycles = args.year_cycles if args.np_bcs else 1
    t_mesh = np.linspace(0, year_cycles * PERIOD, year_cycles * m + 1)
    delta_t = PERIOD / m
    scheme_output = build_scheme(param_func, t_mesh, n, m, args.delta_s,
        delta_t, args.np_bcs, x_inf=x_inf, x_sup=x_sup, side=args.side)
    if args.np_bcs:
        M_mats, G_mats, H_mats = scheme_output
    else:
        A_mat, G_mats, R_vecs = scheme_output

    start_time = datetime.now()
    print('Solving linear systems:')
    # x_size = n if args.side else n - 1
    x_size = n - 1
    u_array = np.zeros((z + 1, year_cycles * m, x_size))
    if args.side == 'left':
        q_func = lambda x: (x - x_inf) / (x_sup - x_inf)
    else:
        q_func = lambda x: (x_sup - x) / (x_sup - x_inf)
    u_array[0, :, :] = 1  # IC of 1 at s = 0
    # Modify this IC by subtracting the linear function which satisfies
    # the inhomogenous BCs in x
    # if args.side:
    #     x_mesh = np.linspace(x_inf, x_sup, n + 1)
    #     for k in range(x_size):
    #         u_array[0, :, k] -= q_func(x_mesh[k + 1]) * \
    #             np.ones(year_cycles * m)
    # print(u_array[0, :, :])
    if args.np_bcs:
        u_array[:, 0, :] = 1  # IC of 1 at t = 0
    for i in range(1, z + 1):
        if i % 10 == 0:
            print(i, '/', z)
        if args.np_bcs:
            # Solve iteratively for each s and t
            for y in range(year_cycles):
                for j in range(m):
                    if j == 0 and y == 0:
                        continue
                    b_vec = G_mats[j] @ u_array[i - 1, y*m + j, :] + \
                        H_mats[j] @ u_array[i, y*m + j - 1, :]
                    if args.side:
                        b_vec -= R_vecs[j]
                    u_array[i, y*m + j, :] = spsolve(M_mats[j], b_vec)
                if y == 0:
                    continue
                yc_norm = (np.abs(u_array[i, (y - 1)*m : y*m, :] - \
                    u_array[i, y*m : (y + 1)*m, :])).sum(axis=(0, 1))
                if yc_norm < 0.1:
                    for _y in range(y + 1, year_cycles):
                        u_array[i, _y*m : (_y + 1)*m, :] = u_array[i, y*m : (y + 1)*m, :]
                    break
                # if i == 3 and j == 1:
                #     np.savetxt('unami_test_outputs/M_mats_3.csv', M_mats[t_index], delimiter=',')
                #     np.savetxt('unami_test_outputs/b_vec_3_np.csv', b_vec, delimiter=',')
                #     np.savetxt('unami_test_outputs/u_array_3_np.csv', u_array[i, :, :], delimiter=',')
                #     exit()
        else:
            # Solve iteratively for each s
            b_vec = np.zeros((m * x_size, 1))
            for j in range(m):
                start = j * x_size
                end = (j + 1) * x_size
                b_vec[start : end] = G_mats[j] @ u_array[i - 1, j, :].reshape((x_size, 1))
                if args.side:
                    b_vec[start : end] -= R_vecs[j].reshape((x_size, 1))
            u_vec = spsolve(A_mat, b_vec)
            for j in range(m):
                start = j * x_size
                end = (j + 1) * x_size
                u_array[i, j, :] = u_vec[start : end]
            # if (i + 1) % 10 == 0:
            #     print(i + 1, u_vec.mean(), np.median(u_vec), u_vec.std(), u_vec.max(), u_vec.min())
            # if i == 3:
            #     np.savetxt('unami_test_outputs/A_mat_3.csv', A_mat, delimiter=',')
            #     np.savetxt('unami_test_outputs/b_vec_3.csv', b_vec.reshape((m, x_size)), delimiter=',')
            #     np.savetxt('unami_test_outputs/u_array_3.csv', u_array[i, :, :], delimiter=',')
            #     exit()
        # If the solution has converged (within specified tolerance), fill it out for
        # all remaining values of s
        # if i > 1:
        #     solution_diff = (np.abs(u_array[i, -m:, :] - u_array[i - 1, -m:, :]))\
        #         .sum(axis=(0, 1))
        #     j1 = np.abs(u_array[i, -m:, :]).sum(axis=(0, 1)) / (n - 1) / m
        #     # print(i, round(solution_diff, 2), round(j1, 2), y)
        #     if solution_diff < 0.1:
        #         for _i in range(i + 1, z + 1):
        #             u_array[_i, :, :] = u_array[i, :, :]
        #         break
    # For one-sided solution, add value of split-off linear function q(x)
    # if args.side:
    #     for k in range(x_size):
    #         u_array[:, :, k] += q_func(x_mesh[k + 1]) * \
    #             np.ones((z + 1, year_cycles * m))
    print(f'Solving time: {datetime.now() - start_time}')
    plot_results(u_array, x_data, t_mesh, n, m, z, args.delta_s, delta_t,
        param_func, start_date=None, non_periodic=False, side=args.side,
        x_inf=x_inf, x_sup=x_sup)

if __name__ == '__main__':
    main()