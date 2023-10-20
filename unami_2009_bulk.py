import os
import ast
import argparse
from dateutil.relativedelta import relativedelta
from datetime import datetime

import pandas as pd
import numpy as np
from scipy.sparse.linalg import spsolve
np.random.seed(0)

from helpers import prepare_df
from unami_2009_helpers import prepare_model_df, calculate_param_coeffs, \
    calculate_param_func, build_scheme, plot_results, get_x_domain

PERIOD = 24 * 365
BOM_DAILY_PATH = 'data_unfused/bom_daily'
FUSED_DAILY_PATH = 'data/fused_upsampled'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_folder', default=None, required=True)
    parser.add_argument('--dataset', default='fused')
    parser.add_argument('--prec_inc', type=float, default=0.2)
    parser.add_argument('--x_steps', type=int, default=30)
    parser.add_argument('--t_steps', type=int, default=24)
    parser.add_argument('--s_steps', type=int, default=100)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--delta_s', type=float, default=10)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--np_bcs', action='store_true', default=False)
    parser.add_argument('--x_quantile', type=float, default=0.005)
    parser.add_argument('--year_cycles', type=int, default=1)
    parser.add_argument('--side', default=None)
    parser.add_argument('--sar_corrected', action='store_true', default=False)
    parser.add_argument('--solution_tol', type=float, default=1)
    parser.add_argument('--solution_tol_exit', action='store_true', default=False)
    parser.add_argument('--decade', type=int, default=None)
    parser.add_argument('--window_years', type=int, default=None)
    args = parser.parse_args()
    
    suffix = 'nsrp' if args.dataset == 'fused_daily_nsrp' else 'orig'
    if args.decade == 1:
        suffix += '_d1'
    elif args.decade == 2:
        suffix += '_d2'
    full_loc_list = []
    _prec_series_list = []
    # _prec_series_list receives data sequentially by location
    if args.dataset == 'fused':
        # Select random locations from the fused dataset
        prec_df, _, _ = prepare_df('data/precipitation', 'FusedData.csv', 'prec')
        num_samples = args.num_samples if args.num_samples else 1391
        _prec_df = prec_df.T.sample(num_samples)
        for loc, row in _prec_df.iterrows():
            full_loc_list.append(loc)
            prec_series = row
            _prec_series_list.append(prec_series)
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
            _prec_series_list.append(prec_series)
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
            _prec_series_list.append(prec_series)

    prec_series_list = []
    for s in _prec_series_list:
        if args.decade == 1:
            s = s[:'2011-03-31']
        elif args.decade == 2:
            s = s['2011-04-01':]
        if args.window_years:
            start = datetime.strptime(np.datetime_as_string(s.index.values[-1])[:10], '%Y-%m-%d') - \
                relativedelta(years=args.window_years)
            s = s[start:]
        prec_series_list.append(s)
            
    n = args.x_steps
    m = args.t_steps
    z = args.s_steps
    delta_t = PERIOD / m

    # Fit the parameters then solve the DE for each prec series
    # Extract parameter values and useful statistics from the DE solution
    calculated_locations = []
    for path in os.scandir(f'unami_results/{args.save_folder}'):
        if path.is_file() and path.name.endswith('npy'):
            calculated_locations.append(path.name.split('_')[2].split('.npy')[0])
    loc_list = []
    for s, prec_series in enumerate(prec_series_list):
        loc = full_loc_list[s]
        # if loc[0] < -14.5:
        #     continue
        # Skip if data for a location has already been calculated
        if args.save_folder and str(loc) in calculated_locations:
            continue
        loc_list.append(loc)
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
            x_inf = pd.read_csv(f'corrected_x_inf_{suffix}.csv').iloc[loc_index, 0]
            x_sup = pd.read_csv(f'corrected_x_sup_{suffix}.csv').iloc[loc_index, 0]
        else:
            beta_hats = calculate_param_coeffs(model_df, PERIOD, shift_zero=True)
            x_inf, x_sup = get_x_domain(model_df['x'], quantile=args.x_quantile)
        param_func = calculate_param_func(model_df, PERIOD, beta_hats)
        x_data = np.array(model_df['x'])
        t_mesh = np.linspace(0, args.year_cycles * PERIOD, args.year_cycles * m + 1)
        scheme_output = build_scheme(param_func, t_mesh, n, m, args.delta_s,
            delta_t, args.np_bcs, x_inf=x_inf, x_sup=x_sup, side=args.side)
        if args.np_bcs:
            M_mats, G_mats, H_mats = scheme_output
        else:
            A_mat, G_mats = scheme_output
        
        print(f'({s + 1} / {len(prec_series_list)}) Solving linear systems for {str(loc)}...')
        print(f'x_inf = {round(x_inf, 3)}, x_sup = {round(x_sup, 3)}')
        year_cycles = args.year_cycles if args.np_bcs else 1
        x_size = n if args.side else n - 1
        u_array = np.zeros((z + 1, year_cycles * m, x_size))
        u_array[0, :, :] = 1  # BC of 1 at s = 0
        if args.np_bcs:
            u_array[:, 0, :] = 1  # BC of 1 at t = 0
        # Number of year cycles required for convergence
        # NaN used to indicate prior convergence wrt s, 0 for failure to converge wrt t
        t_conv = np.zeros(z + 1)
        t_conv.fill(np.nan)
        for i in range(1, z + 1):
            t_conv[i] = 0
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
                        u_array[i, y*m + j, :] = spsolve(M_mats[j], b_vec)
                    if y > 0 and args.solution_tol_exit:
                        yc_norm = (np.abs(u_array[i, (y - 1)*m : y*m, :] - \
                            u_array[i, y*m : (y + 1)*m, :])).sum(axis=(0, 1))
                        # print(i, y, round(yc_norm, 2))
                        if yc_norm < args.solution_tol:
                            t_conv[i] = y
                            for _y in range(y + 1, year_cycles):
                                u_array[i, _y*m : (_y + 1)*m, :] = u_array[i, y*m : (y + 1)*m, :]
                            break
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
            # If the solution has converged (within specified tolerance), fill it out for
            # all remaining values of s
            if i > 1 and args.solution_tol_exit:
                solution_diff = (np.abs(u_array[i, -m:, :] - u_array[i - 1, -m:, :]))\
                    .sum(axis=(0, 1))
                if solution_diff < args.solution_tol:
                    for _i in range(i + 1, z + 1):
                        u_array[_i, :, :] = u_array[i, :, :]
                    break
        if args.plot:
            plot_results(u_array, x_data, t_mesh, n, m, z, args.delta_s, delta_t,
                param_func, start_date=model_df['t'].iloc[0] if args.np_bcs else None,
                x_inf=x_inf, x_sup=x_sup, side=args.side, title=str(loc))
        np.save(f'unami_results/{args.save_folder}/u_array_{str(loc)}.npy', u_array[:, -m:, :])
        np.savetxt(f'unami_results/{args.save_folder}/t_conv_{str(loc)}.csv', t_conv, delimiter=",")
    
if __name__ == '__main__':
    main()