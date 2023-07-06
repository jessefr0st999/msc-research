import argparse
from datetime import datetime

import pandas as pd
import numpy as np
from scipy.sparse.linalg import spsolve

from helpers import prepare_df
from unami_2009_helpers import prepare_model_df, estimate_params, plot_data, plot_params, \
    build_scheme, build_scheme_time_indep, plot_results, plot_results_time_indep, get_x_domain

np.random.seed(0)
np.set_printoptions(suppress=True)
YEARS = range(2000, 2022)
# FUSED_SERIES_KEY = (-28.75, 153.5) # Lismore
# FUSED_SERIES_KEY = (-17.75, 140.5) # Mid-NT
# FUSED_SERIES_KEY = (-25.75, 133.5) # Central Australia
FUSED_SERIES_KEY = (-37.75, 145.5) # Melbourne
# FUSED_SERIES_KEY = (-12.75, 131.5) # Darwin
# FUSED_SERIES_KEY = (-33.25, 151.5) # Central Coast
# FUSED_SERIES_KEY = (-17.75, 122.5) # Broome
# FUSED_SERIES_KEY = (-42.75, 147.5) # Hobart
# FUSED_SERIES_KEY = (-16.75, 145.5) # Cairns
# FUSED_SERIES_KEY = (-35.25, 138.5) # Adelaide
# FUSED_SERIES_KEY = (-27.75, 152.5) # Brisbane
# FUSED_SERIES_KEY = (-31.75, 116.5) # Perth
# FUSED_SERIES_KEY = (-18.25, 133.5) # Lake Woods (anomaly in fused dataset)
# BOM_DAILY_FILE = 'BOMDaily086213_rosebud'
# BOM_DAILY_FILE = 'BOMDaily033250_mid_qld_coast'
BOM_DAILY_FILE = 'BOMDaily009930_albany'
# BOM_DAILY_FILE = 'BOMDaily051043_desert_nsw'
# BOM_DAILY_FILE = 'BOMDaily001026_northern_wa'

# t_data is in hours, so set period as one year accordingly
PERIOD = 24 * 365

# s is in units of mm
# t is in units of hours
# x is in units of log(mm/h)
# In paper, prec_inc (delta) = 0.2 mm, delta_s = 0.02 mm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='fused')
    parser.add_argument('--prec_inc', type=float, default=0.5)
    parser.add_argument('--x_steps', type=int, default=50)
    parser.add_argument('--t_steps', type=int, default=50)
    parser.add_argument('--s_steps', type=int, default=100)
    parser.add_argument('--delta_s', type=float, default=10)
    parser.add_argument('--use_existing', action='store_true', default=False)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--fd', action='store_true', default=False)
    parser.add_argument('--time_indep', action='store_true', default=False)
    parser.add_argument('--deseasonalise', action='store_true', default=False)
    parser.add_argument('--np_params', action='store_true', default=False)
    parser.add_argument('--np_bcs', action='store_true', default=False)
    parser.add_argument('--shrink_x_proportion', type=float, default=None)
    parser.add_argument('--shrink_x_quantile', type=float, default=None)
    parser.add_argument('--shrink_x_mixed', action='store_true', default=False)
    parser.add_argument('--decade', type=int, default=None)
    parser.add_argument('--year_cycles', type=int, default=1)
    parser.add_argument('--side', default=None)
    parser.add_argument('--trend_polynomial', type=int, default=None)
    args = parser.parse_args()

    if args.dataset == 'fused':
        prec_df, _, _ = prepare_df('data/precipitation', 'FusedData.csv', 'prec')
        prec_series = pd.Series(prec_df[FUSED_SERIES_KEY], index=pd.DatetimeIndex(prec_df.index))
        dataset = f'fused_{FUSED_SERIES_KEY[0]}_{FUSED_SERIES_KEY[1]}'
    elif args.dataset == 'fused_daily':
        prec_series = pd.read_csv(f'data/fused_upsampled/fused_daily_'
            f'{FUSED_SERIES_KEY[0]}_{FUSED_SERIES_KEY[1]}_it_3000.csv', index_col=0)
        prec_series = pd.Series(prec_series.values[:, 0], index=pd.DatetimeIndex(prec_series.index))
        dataset = f'fused_daily_{FUSED_SERIES_KEY[0]}_{FUSED_SERIES_KEY[1]}'
    elif args.dataset == 'bom_daily':
        prec_df = pd.read_csv(f'data_unfused/{BOM_DAILY_FILE}.csv')
        prec_df.index = pd.DatetimeIndex(prec_df['Date'])
        prec_series = pd.Series(prec_df['Rain']).dropna().loc['2000-04-01':]
        dataset = BOM_DAILY_FILE
    elif args.dataset == 'test':
        prec_series = pd.read_csv(f'data_unfused/test_data.csv', index_col=0, header=None)
        prec_series = pd.Series(prec_series.values[:, 0], index=pd.DatetimeIndex(prec_series.index))
        dataset = 'test_data'
    if args.decade == 1:
        prec_series = prec_series[:'2011-03-31']
    elif args.decade == 2:
        prec_series = prec_series['2011-04-01':]
    prec_inc_str = str(args.prec_inc).replace('.', 'p')
    filename = f'data/unami/{dataset}_delta_{prec_inc_str}.csv'
    if args.use_existing:
        model_df = pd.read_csv(filename)
    else:
        if args.deseasonalise:
            ds_period = 12 if args.dataset == 'fused' else 365
        else:
            ds_period = None
        model_df = prepare_model_df(prec_series, args.prec_inc, ds_period)
        model_df.to_csv(filename)
        print(f'model_df saved to file {filename}')
    
    n = args.x_steps
    m = args.t_steps
    z = args.s_steps
    x_data = np.array(model_df['x'])
    if args.np_params:
        param_info = {
            'beta': (0, 15),
            'psi': (2, 10),
            'kappa': (0, 10),
        }
        param_func = estimate_params(model_df,
            24 * (model_df['t'].iloc[-1] - model_df['t'].iloc[0]).days, param_info)
    else:
        param_func = estimate_params(model_df, PERIOD, shift_zero=True,
            trend_polynomial=args.trend_polynomial)
    x_inf, x_sup = get_x_domain(model_df['x'], args.shrink_x_proportion,
        args.shrink_x_quantile, FUSED_SERIES_KEY if args.shrink_x_mixed else None)
    if args.plot:
        plot_data(model_df, prec_series, x_inf, x_sup)
        if args.np_params:
            t_mesh = np.arange(model_df['t'].iloc[0], model_df['t'].iloc[-1], dtype='datetime64[D]')
        else:
            t_mesh = np.arange('2000', '2001', dtype='datetime64[D]')
        plot_params(model_df, t_mesh, param_func)
        
    if args.time_indep:
        t_indices = np.arange(0, 360, 30)
        u_arrays = []
        for t in t_indices:
            M_mat, G_mat = build_scheme_time_indep(param_func, x_data, n, args.delta_s, t * 24)
            # Solve iteratively for each s
            u_vecs = [np.ones(n - 1)]
            for i in range(z):
                b_vec = G_mat @ u_vecs[i].reshape((n - 1, 1))
                u_vec = spsolve(M_mat, b_vec)
                u_vecs.append(u_vec)
            u_arrays.append(np.stack(u_vecs, axis=0).reshape((z + 1, n - 1)))
        plot_results_time_indep(u_arrays, x_data, n, z, args.delta_s, param_func)
    else:
        period = 24 * (model_df['t'].iloc[-1] - model_df['t'].iloc[0]).days if args.np_params \
            else PERIOD
        t_mesh = np.linspace(0, args.year_cycles * period, args.year_cycles * m + 1)
        delta_t = period / m
        scheme_output = build_scheme(param_func, t_mesh, n, m, args.delta_s,
            delta_t, args.np_bcs, args.fd, x_inf=x_inf, x_sup=x_sup, side=args.side)
        if args.np_bcs:
            M_mats, G_mats, H_mats = scheme_output
        else:
            A_mat, G_mats = scheme_output

        start_time = datetime.now()
        print('Solving linear systems:')
        if args.np_bcs:
            u_array = np.zeros((z + 1, args.year_cycles * m, n)) if args.side \
                else np.zeros((z + 1, args.year_cycles * m, n - 1))
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
                            b_vec = G_mats[t_index] @ u_array[i - 1, j, :] + \
                                H_mats[t_index] @ u_array[i, j - 1, :]
                            # Add extra RHS term if solving one-sided solution
                            u_array[i, j, :] = spsolve(M_mats[t_index], b_vec)
                        except IndexError:
                            t_index -= m
        else:
            # TODO: handle one-sided
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
        print(f'Solving time: {datetime.now() - start_time}')
        plot_results(u_array, x_data, t_mesh, n, m, z, args.delta_s, delta_t, param_func,
            start_date=model_df['t'].iloc[0] if args.np_params else None,
            non_periodic=args.np_params, x_inf=x_inf, x_sup=x_sup, side=args.side)

if __name__ == '__main__':
    main()