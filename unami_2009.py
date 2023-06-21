import argparse
from datetime import datetime

import pandas as pd
import numpy as np
from scipy.sparse.linalg import spsolve

from helpers import prepare_df
from unami_2009_helpers import prepare_model_df, estimate_params, plot_data, plot_params, \
    build_scheme, build_scheme_time_indep, plot_results, plot_results_time_indep

np.random.seed(0)
np.set_printoptions(suppress=True)
YEARS = range(2000, 2022)
# FUSED_SERIES_KEY = (-28.75, 153.5) # Lismore
# FUSED_SERIES_KEY = (-25.75, 133.5) # Central Australia
# FUSED_SERIES_KEY = (-37.75, 145.5) # Melbourne
FUSED_SERIES_KEY = (-12.75, 131.5) # Darwin
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
    parser.add_argument('--x_steps', type=int, default=60)
    parser.add_argument('--t_steps', type=int, default=60)
    parser.add_argument('--s_steps', type=int, default=100)
    parser.add_argument('--delta_s', type=float, default=1)
    parser.add_argument('--use_existing', action='store_true', default=False)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--fd', action='store_true', default=False)
    parser.add_argument('--time_indep', action='store_true', default=False)
    args = parser.parse_args()

    if args.dataset == 'fused':
        prec_df, _, _ = prepare_df('data/precipitation', 'FusedData.csv', 'prec')
        prec_series = prec_df[FUSED_SERIES_KEY]
        dataset = f'fused_{FUSED_SERIES_KEY[0]}_{FUSED_SERIES_KEY[1]}'
    elif args.dataset == 'bom_daily':
        prec_df = pd.read_csv(f'data_unfused/{BOM_DAILY_FILE}.csv')
        prec_df.index = pd.DatetimeIndex(prec_df['Date'])
        prec_series = pd.Series(prec_df['Rain']).dropna().loc['2000-04-01':]
        dataset = BOM_DAILY_FILE
    elif args.dataset == 'test':
        prec_series = pd.read_csv(f'data_unfused/test_data.csv', index_col=0, header=None)
        prec_series = pd.Series(prec_series.values[:, 0], index=pd.DatetimeIndex(prec_series.index))
        dataset = 'test_data'
    prec_inc_str = str(args.prec_inc).replace('.', 'p')
    filename = f'data/precipitation/unami_2009_{dataset}_delta_{prec_inc_str}.csv'
    if args.use_existing:
        model_df = pd.read_csv(filename)
    else:
        model_df = prepare_model_df(prec_series, args.prec_inc)
        model_df.to_csv(filename)
        print(f'model_df saved to file {filename}')
    
    n = args.x_steps
    m = args.t_steps
    z = args.s_steps
    x_data = np.array(model_df['x'])
    param_func = estimate_params(model_df, PERIOD)
    if args.plot:
        plot_data(model_df)
        plot_params(x_data, param_func)
        
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
        t_mesh = np.linspace(0, 24 * 365, m + 1)
        delta_t = PERIOD / m
        A_mat, G_mats = build_scheme(param_func, x_data, t_mesh, n, m, args.delta_s, delta_t, args.fd)

        # Solve iteratively for each s
        u_vecs = [np.ones(m * (n - 1))]
        start_time = datetime.now()
        print('Solving linear systems:')
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
        print(f'Solving time: {datetime.now() - start_time}')
        u_array = np.stack(u_vecs, axis=0).reshape((z + 1, m, n - 1))
        plot_results(u_array, x_data, t_mesh, n, m, z, args.delta_s, delta_t, param_func)

if __name__ == '__main__':
    main()