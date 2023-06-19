import argparse
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from math import floor

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

from helpers import get_map, scatter_map, prepare_df, configure_plots
from unami_2009 import dt_to_prev_month_days, plot_data, plot_params, prepare_model_df

np.random.seed(0)
np.set_printoptions(suppress=True)
YEARS = range(2000, 2022)
FUSED_SERIES_KEY = (-28.75, 153.5) # Lismore
# FUSED_SERIES_KEY = (-25.75, 133.5) # Central Australia
# FUSED_SERIES_KEY = (-37.75, 145.5) # Melbourne
# FUSED_SERIES_KEY = (-12.75, 131.5) # Darwin
BOM_DAILY_FILE = 'BOMDaily086213_rosebud'
# BOM_DAILY_FILE = 'BOMDaily033250_mid_qld_coast'
# BOM_DAILY_FILE = 'BOMDaily009930_albany'
# BOM_DAILY_FILE = 'BOMDaily051043_desert_nsw'
# BOM_DAILY_FILE = 'BOMDaily001026_northern_wa'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='fused')
    parser.add_argument('--prec_inc', type=float, default=500)
    parser.add_argument('--x_steps', type=int, default=60)
    parser.add_argument('--s_steps', type=int, default=100)
    parser.add_argument('--delta_s', type=float, default=1)
    # Day of year to use for parameter values, default is 1 January
    parser.add_argument('--day', type=int, default=0)
    parser.add_argument('--use_existing', action='store_true', default=False)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--max_one', action='store_true', default=False)
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
    if args.plot:
        plot_data(model_df)

    # Estimate parameter functions by estimating least squares coefficients for fitting the
    # functions as polynomials in x multiplied by sin/cos in t
    # Number of least squares coefficients estimated for each parameter is (n_X + 1) * (1 + 2*n_t)
    def beta_f_hat(x, x_p, s, s_p):
        return x

    def psi_f_hat(x, x_p, s, s_p):
        return np.log((x_p - x)**2 / (s_p - s))

    def kappa_f_hat(x, x_p, s, s_p, beta):
        return np.log(np.abs(x_p - x) / np.abs(beta - x) / (s_p - s))
    
    n = model_df.shape[0] - 1
    x_data = np.array(model_df['x'])
    s_data = np.array(model_df['s'])
    model_df['t'] = pd.to_datetime(model_df['t'])
    # Zero for time at 1 January
    t_0 = datetime(model_df['t'][0].year, 1, 1)
    t_data = np.array([(t - t_0).days * 24 for t in model_df['t']])
    # t_data is in hours, so set period as one year accordingly
    P = 24 * 365
    X_mats = {}
    beta_hats = {}
    param_info = {
        'beta': (0, 3, beta_f_hat),
        'psi': (2, 2, psi_f_hat),
        'kappa': (0, 2, kappa_f_hat),
    }
    for param, (n_X, n_t, f_hat) in param_info.items():
        X_mat = np.zeros((n, (n_X + 1) * (1 + 2*n_t)))
        y_vec = np.zeros(n)
        for j in range(n):
            if param == 'kappa':
                y_vec[j] = f_hat(x_data[j], x_data[j + 1], s_data[j], s_data[j + 1],
                    beta=X_mats['beta'][j, :] @ beta_hats['beta'])
            else:
                y_vec[j] = f_hat(x_data[j], x_data[j + 1], s_data[j], s_data[j + 1])
            for i in range(n_X + 1):
                x_pow = x_data[j] ** i
                X_mat[j, (1 + 2*n_t) * i] = x_pow
                for k in range(n_t):
                    theta = 2*np.pi * (k + 1) * t_data[j] / P
                    X_mat[j, (1 + 2*n_t) * i + 2*k + 1] = x_pow * np.sin(theta)
                    X_mat[j, (1 + 2*n_t) * i + 2*k + 2] = x_pow * np.cos(theta)
        X_mats[param] = X_mat
        beta_hats[param] = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y_vec
    
    # Next, build discretisation scheme
    n = args.x_steps
    z = args.s_steps
    x_inf = x_data.min()
    x_sup = x_data.max()
    # s is in units of mm
    # t is in units of hours
    # x is in units of log(mm/h)
    # In paper, prec_inc (delta) = 0.2 mm, delta_s = 0.02 mm
    delta_x = (x_sup - x_inf) / n
    delta_s = args.delta_s
    print('x_inf', x_inf)
    print('x_sup', x_sup)
    print('delta_x', delta_x)
    print('delta_s', delta_s)
    x_mesh = np.linspace(x_inf, x_sup, n + 1)
    s_mesh = np.linspace(0, z * delta_s, z + 1)
    _time = args.day * 24

    def param_func(param, t, x=None):
        # Input t should be in units of hours
        n_X, n_t, _ = param_info[param]
        X_vec = np.zeros((n_X + 1) * (1 + 2*n_t))
        for i in range(n_X + 1):
            # Allow x to be undefined for beta and kappa
            x_pow = 1 if i == 0 else x ** i
            X_vec[(1 + 2*n_t) * i] = x_pow
            for k in range(n_t):
                theta = 2*np.pi * (k + 1) * t / P
                X_vec[(1 + 2*n_t) * i + 2*k + 1] = x_pow * np.sin(theta)
                X_vec[(1 + 2*n_t) * i + 2*k + 2] = x_pow * np.cos(theta)
        return X_vec @ beta_hats[param]

    def _beta(t):
        return param_func('beta', t)
    def _v(t, x):
        return np.exp(param_func('psi', t, x))
    def _K(t):
        return np.exp(param_func('kappa', t))

    if args.plot:
        plot_params(x_data, param_func)
    
    def peclet_num(x, t):
        return _K(t) * (_beta(t) - x) * delta_x / _v(t, x)
    
    M_mat = np.zeros((n - 1, n - 1, 4))
    G_mat = np.zeros((n - 1, n - 1))
    for k in range(n - 1):
        if k % 10 == 0:
            print(k, '/', n)
        x_k = x_mesh[k]
        x_km1 = x_mesh[k - 1] if k > 0 else x_inf
        x_km0p5 = (x_km1 + x_k) / 2
        x_kp1 = x_mesh[k + 1] if k < n - 2 else x_sup
        x_kp0p5 = (x_k + x_kp1) / 2
        
        pe_l = peclet_num(x_km0p5, _time)
        p_l = np.exp(pe_l)
        pe_r = peclet_num(x_kp0p5, _time)
        p_r = np.exp(-pe_r)

        # Contributions from phi du/ds term
        f = lambda _x, _t: 1 / _v(_t, _x)
        if k > 0:
            G_mat[k, k - 1] = -delta_x**2 / delta_s * f(x_km1, _time) \
                / (p_l + 1) / (p_l + 2)
            M_mat[k, k - 1, 0] = G_mat[k, k - 1]
        G_mat[k, k] = -delta_x**2 / delta_s * f(x_k, _time) \
            * (1 / (p_l + 2) + 1 / (p_r + 2))
        M_mat[k, k, 0] = G_mat[k, k]
        if k < n - 2:
            G_mat[k, k + 1] = -delta_x**2 / delta_s * f(x_kp1, _time) \
                / (p_r + 1) / (p_r + 2)
            M_mat[k, k + 1, 0] = G_mat[k, k + 1]

        # Contributions from phi du/dx term
        if k > 0:
            M_mat[k, k - 1, 2] = -pe_l / (p_l + 1)
        M_mat[k, k, 2] = pe_l / (p_l + 1) - pe_r / (p_r + 1)
        if k < n - 2:
            M_mat[k, k + 1, 2] = pe_r / (p_r + 1)

        # Contributions from dphi/dx du/dx term
        if k > 0:
            M_mat[k, k - 1, 3] = 1/2
        M_mat[k, k, 3] = -1
        if k < n - 2:
            M_mat[k, k + 1, 3] = 1/2

    M_mat = M_mat.sum(axis=2)

    # Solve iteratively for each s
    u_vecs = [np.ones(n - 1)]
    for i in range(z):
        b_vec = G_mat @ u_vecs[i].reshape((n - 1, 1))
        u_vec = spsolve(M_mat, b_vec)
        if args.max_one:
            u_vec[u_vec > 1] = 1
        u_vecs.append(u_vec)
        # print(i, u_vec.mean(), np.median(u_vec), u_vec.std(), u_vec.max(), u_vec.min())
        # if i == 0:
        #     np.savetxt('M_mat_0_t_indep.csv', M_mat, delimiter=',')
        #     np.savetxt('b_vec_0_t_indep.csv', b_vec, delimiter=',')
        #     np.savetxt('u_vec_0_t_indep.csv', u_vec, delimiter=',')

    u_array = np.stack(u_vecs, axis=0).reshape((z + 1, n - 1))
    figure, axis = plt.subplots(1)
    X, S = np.meshgrid(x_mesh[1 : -1], s_mesh)
    cmap = axis.pcolormesh(S, X, u_array, cmap='viridis',
        vmin=u_array.min(axis=(0, 1)), vmax=u_array.max(axis=(0, 1)))
    plt.colorbar(cmap)
    axis.set_xlabel('s')
    axis.set_ylabel('x')
    beta = _beta(_time)
    print('beta', beta)
    print('K', _K(_time))
    print('v at x_inf', _v(_time, x_inf))
    print('v at x_sup', _v(_time, x_sup))
    axis.plot(s_mesh, [beta for _ in s_mesh], 'r-')
    plt.show()

    figure, axis = plt.subplots(1)
    j_inf = u_array.max(axis=1)
    j_2 = np.zeros(z + 1)
    for i in range(z + 1):
        j_2[i] = (u_array[i, :] ** 2).sum()
    j_2 /= (n - 1)
    axis.plot(s_mesh, j_inf, 'r', label='j_inf')
    axis.plot(s_mesh, j_2, 'b', label='j_2')
    axis.set_xlabel('s')
    axis.set_ylabel('u')
    axis.legend()
    plt.show()
    
if __name__ == '__main__':
    main()