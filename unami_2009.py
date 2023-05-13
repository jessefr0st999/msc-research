import argparse
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from helpers import get_map, scatter_map, prepare_df, configure_plots

np.set_printoptions(suppress=True)
YEARS = range(2000, 2022)
SERIES_KEY = (-28.75, 153.5)
OUTPUT_FILE = 'unami_2009_proc_data.csv'

def dt_to_prev_month_days(dt: datetime):
    if dt.month in [5, 7, 10, 12]:
        return 30
    elif dt.month == 3:
        if dt.year % 100 == 0:
            return 29 if dt.year % 400 == 0 else 28
        return 29 if dt.year % 4 == 0 else 28
    return 31

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/precipitation')
    parser.add_argument('--data_file', default='FusedData.csv')
    parser.add_argument('--prec_inc', type=int, default=500)
    parser.add_argument('--x_steps', type=int, default=30)
    parser.add_argument('--t_steps', type=int, default=30)
    parser.add_argument('--s_steps', type=int, default=1000)
    parser.add_argument('--calculate_model_df', action='store_true', default=False)
    parser.add_argument('--plot', action='store_true', default=False)
    args = parser.parse_args()

    prec_df, lats, lons = prepare_df(args.data_dir, args.data_file, 'prec')
    filename = f'{args.data_dir}/{OUTPUT_FILE}'
    if args.calculate_model_df:
        cum_sums = prec_df[SERIES_KEY].cumsum()
        times = prec_df.index.values
        # Calculate values of the temporal variable X
        model_df_values = []
        for i, s in enumerate(cum_sums):
            # t_delta: at a given t, how far (in hours) do you have to look back until the
            # difference surpasses prec_inc (delta in paper)
            if s - cum_sums[0] < args.prec_inc:
                continue
            i_prev = i - 1
            while s - cum_sums[i_prev] < args.prec_inc:
                i_prev -= 1
            if i_prev == i - 1:
                # Handle case where the threshold is exceeded by multiples of the jump on the
                # latest timestamp by using a fraction of the time granularity for t_delta
                inc_jumps = int((s - cum_sums[i_prev]) // args.prec_inc)
                t_delta = (times[i_prev + 1] - times[i_prev]).days / inc_jumps
                ######################################################
                # print(times[i].strftime('%Y %m'), i, i - i_prev, round(s),
                #     round(s - cum_sums[i_prev]), inc_jumps, round(t_delta, 2))
                ######################################################
            else:
                t_delta = (times[i] - times[i_prev]).days
            t_delta += np.random.normal(0, 10)
            x = np.log(args.prec_inc / t_delta)
            model_df_values.append((s, times[i], x, t_delta))
        model_df = pd.DataFrame(model_df_values)
        model_df.columns = ['s', 't', 'x', 't_delta']
        model_df.to_csv(filename)
        print(f'model_df saved to file {filename}')
    else:
        model_df = pd.read_csv(filename)

    # Estimate parameter functions by estimating least squares coefficients for fitting the
    # functions as polynomials in x multiplied by sin/cos in t
    # Number of least squares coefficients estimated for each parameter is (n_X + 1) * (1 + 2*n_t)
    def beta_f_hat(x, x_p, s, s_p):
        return x

    def psi_f_hat(x, x_p, s, s_p):
        return np.log((x_p - x)**2 / (s_p - s))

    def kappa_f_hat(x, x_p, s, s_p, beta):
        return np.log(np.abs(x_p - x) / np.abs(beta - x) / np.abs(s_p - s))
    
    n = model_df.shape[0] - 1
    x_data = np.array(model_df['x'])
    s_data = np.array(model_df['s'])
    # Data is uniform in time, so just implement with a sequence of integers representing the month
    # Hence, period is just 12
    t_data = np.array(range(n))
    P = 12
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
                y_vec[j] = f_hat(x_data[j], s_data[j + 1], x_data[j], s_data[j + 1],
                    beta=X_mats['beta'][j, :] @ beta_hats['beta'])
            else:
                y_vec[j] = f_hat(x_data[j], s_data[j + 1], x_data[j], s_data[j + 1])
            for i in range(n_X + 1):
                x_pow = x_data[j] ** i
                X_mat[j, (1 + 2*n_t) * i] = x_pow
                for k in range(n_t):
                    theta = 2*np.pi * t_data[j] * (k + 1) / P
                    X_mat[j, (1 + 2*n_t) * i + 2*k + 1] = x_pow * np.sin(theta)
                    X_mat[j, (1 + 2*n_t) * i + 2*k + 2] = x_pow * np.cos(theta)
        X_mats[param] = X_mat
        beta_hats[param] = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y_vec
    
    # Next, build discretisation scheme
    n = args.x_steps
    m = args.t_steps
    z = args.s_steps
    t_data = np.array(range(m + 1))
    x_inf = x_data.min()
    x_sup = x_data.max()
    delta_x = (x_sup - x_inf) / n
    delta_t = 1 # Should this be in a time unit?
    # delta_s = s_data.max() / z
    delta_s = 1
    print('delta_x', delta_x)
    print('delta_t', delta_t)
    print('delta_s', delta_s)

    def param_func(param, t, x=None):
        n_X, n_t, _ = param_info[param]
        X_vec = np.zeros((n_X + 1) * (1 + 2*n_t))
        for i in range(n_X + 1):
            # Allow x to be undefined for beta and kappa
            x_pow = 1 if i == 0 else x ** i
            X_vec[(1 + 2*n_t) * i] = x_pow
            for k in range(n_t):
                # Time period now equal to number of time steps
                theta = 2*np.pi * t * (k + 1) / m
                X_vec[(1 + 2*n_t) * i + 2*k + 1] = x_pow * np.sin(theta)
                X_vec[(1 + 2*n_t) * i + 2*k + 2] = x_pow * np.cos(theta)
        return X_vec @ beta_hats[param]
    def _beta(t):
        return param_func('beta', t)
    def _v(t, x):
        return np.exp(param_func('psi', t, x))
    def _K(t):
        return np.exp(param_func('kappa', t))
    
    def peclet_num(x, t):
        return _K(t) * (_beta(t) - x) * delta_x / _v(t, x)
    
    M_mats = [np.zeros((n - 1, n - 1, 4)) for _ in range(m)]
    G_mats = [np.zeros((n - 1, n - 1)) for _ in range(m)]
    H_mats = [np.zeros((n - 1, n - 1)) for _ in range(m)]
    for k in range(n - 1):
        print(k, '/', n)
        x_k = x_data[k]
        x_km1 = x_data[k - 1] if k > 0 else 0
        x_km0p5 = (x_km1 + x_k) / 2
        x_kp1 = x_data[k + 1] if k < n - 2 else 0
        x_kp0p5 = (x_k + x_kp1) / 2
        for j in range(m):
            t = t_data[j + 1]
            pe_l = peclet_num(x_km0p5, t)
            p_l = np.exp(pe_l)
            pe_r = peclet_num(x_kp0p5, t)
            p_r = np.exp(-pe_r)

            # Contributions from phi du/ds term
            f = lambda x: 1 /_v(t, x)
            if k > 0:
                G_mats[j][k, k - 1] = delta_x**2 / delta_s * f(x_km1) \
                    / (p_l + 1) / (p_l + 2)
                M_mats[j][k, k - 1, 0] = -G_mats[j][k, k - 1]
            G_mats[j][k, k] = delta_x**2 / delta_s * f(x_k) \
                * (1 / (p_l + 2) + 1 / (p_r + 2))
            M_mats[j][k, k, 0] = -G_mats[j][k, k]
            if k < n - 2:
                G_mats[j][k, k + 1] = delta_x**2 / delta_s * f(x_kp1) \
                    / (p_r + 1) / (p_r + 2)
                M_mats[j][k, k + 1, 0] = -G_mats[j][k, k + 1]

            # Contributions from phi du/dt term
            f = lambda x: np.exp(-x) /_v(t, x)
            if k > 0:
                H_mats[j][k, k - 1] = delta_x**2 / delta_t * f(x_km1) \
                    / (p_l + 1) / (p_l + 2)
                M_mats[j][k, k - 1, 1] = -H_mats[j][k, k - 1]
            H_mats[j][k, k] = delta_x**2 / delta_t * f(x_k) \
                * (1 / (p_l + 2) + 1 / (p_r + 2))
            M_mats[j][k, k, 1] = -H_mats[j][k, k]
            if k < n - 2:
                H_mats[j][k, k + 1] = delta_x**2 / delta_t * f(x_kp1) \
                    / (p_r + 1) / (p_r + 2)
                M_mats[j][k, k + 1, 1] = -H_mats[j][k, k + 1]

            # Contributions from phi du/dx term``
            if k > 0:
                M_mats[j][k, k - 1, 2] = -pe_l / (p_l + 1)
            M_mats[j][k, k, 2] = pe_l / (p_l + 1) - pe_r / (p_r + 1)
            if k < n - 2:
                M_mats[j][k, k + 1, 2] = pe_r / (p_r + 1)

            # Contributions from dphi/dx du/dx term
            if k > 0:
                M_mats[j][k, k - 1, 3] = 1/2
            M_mats[j][k, k, 3] = -1
            if k < n - 2:
                M_mats[j][k, k + 1, 3] = 1/2

    M_mats = [mat.sum(axis=2) for mat in M_mats]
    
    # Solve iteratively for each s
    u_vecs = [np.ones(m * (n - 1))]
    for i in range(z):
        A_mat = np.zeros((m * (n - 1), m * (n - 1)))
        b_vec = np.zeros((m * (n - 1), 1))
        for j in range(m):
            start = j*(n - 1)
            end = start + n - 1
            b_vec[start : end] = (G_mats[j] @ u_vecs[i][start : end]).reshape((n - 1, 1))
            A_mat[start : end, start : end] = M_mats[j]
            if j <= m - 2:
                A_mat[start : end, start + n - 1 : end + n - 1] = H_mats[j + 1]
        # Periodic boundary condition
        A_mat[(m - 1)*(n - 1) : m*(n - 1), 0 : n - 1] = -H_mats[0]
        u_vecs.append(np.linalg.solve(A_mat, b_vec))
        print()
        print(u_vecs[i+1].reshape((m, n - 1)))
        print(i, u_vecs[i+1].mean(), u_vecs[i+1].std(), u_vecs[i+1].sum())

    if not args.plot:
        return

    figure, axes = plt.subplots(3, 2)
    axes = iter(axes.T.flatten())
    axis = next(axes)
    axis.plot(model_df['t'], model_df['s'], 'bo-')
    axis.set_xlabel('t')
    axis.set_ylabel('s')
    
    axis = next(axes)
    axis.plot(model_df['t'], model_df['x'], 'ro-')
    axis.set_xlabel('t')
    axis.set_ylabel('x')

    axis = next(axes)
    axis.plot(model_df['t'], model_df['t_delta'], 'go-')
    axis.set_xlabel('t')
    axis.set_ylabel('t_delta')

    axis = next(axes)
    axis.plot(model_df['t'][:13], [param_func('beta', t) for t in range(13)], 'bo-')
    axis.set_xlabel('t')
    axis.set_ylabel('beta')

    axis = next(axes)
    axis.plot(model_df['t'][:13], [param_func('psi', t, x_data.mean()) for t in range(13)], 'ro-')
    axis.set_xlabel('t')
    axis.set_ylabel('psi')

    axis = next(axes)
    axis.plot(model_df['t'][:13], [param_func('kappa', t) for t in range(13)], 'go-')
    axis.set_xlabel('t')
    axis.set_ylabel('kappa')
    plt.show()
    
if __name__ == '__main__':
    main()