import argparse
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

from helpers import get_map, scatter_map, prepare_df, configure_plots

np.random.seed(0)
np.set_printoptions(suppress=True)
YEARS = range(2000, 2022)
FUSED_SERIES_KEY = (-28.75, 153.5)
BOM_DAILY_FILE = 'BOMDaily086213_rosebud'

def dt_to_prev_month_days(dt: datetime):
    if dt.month in [5, 7, 10, 12]:
        return 30
    elif dt.month == 3:
        if dt.year % 100 == 0:
            return 29 if dt.year % 400 == 0 else 28
        return 29 if dt.year % 4 == 0 else 28
    return 31

def plot_data(model_df):
    figure, axes = plt.subplots(3, 1)
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
    plt.show()

def plot_params(x_vec, param_func):
    t_vec = range(365)
    figure, axes = plt.subplots(3, 1)
    axes = iter(axes.T.flatten())
    axis = next(axes)
    axis.plot(t_vec, [param_func('beta', t) for t in t_vec], 'bo-')
    axis.set_xlabel('t')
    axis.set_ylabel('beta')

    axis = next(axes)
    axis.plot(t_vec, [param_func('psi', t, x_vec.mean()) for t in t_vec], 'ro-')
    axis.set_xlabel('t')
    axis.set_ylabel('psi = log(v)')

    axis = next(axes)
    axis.plot(t_vec, [param_func('kappa', t) for t in t_vec], 'go-')
    axis.set_xlabel('t')
    axis.set_ylabel('kappa = log(K)')
    plt.show()

def prepare_model_df(prec_series, prec_inc):
    cum_sums = prec_series.cumsum()
    times = prec_series.index
    # Calculate values of the temporal variable X
    model_df_values = []
    for i, s in enumerate(cum_sums):
        # t_delta: at a given t, how far (in hours) do you have to look back until the
        # difference surpasses prec_inc (delta in paper)
        if s - cum_sums[0] < prec_inc:
            continue
        i_prev = i - 1
        # Skip consecutive equal s-values
        if s == cum_sums[i_prev]:
            continue
        while s - cum_sums[i_prev] < prec_inc:
            i_prev -= 1
        if i_prev == i - 1:
            # Handle case where the threshold is exceeded by multiples of the jump on the latest
            # timestamp by using a fraction of the jump between previous timestamps for t_delta
            inc_jumps = int((s - cum_sums[i_prev]) // prec_inc)
            t_delta = 24 * (times[i_prev + 1] - times[i_prev]).days / inc_jumps
        else:
            t_delta = 24 * (times[i] - times[i_prev]).days
        # Used to prevent consecutive x-values being equal
        t_delta += np.abs(np.random.normal(0, 6))
        x = np.log(prec_inc / t_delta)
        model_df_values.append((s, times[i], x, t_delta))
    model_df = pd.DataFrame(model_df_values)
    model_df.columns = ['s', 't', 'x', 't_delta']
    return model_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', default='fused')
    parser.add_argument('--prec_inc', type=int, default=500)
    parser.add_argument('--x_steps', type=int, default=30)
    parser.add_argument('--t_steps', type=int, default=30)
    parser.add_argument('--s_steps', type=int, default=1000)
    parser.add_argument('--delta_s', type=int, default=0.02)
    parser.add_argument('--calculate_model_df', action='store_true', default=False)
    parser.add_argument('--plot', action='store_true', default=False)
    args = parser.parse_args()

    if args.data_type == 'fused':
        prec_df, _, _ = prepare_df('data/precipitation', 'FusedData.csv', 'prec')
        prec_series = prec_df[FUSED_SERIES_KEY]
    elif args.data_type == 'bom_daily':
        prec_df = pd.read_csv(f'data_unfused/{BOM_DAILY_FILE}.csv')
        prec_df.index = pd.DatetimeIndex(prec_df['Date'])
        # prec_series = pd.Series(prec_df['Rain']).dropna().loc['2000-04-01':]
        prec_series = pd.Series(prec_df['Rain']).dropna().loc['2000-04-01':]
    filename = f'data/precipitation/unami_2009_proc_delta_{args.prec_inc}.csv'
    if args.calculate_model_df:
        model_df = prepare_model_df(prec_series, args.prec_inc)
        model_df.to_csv(filename)
        print(f'model_df saved to file {filename}')
    else:
        model_df = pd.read_csv(filename)
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
                y_vec[j] = f_hat(x_data[j], x_data[j + 1], s_data[j], s_data[j + 1],
                    beta=X_mats['beta'][j, :] @ beta_hats['beta'])
            else:
                y_vec[j] = f_hat(x_data[j], x_data[j + 1], s_data[j], s_data[j + 1])
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
    # s is in units of mm
    # t is in units of hours
    # x is in units of log(mm/h)
    delta_x = (x_sup - x_inf) / n
    P = 24 * 365
    delta_t = P / m
    delta_s = args.delta_s

    def param_func(param, t, x=None):
        # Input t is in units of days, so convert to hours
        t = t * 24
        n_X, n_t, _ = param_info[param]
        X_vec = np.zeros((n_X + 1) * (1 + 2*n_t))
        for i in range(n_X + 1):
            # Allow x to be undefined for beta and kappa
            x_pow = 1 if i == 0 else x ** i
            X_vec[(1 + 2*n_t) * i] = x_pow
            for k in range(n_t):
                theta = 2*np.pi * t * (k + 1) / P
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
    
    M_mats = [np.zeros((n - 1, n - 1, 4)) for _ in range(m)]
    G_mats = [np.zeros((n - 1, n - 1)) for _ in range(m)]
    H_mats = [np.zeros((n - 1, n - 1)) for _ in range(m)]
    for k in range(n - 1):
        if k % 10 == 0:
            print(k, '/', n)
        x_k = x_data[k]
        x_km1 = x_data[k - 1] if k > 0 else x_inf
        x_km0p5 = (x_km1 + x_k) / 2
        x_kp1 = x_data[k + 1] if k < n - 2 else x_sup
        x_kp0p5 = (x_k + x_kp1) / 2
        for j in range(m):
            t = t_data[j + 1]
            pe_l = peclet_num(x_km0p5, t)
            p_l = np.exp(pe_l)
            pe_r = peclet_num(x_kp0p5, t)
            p_r = np.exp(-pe_r)

            tp1 = t_data[j + 1]
            pe_l_tp1 = peclet_num(x_km0p5, tp1)
            p_l_tp1 = np.exp(pe_l_tp1)
            pe_r_tp1 = peclet_num(x_kp0p5, tp1)
            p_r_tp1 = np.exp(-pe_r_tp1)

            # Contributions from phi du/ds term
            f = lambda x: 1 /_v(t, x)
            if k > 0:
                G_mats[j][k, k - 1] = delta_x**2 / delta_s * f(x_km1) \
                    / (p_l + 1) / (p_l + 2)
                M_mats[j][k, k - 1, 0] = G_mats[j][k, k - 1]
            G_mats[j][k, k] = delta_x**2 / delta_s * f(x_k) \
                * (1 / (p_l + 2) + 1 / (p_r + 2))
            M_mats[j][k, k, 0] = G_mats[j][k, k]
            if k < n - 2:
                G_mats[j][k, k + 1] = delta_x**2 / delta_s * f(x_kp1) \
                    / (p_r + 1) / (p_r + 2)
                M_mats[j][k, k + 1, 0] = G_mats[j][k, k + 1]

            # Contributions from phi du/dt term
            f = lambda x: np.exp(-x) /_v(t, x)
            if k > 0:
                H_mats[j][k, k - 1] = delta_x**2 / delta_t * f(x_km1) \
                    / (p_l_tp1 + 1) / (p_l_tp1 + 2)
                M_mats[j][k, k - 1, 1] = H_mats[j][k, k - 1]
            H_mats[j][k, k] = delta_x**2 / delta_t * f(x_k) \
                * (1 / (p_l_tp1 + 2) + 1 / (p_r_tp1 + 2))
            M_mats[j][k, k, 1] = H_mats[j][k, k]
            if k < n - 2:
                H_mats[j][k, k + 1] = delta_x**2 / delta_t * f(x_kp1) \
                    / (p_r_tp1 + 1) / (p_r_tp1 + 2)
                M_mats[j][k, k + 1, 1] = H_mats[j][k, k + 1]

            # Contributions from phi du/dx term
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

    # Alternating negative solutions fixed due to incorrect signs in G, H matrices

    # Mean, SD of output vary on delta_x but not delta_t, why?
    
    # Elements of A matrix corresponding to H are much smaller than those for M
    # phi du/ds part of each M_mat term dominates
    # phi du/dt, phi du/dx parts are negligible
    # Peclet numbers are quite small (around 10^-5) due to _v being large (around 7000)
    # Fixed due to typo in param estimation, now between 1 and 2
    # Now the phi du/ds parts dominate (again)

    M_mats = [mat.sum(axis=2) for mat in M_mats]
    
    # Solve iteratively for each s
    u_vecs = [np.ones(m * (n - 1))]
    start_time = datetime.now()
    for i in range(z):
        A_mat = np.zeros((m * (n - 1), m * (n - 1)))
        b_vec = np.zeros((m * (n - 1), 1))
        for j in range(m):
            start = j*(n - 1)
            end = start + n - 1
            b_vec[start : end] = (G_mats[j] @ u_vecs[i][start : end]).reshape((n - 1, 1))
            A_mat[start : end, start : end] = M_mats[j]
            if j <= m - 2:
                A_mat[start : end, start + n - 1 : end + n - 1] = -H_mats[j + 1]
        # Periodic boundary condition
        A_mat[(m - 1)*(n - 1) : m*(n - 1), 0 : n - 1] = -H_mats[0]
        if i == 0:
            np.savetxt('A_mat.csv', A_mat, delimiter=',')
        u_vecs.append(spsolve(A_mat, b_vec))
        if i == 5:
            np.savetxt(f'u_{i+1}.csv', u_vecs[i+1].reshape((m, n - 1)), delimiter=',')
        print(i, u_vecs[i+1].mean(), u_vecs[i+1].std(), u_vecs[i+1].sum())
    print(f'Solving time: {datetime.now() - start_time}')
    
if __name__ == '__main__':
    main()