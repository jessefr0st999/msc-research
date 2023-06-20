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

def dt_to_prev_month_days(dt: datetime):
    if dt.month in [5, 7, 10, 12]:
        return 30
    elif dt.month == 3:
        if dt.year % 100 == 0:
            return 29 if dt.year % 400 == 0 else 28
        return 29 if dt.year % 4 == 0 else 28
    return 31

def plot_data(model_df):
    # Plot x vs time of year
    figure, axis = plt.subplots(1, 1)
    years = list(range(model_df['t'].iloc[0].year, model_df['t'].iloc[-1].year + 1))
    x_arrays = []
    t_arrays = []
    for y in years:
        model_df_slice = model_df[(model_df['t'] >= datetime(y, 1, 1)) & \
            (model_df['t'] <= datetime(y, 12, 31))]
        x_arrays.append(model_df_slice['x'])
        t_arrays.append([datetime(2000, t.month, t.day) for t in model_df_slice['t']])
    for t, x in zip(t_arrays, x_arrays):
        axis.plot(t, x, 'ro')
    axis.set_xlabel('t')
    axis.set_ylabel('x')
    plt.show()

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
    t_vec = np.array(range(365))
    figure, axes = plt.subplots(2, 1)
    axes = iter(axes.T.flatten())
    axis = next(axes)
    axis.plot(t_vec, [param_func('beta', t * 24) for t in t_vec], 'bo-')
    axis.set_xlabel('t')
    axis.set_ylabel('beta')

    axis = next(axes)
    axis.plot(t_vec, [param_func('kappa', t * 24) for t in t_vec], 'go-')
    axis.set_xlabel('t')
    axis.set_ylabel('kappa = log(K)')
    plt.show()
    
    figure, axis = plt.subplots(1)
    _x_vec = np.linspace(x_vec.min(), x_vec.max(), 201)
    _psi = lambda t, x: param_func('psi', t * 24, x)
    X, T = np.meshgrid(_x_vec, t_vec)
    U = T * 0
    for i, t in enumerate(t_vec):
        for j, x in enumerate(_x_vec):
            U[i, j] = _psi(t, x)
    cmap = axis.pcolormesh(T, X, U, cmap='viridis',
        vmin=U.min().min(), vmax=U.max().max())
    plt.colorbar(cmap)
    axis.set_xlabel('t')
    axis.set_ylabel('x')
    axis.set_title('psi = log(v)')
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

# Estimate parameter functions by estimating least squares coefficients for fitting the
# functions as polynomials in x multiplied by sin/cos in t
# Number of least squares coefficients estimated for each parameter is (n_X + 1) * (1 + 2*n_t)
def estimate_params(model_df):
    n = model_df.shape[0] - 1
    x_data = np.array(model_df['x'])
    s_data = np.array(model_df['s'])
    model_df['t'] = pd.to_datetime(model_df['t'])
    # Zero for time at 1 January
    t_0 = datetime(model_df['t'][0].year, 1, 1)
    t_data = np.array([(t - t_0).days * 24 for t in model_df['t']])
    X_mats = {}
    beta_hats = {}
    def beta_f_hat(x, x_p, s, s_p):
        return x
    def psi_f_hat(x, x_p, s, s_p):
        return np.log((x_p - x)**2 / (s_p - s))
    def kappa_f_hat(x, x_p, s, s_p, beta):
        return np.log(np.abs(x_p - x) / np.abs(beta - x) / (s_p - s))
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
                    theta = 2*np.pi * (k + 1) * t_data[j] / PERIOD
                    X_mat[j, (1 + 2*n_t) * i + 2*k + 1] = x_pow * np.sin(theta)
                    X_mat[j, (1 + 2*n_t) * i + 2*k + 2] = x_pow * np.cos(theta)
        X_mats[param] = X_mat
        beta_hats[param] = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y_vec

    def param_func(param, t, x=None):
        # Input t should be in units of hours
        n_X, n_t, _ = param_info[param]
        X_vec = np.zeros((n_X + 1) * (1 + 2*n_t))
        for i in range(n_X + 1):
            # Allow x to be undefined for beta and kappa
            x_pow = 1 if i == 0 else x ** i
            X_vec[(1 + 2*n_t) * i] = x_pow
            for k in range(n_t):
                theta = 2*np.pi * (k + 1) * t / PERIOD
                X_vec[(1 + 2*n_t) * i + 2*k + 1] = x_pow * np.sin(theta)
                X_vec[(1 + 2*n_t) * i + 2*k + 2] = x_pow * np.cos(theta)
        return X_vec @ beta_hats[param]
    
    return param_func

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
    
    # Build discretisation scheme
    n = args.x_steps
    m = args.t_steps
    z = args.s_steps
    x_data = np.array(model_df['x'])
    x_inf = x_data.min()
    x_sup = x_data.max()
    # s is in units of mm
    # t is in units of hours
    # x is in units of log(mm/h)
    delta_x = (x_sup - x_inf) / n
    delta_t = PERIOD / m
    delta_s = args.delta_s
    t_mesh = np.linspace(0, 24 * 365, m + 1)
    x_mesh = np.linspace(x_inf, x_sup, n + 1)
    s_mesh = np.linspace(0, z * delta_s, z + 1)

    param_func = estimate_params(model_df)
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
    print('Setting up linear systems:')
    for k in range(n - 1):
        if k % 10 == 0:
            print(k, '/', n)
        x_k = x_mesh[k]
        x_km1 = x_mesh[k - 1] if k > 0 else x_inf
        x_km0p5 = (x_km1 + x_k) / 2
        x_kp1 = x_mesh[k + 1] if k < n - 2 else x_sup
        x_kp0p5 = (x_k + x_kp1) / 2
        for j in range(m):
            t = t_mesh[j]
            pe_l = peclet_num(x_km0p5, t)
            p_l = np.exp(pe_l)
            pe_r = peclet_num(x_kp0p5, t)
            p_r = np.exp(-pe_r)

            # Contributions from phi du/ds term
            f = lambda _x, _t: 1 / _v(_t, _x)
            if k > 0:
                M_mats[j][k, k - 1, 0] = -delta_x**2 / delta_s * f(x_km1, t) \
                    / (p_l + 1) / (p_l + 2)
            M_mats[j][k, k, 0] = -delta_x**2 / delta_s * f(x_k, t) \
                * (1 / (p_l + 2) + 1 / (p_r + 2))
            if k < n - 2:
                M_mats[j][k, k + 1, 0] = -delta_x**2 / delta_s * f(x_kp1, t) \
                    / (p_r + 1) / (p_r + 2)

            # Contributions from phi du/dt term
            f = lambda _x, _t: np.exp(-_x) / _v(_t, _x)
            if k > 0:
                M_mats[j][k, k - 1, 1] = -delta_x**2 / delta_t * f(x_km1, t) \
                    / (p_l + 1) / (p_l + 2)
            M_mats[j][k, k, 1] = -delta_x**2 / delta_t * f(x_k, t) \
                * (1 / (p_l + 2) + 1 / (p_r + 2))
            if k < n - 2:
                M_mats[j][k, k + 1, 1] = -delta_x**2 / delta_t * f(x_kp1, t) \
                    / (p_r + 1) / (p_r + 2)

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

    G_mats = [mat[:, :, 0] for mat in M_mats]
    H_mats = [mat[:, :, 1] for mat in M_mats]
    M_mats = [mat.sum(axis=2) if args.fd else mat[:, :, np.r_[:2, 3]].sum(axis=2) for mat in M_mats]

    # Create LHS matrix
    A_mat = np.zeros((m * (n - 1), m * (n - 1)))
    for j in range(m):
        start = j*(n - 1)
        end = start + n - 1
        A_mat[start : end, start : end] = M_mats[j]
        if j >= 1 and not args.fd:
            A_mat[start : end, start - (n - 1) : end - (n - 1)] = 0.5 * H_mats[j - 1]
        if j <= m - 2:
            A_mat[start : end, start + n - 1 : end + n - 1] = -H_mats[j + 1] if args.fd \
                else -0.5 * H_mats[j + 1]
    # Periodic boundary condition
    if args.fd:
        A_mat[(m - 1)*(n - 1) : m*(n - 1), 0 : n - 1] = -H_mats[0]
    else:
        A_mat[0 : n - 1, (m - 1)*(n - 1) : m*(n - 1)] = 0.5 * H_mats[m - 1]
        A_mat[(m - 1)*(n - 1) : m*(n - 1), 0 : n - 1] = -0.5 * H_mats[0]

    # Solve iteratively for each s
    u_vecs = [np.ones(m * (n - 1))]
    start_time = datetime.now()
    print('Solving linear systems:')
    for i in range(z):
        b_vec = np.zeros((m * (n - 1), 1))
        for j in range(m):
            start = j*(n - 1)
            end = start + n - 1
            b_vec[start : end] = (G_mats[j] @ u_vecs[i][start : end]).reshape((n - 1, 1))
        u_vec = spsolve(A_mat, b_vec)
        u_vecs.append(u_vec)
        if (i + 1) % 10 == 0:
            print(i + 1, u_vec.mean(), np.median(u_vec), u_vec.std(), u_vec.max(), u_vec.min())
        if i == 0:
            np.savetxt('A_mat_0.csv', A_mat, delimiter=',')
            np.savetxt('b_vec_0.csv', b_vec.reshape((m, n - 1)), delimiter=',')
            np.savetxt('u_vec_0.csv', u_vec.reshape((m, n - 1)), delimiter=',')
    print(f'Solving time: {datetime.now() - start_time}')

    u_array = np.stack(u_vecs, axis=0).reshape((z + 1, m, n - 1))
    last_u = u_vecs[-1].reshape((m, n - 1))
    
    figure, axes = plt.subplots(3, 4)
    axes = iter(axes.flatten())
    u_min = np.Inf
    u_max = -np.Inf
    t_plot_indices = [floor(i * m / 12) for i in range(12)]
    for i, plot_t in enumerate(t_plot_indices):
        u_slice = u_array[:, plot_t, :]
        u_min = np.min([u_min, u_slice.min(axis=(0, 1))])
        u_max = np.max([u_max, u_slice.max(axis=(0, 1))])
    for i, plot_t in enumerate(t_plot_indices):
        axis = next(axes)
        X, S = np.meshgrid(x_mesh[1 : -1], s_mesh)
        cmap = axis.pcolormesh(S, X, u_array[:, plot_t, :], cmap='viridis',
            vmin=u_min, vmax=u_max)
        plt.colorbar(cmap)
        axis.set_xlabel('s')
        axis.set_ylabel('x')
        axis.set_title((datetime(2000, 1, 1) + timedelta(days=plot_t * delta_t / 24)).strftime('%b %d'))
        beta = _beta(plot_t * delta_t)
        axis.plot(s_mesh, [beta for _ in s_mesh], 'r-')
    plt.show()

    model_df_slice = model_df[(model_df['t'] >= datetime(2021, 1, 1)) & \
        (model_df['t'] <= datetime(2021, 12, 31))]
    figure, axes = plt.subplots(3, 4)
    axes = iter(axes.flatten())
    u_min = np.Inf
    u_max = -np.Inf
    s_plot_indices = [floor(i * z / 12) for i in range(12)]
    for i, plot_s in enumerate(s_plot_indices):
        u_slice = u_array[plot_s, :, :]
        u_min = np.min([u_min, u_slice.min(axis=(0, 1))])
        u_max = np.max([u_max, u_slice.max(axis=(0, 1))])
    for i, plot_s in enumerate(s_plot_indices):
        axis = next(axes)
        X, T = np.meshgrid(x_mesh[1 : -1], t_mesh[1:] / 24)
        cmap = axis.pcolormesh(T, X, u_array[plot_s, :, :], cmap='viridis', vmin=u_min, vmax=u_max)
        plt.colorbar(cmap)
        axis.set_xlabel('t')
        axis.set_ylabel('x')
        axis.set_title(f's = {plot_s * delta_s} mm')
        axis.plot(t_mesh[1:] / 24, [_beta(t) for t in t_mesh[1:]], 'r-')
    plt.show()

    figure, axis = plt.subplots(1)
    j_inf = u_array.max(axis=(1, 2))
    j_2 = np.zeros(z + 1)
    for i in range(z + 1):
        j_2[i] = (u_array[i, :, :] ** 2).sum(axis=(0, 1))
    j_2 /= (n - 1) * m
    axis.plot(s_mesh, j_inf, 'r', label='j_inf')
    axis.plot(s_mesh, j_2, 'b', label='j_2')
    axis.set_xlabel('s')
    axis.set_ylabel('u')
    axis.legend()
    plt.show()
    
if __name__ == '__main__':
    main()