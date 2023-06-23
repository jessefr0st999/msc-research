from datetime import datetime, timedelta
from math import floor
from statsmodels.tsa.seasonal import seasonal_decompose

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def prepare_model_df(prec_series, prec_inc, ds_period=None):
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
        ######################################################
        # print(times[i].strftime('%Y_%m_%d'), i, i - i_prev, round(s), round(cum_sums[i_prev]),
        #     int((s - cum_sums[i_prev]) // prec_inc), round(t_delta, 2))
        ######################################################
        # Add a small amount of noise to prevent consecutive x-values being equal, which
        # causes log of zero in the parameter estimation
        x = np.log(prec_inc / t_delta) + np.random.normal(0, 0.1)
        model_df_values.append((s, times[i], x, t_delta))
    model_df = pd.DataFrame(model_df_values)
    model_df.columns = ['s', 't', 'x', 't_delta']
    if ds_period:
        x_decomp = seasonal_decompose(model_df['x'], model='additive', period=ds_period)
        model_df['x'] = model_df['x'] - x_decomp.seasonal
    return model_df


# Estimate parameter functions by estimating least squares coefficients for fitti ng the
# functions as polynomials in x multiplied by sin/cos in t
# Number of least squares coefficients estimated for each parameter is (n_X + 1) * (1 + 2*n_t)
def estimate_params(model_df, period, param_info=None):
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
    param_f_hats = {
        'beta': beta_f_hat,
        'psi': psi_f_hat,
        'kappa': kappa_f_hat,
    }
    if param_info is None:
        param_info = {
            'beta': (0, 3),
            'psi': (2, 2),
            'kappa': (0, 2),
        }
    for param, (n_X, n_t) in param_info.items():
        X_mat = np.zeros((n, (n_X + 1) * (1 + 2*n_t)))
        y_vec = np.zeros(n)
        f_hat = param_f_hats[param]
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
                    theta = 2*np.pi * (k + 1) * t_data[j] / period
                    X_mat[j, (1 + 2*n_t) * i + 2*k + 1] = x_pow * np.sin(theta)
                    X_mat[j, (1 + 2*n_t) * i + 2*k + 2] = x_pow * np.cos(theta)
        X_mats[param] = X_mat
        beta_hats[param] = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y_vec

    def param_func(param, t, x=None):
        # Input t should be in units of hours
        n_X, n_t = param_info[param]
        X_vec = np.zeros((n_X + 1) * (1 + 2*n_t))
        for i in range(n_X + 1):
            # Allow x to be undefined for beta and kappa
            x_pow = 1 if i == 0 else x ** i
            X_vec[(1 + 2*n_t) * i] = x_pow
            for k in range(n_t):
                theta = 2*np.pi * (k + 1) * t / period
                X_vec[(1 + 2*n_t) * i + 2*k + 1] = x_pow * np.sin(theta)
                X_vec[(1 + 2*n_t) * i + 2*k + 2] = x_pow * np.cos(theta)
        return X_vec @ beta_hats[param]
    
    return param_func


def plot_data(model_df, prec_series):
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

    figure, axes = plt.subplots(4, 1)
    axes = iter(axes.flatten())
    axis = next(axes)
    axis.plot(prec_series, 'mo-')
    axis.set_xlabel('t')
    axis.set_ylabel('prec')
    
    axis = next(axes)
    axis.plot(model_df['t'], model_df['x'], 'ro-')
    axis.set_xlabel('t')
    axis.set_ylabel('x')

    axis = next(axes)
    axis.plot(model_df['s'], model_df['x'], 'ko-')
    axis.set_xlabel('s')
    axis.set_ylabel('x')

    axis = next(axes)
    axis.plot(model_df['t'], model_df['t_delta'], 'go-')
    axis.set_xlabel('t')
    axis.set_ylabel('t_delta')
    plt.show()


def plot_params(x_vec, t_vec, param_func):
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


def build_scheme(param_func, x_data, t_mesh, n, m, delta_s, delta_t,
        non_periodic=False, fwd_diff=False):
    # TODO: consider a different x-domain
    x_inf = x_data.min()
    x_sup = x_data.max()
    delta_x = (x_sup - x_inf) / n
    x_mesh = np.linspace(x_inf, x_sup, n + 1)

    def _beta(t):
        return param_func('beta', t)
    def _v(t, x):
        return np.exp(param_func('psi', t, x))
    def _K(t):
        return np.exp(param_func('kappa', t))
    def peclet_num(x, t):
        return _K(t) * (_beta(t) - x) * delta_x / _v(t, x)
    
    M_mats = [np.zeros((n - 1, n - 1, 4)) for _ in range(m)]
    print('Setting up linear systems...')
    for k in range(n - 1):
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
    M_mats = [mat.sum(axis=2) if fwd_diff or non_periodic \
        else mat[:, :, np.r_[0, 2:4]].sum(axis=2) for mat in M_mats]
    if non_periodic:
        return M_mats, G_mats, H_mats

    # Create LHS matrix
    A_mat = np.zeros((m * (n - 1), m * (n - 1)))
    for j in range(m):
        start = j*(n - 1)
        end = start + n - 1
        A_mat[start : end, start : end] = M_mats[j]
        if j >= 1 and not fwd_diff:
            A_mat[start : end, start - (n - 1) : end - (n - 1)] = 0.5 * H_mats[j - 1]
        if j <= m - 2:
            A_mat[start : end, start + n - 1 : end + n - 1] = -H_mats[j + 1] if fwd_diff \
                else -0.5 * H_mats[j + 1]
    # Periodic boundary condition
    if fwd_diff:
        A_mat[(m - 1)*(n - 1) : m*(n - 1), 0 : n - 1] = -H_mats[0]
    else:
        A_mat[0 : n - 1, (m - 1)*(n - 1) : m*(n - 1)] = 0.5 * H_mats[m - 1]
        A_mat[(m - 1)*(n - 1) : m*(n - 1), 0 : n - 1] = -0.5 * H_mats[0]
    return A_mat, G_mats


def build_scheme_time_indep(param_func, x_data, n, delta_s, _time):
    x_inf = x_data.min()
    x_sup = x_data.max()
    delta_x = (x_sup - x_inf) / n
    x_mesh = np.linspace(x_inf, x_sup, n + 1)

    def _beta(t):
        return param_func('beta', t)
    def _v(t, x):
        return np.exp(param_func('psi', t, x))
    def _K(t):
        return np.exp(param_func('kappa', t))
    def peclet_num(x, t):
        return _K(t) * (_beta(t) - x) * delta_x / _v(t, x)
    
    M_mat = np.zeros((n - 1, n - 1, 4))
    G_mat = np.zeros((n - 1, n - 1))
    for k in range(n - 1):
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
    return M_mat, G_mat
    

def plot_results(u_array, x_data, t_mesh, n, m, z, delta_s, delta_t, param_func, np_start_date=None):
    x_inf = x_data.min()
    x_sup = x_data.max()
    x_mesh = np.linspace(x_inf, x_sup, n + 1)
    s_mesh = np.linspace(0, z * delta_s, z + 1)
    print('x_inf, x_sup, delta_x, delta_t:', x_inf, x_sup, (x_sup - x_inf) / n, delta_t)
    
    # TODO: plot x-boundaries
    figure, axes = plt.subplots(4, 5) if np_start_date else plt.subplots(3, 4)
    axes = iter(axes.flatten())
    u_min = np.Inf
    u_max = -np.Inf
    num_plots = 20 if np_start_date else 12
    t_plot_indices = [floor(i * m / num_plots) for i in range(num_plots)]
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
        date_format = '%b %d %Y' if np_start_date else '%b %d'
        start_date = np_start_date or datetime(2000, 1, 1)
        axis.set_title((start_date + timedelta(days=plot_t * delta_t / 24)).strftime(date_format))
        beta = param_func('beta', plot_t * delta_t)
        axis.plot(s_mesh, [beta for _ in s_mesh], 'r-')
    plt.show()

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
        axis.plot(t_mesh[1:] / 24, [param_func('beta', t) for t in t_mesh[1:]], 'r-')
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

    if np_start_date:
        figure, axis = plt.subplots(1)
        j_inf = u_array.max(axis=(0, 2))
        j_2 = np.zeros(m)
        for j in range(m):
            j_2[j] = (u_array[:, j, :] ** 2).sum(axis=(0, 1))
        j_2 /= (n - 1) * (z + 1)
        axis.plot(t_mesh[1:], j_inf, 'r', label='j_inf')
        axis.plot(t_mesh[1:], j_2, 'b', label='j_2')
        axis.set_xlabel('t')
        axis.set_ylabel('u')
        axis.legend()
        plt.show()
    

def plot_results_time_indep(u_arrays, x_data, n, z, delta_s, param_func, title=None):
    x_inf = x_data.min()
    x_sup = x_data.max()
    x_mesh = np.linspace(x_inf, x_sup, n + 1)
    s_mesh = np.linspace(0, z * delta_s, z + 1)
    print('x_inf, x_sup, delta_x:', x_inf, x_sup, (x_sup - x_inf) / n)

    def _beta(t):
        return param_func('beta', t)
    def _v(t, x):
        return np.exp(param_func('psi', t, x))
    def _K(t):
        return np.exp(param_func('kappa', t))
    def build_title(t):
        date_part = (datetime(2000, 1, 1) + timedelta(days=int(t))).strftime('%b %d')
        if title:
            return f'{title}: {date_part}'
        return date_part

    figure, axes = plt.subplots(3, 4)
    axes = iter(axes.flatten())
    u_min = np.Inf
    u_max = -np.Inf
    t_indices = np.arange(0, 360, 30)
    for i, t in enumerate(t_indices):
        u_slice = u_arrays[i]
        u_min = np.min([u_min, u_slice.min(axis=(0, 1))])
        u_max = np.max([u_max, u_slice.max(axis=(0, 1))])
    for i, t in enumerate(t_indices):
        axis = next(axes)
        X, S = np.meshgrid(x_mesh[1 : -1], s_mesh)
        cmap = axis.pcolormesh(S, X, u_arrays[i], cmap='viridis',
            vmin=u_min, vmax=u_max)
        plt.colorbar(cmap)
        axis.set_xlabel('s')
        axis.set_ylabel('x')
        axis.set_title(build_title(t))
        _time = t * 24
        beta = _beta(_time)
        print(f't = {t} days: beta = {beta}, K = {_K(_time)}, v at x_inf = { _v(_time, x_inf)},'
            f' v at x_sup = { _v(_time, x_sup)}')
        axis.plot(s_mesh, [beta for _ in s_mesh], 'r-')
    plt.show()

    figure, axes = plt.subplots(3, 4)
    axes = iter(axes.flatten())
    u_min = np.Inf
    u_max = -np.Inf
    t_indices = np.arange(0, 360, 30)
    for i, t in enumerate(t_indices):
        axis = next(axes)
        j_inf = u_arrays[0].max(axis=1)
        j_2 = np.zeros(z + 1)
        for j in range(z + 1):
            j_2[j] = (u_arrays[i][j, :] ** 2).sum()
        j_2 /= (n - 1)
        axis.plot(s_mesh, j_inf, 'r', label='j_inf')
        axis.plot(s_mesh, j_2, 'b', label='j_2')
        axis.set_xlabel('s')
        axis.set_ylabel('u')
        axis.set_title(build_title(t))
        axis.legend()
    plt.show()


# def phi(x, t, k):
#     [x_km1, x_k, x_kp1] = x_data[k - 1 : k + 2]
#     x_km0p5 = (x_km1 + x_k) / 2
#     x_kp0p5 = (x_k + x_kp1) / 2
#     if x > x_km1 and x <= x_k:
#         pe = peclet_num(x_km0p5, t)
#         return ((x - x_km1) / delta_x) ** np.exp(pe)
#     if x > x_k and x <= x_kp1:
#         pe = peclet_num(x_kp0p5, t)
#         return ((x_kp1 - x) / delta_x) ** np.exp(-pe)
#     return 0