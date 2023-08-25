from datetime import datetime, timedelta
from math import floor
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.tsatools import detrend

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

DEFAULT_PARAM_INFO = {
    'beta': (0, 3),
    'psi': (2, 2),
    # 'psi': (0, 2),
    'kappa': (0, 2),
}

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


def deseasonalise_x(model_df, param_func):
    t0 = datetime(2000, 1, 1)
    def x_periodic(t):
        return param_func('beta', 24 * (t - t0).days)
    x_seasonal = model_df['t'].apply(x_periodic)
    x_deseasonalised = pd.Series(model_df['x'] - x_seasonal)
    x_deseasonalised.index = model_df['t']
    return x_deseasonalised


def detrend_x(x_series, rolling=None, polynomial=1):
    if rolling:
        x_rolling_mean = x_series.rolling(window=rolling, center=True).mean().fillna(0)
        return pd.Series(x_series - x_rolling_mean, index=x_series.index)
    return pd.Series(detrend(x_series, order=polynomial), index=x_series.index)


# Estimate parameter functions by estimating least squares coefficients for fitting the
# functions as polynomials in x multiplied by sin/cos in t
# Number of least squares coefficients estimated for each parameter is (n_X + 1) * (1 + 2*n_t)
def calculate_param_coeffs(model_df, period, shift_zero=False, param_info=None):
    n = model_df.shape[0] - 1
    x_data = np.array(model_df['x'])
    s_data = np.array(model_df['s'])
    model_df['t'] = pd.to_datetime(model_df['t'])
    if shift_zero:
        # Zero for time at 1 January
        t_0 = datetime(model_df['t'][0].year, 1, 1)
    else:
        t_0 = model_df['t'][0]
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
        param_info = dict(DEFAULT_PARAM_INFO)
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
    return beta_hats

def calculate_param_func(model_df, period, beta_hats, param_info=None, trend_polynomial=None):
    if param_info is None:
        param_info = dict(DEFAULT_PARAM_INFO)
    def _param_func(param, t, x=None):
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
    if trend_polynomial:
        # TODO: Handle this for quadratic and moving average trends
        x_deseasonalised = deseasonalise_x(model_df, _param_func)
        x_detrended = detrend_x(x_deseasonalised, polynomial=1)
        trend = x_deseasonalised - x_detrended
        trend_slope = (trend.iloc[1] - trend.iloc[0]) / \
            (24 * (trend.index[1] - trend.index[0]).days)
        def _trend_param_func(param, t, x=None):
            if param == 'beta':
                return _param_func(param, t, x) - trend_slope * t
            return _param_func(param, t, x)
        return _trend_param_func
    return _param_func


def plot_data(model_df, prec_series, x_inf, x_sup):
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
    axis.plot(model_df['t'], model_df['x'] * 0 + x_sup, 'c-')
    axis.plot(model_df['t'], model_df['x'] * 0 + x_inf, 'c-')
    axis.set_xlabel('t')
    axis.set_ylabel('x')

    axis = next(axes)
    axis.plot(model_df['s'], model_df['x'], 'ko-')
    axis.plot(model_df['s'], model_df['x'] * 0 + x_sup, 'c-')
    axis.plot(model_df['s'], model_df['x'] * 0 + x_inf, 'c-')
    axis.set_xlabel('s')
    axis.set_ylabel('x')

    axis = next(axes)
    axis.plot(model_df['t'], model_df['t_delta'], 'go-')
    axis.set_xlabel('t')
    axis.set_ylabel('t_delta')
    plt.show()


def plot_params(model_df, t_mesh, param_func):
    x_inf = np.min(model_df['x'])
    x_median = np.median(model_df['x'])
    x_sup = np.max(model_df['x'])

    figure, axes = plt.subplots(3, 1)
    axes = iter(axes.T.flatten())
    axis = next(axes)
    axis.plot(t_mesh, [param_func('beta', i * 24) for i, t in enumerate(t_mesh)], 'bo-')
    axis.set_xlabel('t')
    axis.set_ylabel('beta')

    axis = next(axes)
    axis.plot(t_mesh, [param_func('kappa', i * 24) for i, t in enumerate(t_mesh)], 'go-')
    axis.set_xlabel('t')
    axis.set_ylabel('kappa = log(K)')

    axis = next(axes)
    axis.plot(t_mesh, [param_func('psi', i * 24, x_inf) for i, t in enumerate(t_mesh)],
        'mo-', label='at x_inf')
    axis.plot(t_mesh, [param_func('psi', i * 24, x_median) for i, t in enumerate(t_mesh)],
        'ko-', label='at x_median')
    axis.plot(t_mesh, [param_func('psi', i * 24, x_sup) for i, t in enumerate(t_mesh)],
        'co-', label='at x_sup')
    axis.set_xlabel('t')
    axis.set_ylabel('psi = log(v)')
    axis.legend()
    plt.show()
    
    figure, axis = plt.subplots(1)
    x_mesh = np.linspace(model_df['x'].min(), model_df['x'].max(), 201)
    _psi = lambda i, x: param_func('psi', i * 24, x)
    X, T = np.meshgrid(x_mesh, t_mesh)
    U = X * 0
    for i, t in enumerate(t_mesh):
        for j, x in enumerate(x_mesh):
            U[i, j] = _psi(i, x)
    cmap = axis.pcolormesh(T, X, U, cmap='viridis',
        vmin=U.min().min(), vmax=U.max().max())
    plt.colorbar(cmap)
    axis.set_xlabel('t')
    axis.set_ylabel('x')
    axis.set_title('psi = log(v)')
    plt.show()


def build_scheme(param_func, t_mesh, n, m, delta_s, delta_t, non_periodic=False,
        x_inf=None, x_sup=None, side=None):
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
    
    M_mats = [np.zeros((n + 1, n + 1, 4)) for _ in range(m)]
    for k in range(n + 1):
        x_km1 = x_mesh[k - 1] if k > 0 else None
        x_k = x_mesh[k]
        x_kp1 = x_mesh[k + 1] if k < n else None
        x_km0p5 = (x_km1 + x_k) / 2 if k > 0 else None
        x_kp0p5 = (x_k + x_kp1) / 2 if k < n else None
        for j in range(m):
            t = t_mesh[j]
            pe_l = peclet_num(x_km0p5, t) if k > 0 else None
            pe_r = peclet_num(x_kp0p5, t) if k < n else None
            p_l = np.exp(pe_l) if k > 0 else None
            p_r = np.exp(-pe_r) if k < n else None

            # Contributions from phi du/ds term
            f = lambda _x, _t: 1 / _v(_t, _x)
            if k == 0:
                M_mats[j][k, k, 0] = -delta_x**2 / delta_s * f(x_kp0p5, t) \
                    / (p_r + 2)
            elif k == n:
                M_mats[j][k, k, 0] = -delta_x**2 / delta_s * f(x_km0p5, t) \
                    / (p_l + 2)
            else:
                M_mats[j][k, k, 0] = -delta_x**2 / delta_s \
                    * (f(x_km0p5, t) / (p_l + 2) + f(x_kp0p5, t) / (p_r + 2))
            if k > 0:
                M_mats[j][k, k - 1, 0] = -delta_x**2 / delta_s * f(x_km0p5, t) \
                    / (p_l + 1) / (p_l + 2)
            if k < n:
                M_mats[j][k, k + 1, 0] = -delta_x**2 / delta_s * f(x_kp0p5, t) \
                    / (p_r + 1) / (p_r + 2)

            # Contributions from phi du/dt term
            f = lambda _x, _t: np.exp(-_x) / _v(_t, _x)
            if k == 0:
                M_mats[j][k, k, 1] = -delta_x**2 / delta_t * f(x_kp0p5, t) \
                    / (p_r + 2)
            elif k == n:
                M_mats[j][k, k, 1] = -delta_x**2 / delta_t * f(x_km0p5, t) \
                    / (p_l + 2)
            else:
                M_mats[j][k, k, 1] = -delta_x**2 / delta_t \
                    * (f(x_km0p5, t) / (p_l + 2) + f(x_kp0p5, t) / (p_r + 2))
            if k > 0:
                M_mats[j][k, k - 1, 1] = -delta_x**2 / delta_t * f(x_km0p5, t) \
                    / (p_l + 1) / (p_l + 2)
            if k < n:
                M_mats[j][k, k + 1, 1] = -delta_x**2 / delta_t * f(x_kp0p5, t) \
                    / (p_r + 1) / (p_r + 2)

            # Contributions from phi du/dx term
            if k == 0:
                M_mats[j][k, k, 2] = -pe_r / (p_r + 1)
            elif k == n:
                M_mats[j][k, k, 2] = pe_l / (p_l + 1)
            else:
                M_mats[j][k, k, 2] = pe_l / (p_l + 1) - pe_r / (p_r + 1)
            if k > 0:
                M_mats[j][k, k - 1, 2] = -pe_l / (p_l + 1)
            if k < n:
                M_mats[j][k, k + 1, 2] = pe_r / (p_r + 1)

            # Contributions from dphi/dx du/dx term
            if k == 0 or k == n:
                M_mats[j][k, k, 3] = -1/2
            else:
                M_mats[j][k, k, 3] = -1
            if k > 0:
                M_mats[j][k, k - 1, 3] = 1/2
            if k < n:
                M_mats[j][k, k + 1, 3] = 1/2

    # For one-sided, there is a Dirichlet BC at the given x end and a Neumann
    # BC at the other x end, so keep the other end's term in the scheme
    # For two-sided, both ends have Dirichlet BCs, so remove both ends' terms
    sliced_M_mats = []
    for mat in M_mats:
        if side == 'left':
            sliced_M_mats.append(mat[1:, 1:, :])
        elif side == 'right':
            sliced_M_mats.append(mat[:n, :n, :])
        else:
            sliced_M_mats.append(mat[1 : n, 1 : n, :])
    # M_mat: at current t and current s, indexed by x
    # G_mat: at current t and previous s, indexed by x
    # H_mat: at next t and current s, indexed by x
    G_mats = [mat[:, :, 0] for mat in sliced_M_mats]
    H_mats = [mat[:, :, 1] for mat in sliced_M_mats]
    sliced_M_mats = [mat.sum(axis=2) for mat in sliced_M_mats]
    if non_periodic:
        return sliced_M_mats, G_mats, H_mats

    # Create the LHS matrix for the periodic BCs case, where each row of
    # sub-matrices represents the system at a value of
    # TODO: check that this correctly implements one-sided for periodic BCs
    x_size = n if side else n - 1
    A_mat = np.zeros((m * x_size, m * x_size))
    # A_mat: at current s, indexed by t and x
    for j in range(m):
        start = j * x_size
        end = (j + 1) * x_size
        A_mat[start : end, start : end] = sliced_M_mats[j]
        if j >= 1:
            A_mat[start : end, start - x_size : end - x_size] = 0.5 * H_mats[j - 1]
        if j <= m - 2:
            A_mat[start : end, start + x_size : end + x_size] = -0.5 * H_mats[j + 1]
    # Periodic boundary condition
    A_mat[0 : x_size, (m - 1) * x_size : m * x_size] = 0.5 * H_mats[m - 1]
    A_mat[(m - 1) * x_size : m * x_size, 0 : x_size] = -0.5 * H_mats[0]
    return A_mat, G_mats
    

def plot_results(u_array, x_data, t_mesh, n, m, z, delta_s, delta_t, param_func,
        start_date=None, non_periodic=False, x_inf=None, x_sup=None, title=None,
        side=None):
    u_last_cycle = u_array if non_periodic else u_array[:, -m:, :]
    s_mesh = np.linspace(0, z * delta_s, z + 1)
    x_inf = x_inf or x_data.min()
    x_sup = x_sup or x_data.max()
    x_mesh = np.linspace(x_inf, x_sup, n + 1)
    if side:
        x_slice = x_mesh[1:] if side == 'left' else x_mesh[:-1]
    else:
        x_slice = x_mesh[1 : -1]
    delta_x = (x_sup - x_inf) / n
    print('x_min, x_inf, x_sup, x_max, delta_x, delta_t:',
          x_data.min(), x_inf, x_sup, x_data.max(), delta_x, delta_t)
    _start_date = start_date if non_periodic else datetime(2000, 1, 1)
    
    # TODO: plot x-boundaries
    figure, axes = plt.subplots(4, 5) if non_periodic else plt.subplots(3, 4)
    axes = iter(axes.flatten())
    u_min = np.Inf
    u_max = -np.Inf
    num_plots = 20 if non_periodic else 12
    t_plot_indices = [floor(i * m / num_plots) for i in range(num_plots)]
    for j in t_plot_indices:
        u_slice = u_last_cycle[:, j, :]
        u_min = np.min([u_min, u_slice.min(axis=(0, 1))])
        u_max = np.max([u_max, u_slice.max(axis=(0, 1))])
    for j in t_plot_indices:
        axis = next(axes)
        X, S = np.meshgrid(x_slice, s_mesh)
        cmap = axis.pcolormesh(S, X, u_last_cycle[:, j, :], cmap='viridis',
            vmin=u_min, vmax=u_max)
        plt.colorbar(cmap)
        axis.set_xlabel('s')
        axis.set_ylabel('x')
        date_title = (_start_date + timedelta(days=j * delta_t / 24)).strftime(
            '%b %d %Y' if non_periodic else '%b %d')
        axis.set_title(f'{title}: {date_title}' if title else date_title)
        beta = param_func('beta', j * delta_t)
        axis.plot(s_mesh, [beta for _ in s_mesh], 'r-')
    plt.show()

    t_plot_indices = [floor(i * m / 4) for i in range(4)]
    x_plot_indices = [floor(i * n / 5) for i in range(1, 5)]
    max_prob = np.zeros((4, 4))
    mean = np.zeros((4, 4))
    median = np.zeros((4, 4))
    mode = np.zeros((4, 4))
    colours = ['g', 'b', 'm', 'r']
    figure, axes = plt.subplots(2, 4)
    axes = iter(axes.T.flatten())
    for t_i, j in enumerate(t_plot_indices):
        date_title = (_start_date + timedelta(days=j * delta_t / 24)).strftime('%b %d')
        axis = next(axes)
        cdfs = {}
        for x_i, k in enumerate(x_plot_indices):
            colour = colours[x_i]
            cdfs[k] = 1 - u_last_cycle[:, j, k]
            axis.plot(s_mesh, cdfs[k], colour, label=f'x = {round(x_inf + k * delta_x, 3)}')
            max_prob[t_i, x_i] = cdfs[k][-1]
            median[t_i, x_i] = None if cdfs[k][-1] < 0.5 else \
                np.argwhere(cdfs[k] >= 0.5)[0, 0] * delta_s
        axis.set_title(f'cdf: t = {date_title}')
        axis.legend()

        axis = next(axes)
        for x_i, k in enumerate(x_plot_indices):
            colour = colours[x_i]
            pdf = [(cdfs[k][i + 1] - cdfs[k][i]) / delta_s for i in range(len(cdfs[k]) - 1)]
            axis.plot(s_mesh[:-1], pdf, colour, label=f'x = {round(x_inf + k * delta_x, 3)}')
            mode[t_i, x_i] = np.argmax(pdf) * delta_s
            mean[t_i, x_i] = None if cdfs[k][-1] == 0 else \
                np.sum([s_mesh[i] * p * delta_s / cdfs[k][-1] for i, p in enumerate(pdf)])
        axis.set_xlabel('s')
        axis.set_title(f'pdf: t = {date_title}')
        axis.legend()
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
        X, T = np.meshgrid(x_slice, t_mesh[1:] / 24)
        cmap = axis.pcolormesh(T, X, u_array[plot_s, :, :], cmap='viridis',
            vmin=u_min, vmax=u_max)
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
        j_2[i] = (u_last_cycle[i, :, :] ** 2).sum(axis=(0, 1))
    j_2 /= (n - 1) * m
    axis.plot(s_mesh, j_inf, 'r', label='j_inf')
    axis.plot(s_mesh, j_2, 'b', label='j_2')
    axis.set_xlabel('s')
    axis.set_ylabel('u')
    axis.legend()
    plt.show()

    figure, axes = plt.subplots(1, 2)
    axes = iter(axes.flatten())
    axis = next(axes)
    j_inf = u_array.max(axis=(0, 2))
    j_2 = np.zeros(len(t_mesh) - 1)
    for j in range(len(t_mesh) - 1):
        j_2[j] = (u_array[:, j, :] ** 2).sum(axis=(0, 1))
    j_2 /= (n - 1) * (z + 1)
    axis.plot(t_mesh[1:], j_inf, 'r', label='j_inf')
    axis.plot(t_mesh[1:], j_2, 'b', label='j_2')
    axis.set_xlabel('t')
    axis.set_ylabel('u')
    axis.legend()
    
    axis = next(axes)
    axis.plot([param_func('beta', t) for t in t_mesh[1:]], j_2, 'b')
    axis.set_xlabel('beta')
    axis.set_ylabel('j_2')
    plt.show()


def get_x_domain(x_series, proportion=None, quantile=None, lower_q=None, upper_q=None):
    if proportion:
        shrinkage = (x_series.max() - x_series.min()) / proportion
        x_sup = x_series.max() - shrinkage
    elif quantile:
        x_sup = np.quantile(x_series, 1 - quantile)
    elif lower_q is not None and upper_q is not None:
        x_sup = np.quantile(x_series, upper_q)
    else:
        x_sup = x_series.max()
    # Extreme x_sup:
    # Calculations for x consider the rainfall to be delivered uniformly throughout
    # a given day; instead suppose all such rainfall is delivered in a single hour
    # Hence, add log(24) to x = log(ds/dt)
    # Extreme x_inf:
    # Since the procedure for calculating x asks how long it has been since prec_inc
    # of rainfall has fallen, long time intervals are taken in account
    # However, for a given drought, the actual value is closest to the worst recorded
    # value (as seen by sequentially increasing t_delta)
    # Hence apply a stricter quantile or just take the minimum value
    return np.quantile(x_series, 0.005), x_sup + np.log(24)


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