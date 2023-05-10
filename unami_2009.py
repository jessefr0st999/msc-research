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
    args = parser.parse_args()

    prec_df, lats, lons = prepare_df(args.data_dir, args.data_file, 'prec')
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
    filename = f'{args.data_dir}/{OUTPUT_FILE}'
    # model_df.to_csv(filename)
    # print(f'model_df saved to file {filename}')

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

    def param_func(param, t, x=None):
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

    print(X_mats['beta'].shape, beta_hats['beta'].shape)
    print(X_mats['psi'].shape, beta_hats['psi'].shape)
    print(X_mats['kappa'].shape, beta_hats['kappa'].shape)

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