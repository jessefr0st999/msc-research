import os
import ast
from datetime import datetime, timedelta
import argparse
from math import floor

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
np.random.seed(0)

from helpers import get_map, scatter_map, prepare_df, configure_plots
from unami_2009_helpers import prepare_model_df, estimate_params, plot_data, plot_params, \
    build_scheme, build_scheme_time_indep, plot_results, plot_results_time_indep

PERIOD = 24 * 365
PATH = 'data_unfused/bom_daily'
START_DATE = datetime(2000, 1, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='fused')
    parser.add_argument('--prec_inc', type=float, default=0.5)
    parser.add_argument('--x_steps', type=int, default=60)
    parser.add_argument('--t_steps', type=int, default=60)
    parser.add_argument('--s_steps', type=int, default=100)
    parser.add_argument('--num_samples', type=int, default=300)
    parser.add_argument('--delta_s', type=float, default=1)
    parser.add_argument('--use_existing', action='store_true', default=False)
    parser.add_argument('--plot_data', action='store_true', default=False)
    parser.add_argument('--plot_results', action='store_true', default=False)
    parser.add_argument('--decadal', action='store_true', default=False)
    parser.add_argument('--time_indep', action='store_true', default=False)
    args = parser.parse_args()
    
    loc_list = []
    decadal_loc_list = []
    prec_series_list = []
    # prec_series_list receives data sequentially by location
    # For decadal, each location is added as a pair with decade 1 (2) data at even (odd) entries
    if args.dataset == 'fused':
        # Select random locations from the fused dataset
        prec_df, _, _ = prepare_df('data/precipitation', 'FusedData.csv', 'prec')
        for loc, row in prec_df.T.sample(args.num_samples).iterrows():
            loc_list.append(loc[::-1])
            decadal_loc_list.extend([loc[::-1], loc[::-1]])
            if args.decadal:
                prec_series_list.append(row.loc[:'2011-04-01'])
                prec_series_list.append(row.loc['2011-04-01':])
            else:
                prec_series_list.append(row)
    elif args.dataset == 'bom_daily':
        # Select random files (each corresponding to a location) from the BOM daily dataset
        info_df = pd.read_csv('bom_info.csv', index_col=0, converters={0: ast.literal_eval})
        filenames = set(info_df.sample(args.num_samples)['filename'])
        for i, path in enumerate(os.scandir(PATH)):
            if path.is_file() and path.name in filenames:
                prec_df = pd.read_csv(f'{PATH}/{path.name}')
                prec_df = prec_df.dropna(subset=['Rain'])
                prec_df.index = pd.DatetimeIndex(prec_df['Date'])
                loc = (prec_df.iloc[0]['Lon'], -prec_df.iloc[0]['Lat'])
                prec_series = pd.Series(prec_df['Rain']).dropna().loc['2000-01-01':]
                prec_series.name = loc
                loc_list.append(loc)
                decadal_loc_list.extend([loc, loc])
                if args.decadal:
                    prec_series_list.append(prec_series.loc[:'2011-04-01'])
                    prec_series_list.append(prec_series.loc['2011-04-01':])
                else:
                    prec_series_list.append(prec_series)
            
    n = args.x_steps
    m = args.t_steps
    z = args.s_steps
    median_list = []
    std_list = []
    beta_list = []
    kappa_list = []
    psi_list = []
    # Fit the parameters then solve the DE for each prec series
    # Extract parameter values and useful statistics from the DE solution
    for s, prec_series in enumerate(prec_series_list):
        model_df = prepare_model_df(prec_series, args.prec_inc)
        param_func = estimate_params(model_df, PERIOD)
        x_data = np.array(model_df['x'])
        if args.plot_data:
            plot_data(model_df)
            plot_params(x_data, param_func)
        if args.decadal:
            decade = 1 if s % 2 == 0 else 2
        else:
            decade = None
        model_median = {'decade': decade}
        model_std = {'decade': decade}
        model_beta = {'decade': decade}
        model_kappa = {'decade': decade}
        model_psi = {'decade': decade}
        if args.time_indep:
            u_arrays = []
            # For time-independent, sample parameter values at evenly-spaced times throughout the year
            t_indices = np.arange(0, 360, 90) if args.decadal else np.arange(0, 360, 30)
            for t in t_indices:
                M_mat, G_mat = build_scheme_time_indep(param_func, x_data, n, args.delta_s, t * 24)
                u_vecs = [np.ones(n - 1)]
                for i in range(z):
                    b_vec = G_mat @ u_vecs[i].reshape((n - 1, 1))
                    u_vec = spsolve(M_mat, b_vec)
                    u_vecs.append(u_vec)
                u_array = np.stack(u_vecs, axis=0).reshape((z + 1, n - 1))
                model_median[t] = np.median(u_array[-1, :])
                model_std[t] = np.std(u_array[-1, :])
                model_beta[t] = param_func('beta', t * 24)
                model_kappa[t] = param_func('kappa', t * 24)
                model_psi[t] = param_func('psi', t * 24, 0.5*(x_data.min() + x_data.max()))
                u_arrays.append(u_array)
            median_list.append(model_median)
            std_list.append(model_std)
            beta_list.append(model_beta)
            kappa_list.append(model_kappa)
            psi_list.append(model_psi)
            if args.plot_results:
                plot_results_time_indep(u_arrays, x_data, n, z, args.delta_s, param_func, title=prec_series.name)
        else:
            t_mesh = np.linspace(0, 24 * 365, m + 1)
            delta_t = PERIOD / m
            A_mat, G_mats = build_scheme(param_func, x_data, t_mesh, n, m, args.delta_s, delta_t)
            u_vecs = [np.ones(m * (n - 1))]
            print(f'Series {s} / {len(prec_series_list)}: solving linear systems:')
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
            # For time-dependent, slice the DE solution at evenly-spaced times throughout the year
            t_indices = [floor(i * m / 4) for i in range(4)] if args.decadal \
                else [floor(i * m / 12) for i in range(12)]
            for t in t_indices:
                model_median[t] = np.median(u_array[-1, t, :])
                model_std[t] = np.std(u_array[-1, t, :])
                model_beta[t] = param_func('beta', t * 24)
                model_kappa[t] = param_func('kappa', t * 24)
                model_psi[t] = param_func('psi', t * 24, 0.5*(x_data.min() + x_data.max()))
            median_list.append(model_median)
            std_list.append(model_std)
            beta_list.append(model_beta)
            kappa_list.append(model_kappa)
            psi_list.append(model_psi)
            if args.plot_results:
                plot_results(u_array, x_data, t_mesh, n, m, z, args.delta_s, delta_t, param_func)

    median_df = pd.DataFrame(median_list, index=decadal_loc_list if args.decadal else loc_list)
    std_df = pd.DataFrame(std_list, index=decadal_loc_list if args.decadal else loc_list)
    beta_df = pd.DataFrame(beta_list, index=decadal_loc_list if args.decadal else loc_list)
    kappa_df = pd.DataFrame(kappa_list, index=decadal_loc_list if args.decadal else loc_list)
    psi_df = pd.DataFrame(psi_list, index=decadal_loc_list if args.decadal else loc_list)
    lons, lats = list(zip(*loc_list))

    def df_min(df):
        return df.drop('decade', axis=1).min().min()
    def df_max(df):
        return df.drop('decade', axis=1).max().max()
    for df, label, _min, _max, show_s in [
        (median_df, 'median', 0, 1, True),
        (std_df, 'std', 0, df_max(std_df), True),
        (beta_df, 'beta', df_min(beta_df), df_max(beta_df), False),
        (kappa_df, 'kappa', df_min(kappa_df), df_max(kappa_df), False),
        (psi_df, 'psi at middle x', df_min(psi_df), df_max(psi_df), False),
    ]:
        figure, axes = plt.subplots(3, 4, layout='compressed')
        axes = iter(axes.T.flatten()) if args.decadal else iter(axes.flatten())
        for i, t in enumerate(df.columns[1:]):
            axis = next(axes)
            _map = get_map(axis)
            mx, my = _map(lons, lats)
            days = int(t) if args.time_indep else int(t) * delta_t / 24
            date_part = (datetime(2000, 1, 1) + timedelta(days=days)).strftime('%b %d')
            title = f'{label} at t = {date_part}'
            # For each analysed time, plot the results
            # For decadal, do this for each decade as well as the decadal difference
            if args.decadal:
                if show_s and i == 0:
                    axis.set_title(f'(s = {args.s_steps * args.delta_s}) {title} (decade 1)')
                else:
                    axis.set_title(f'{title} (decade 1)')
                scatter_map(axis, mx, my, df[df['decade'] == 1][t], cb_min=_min,
                    cb_max=_max, size_func=lambda series: 15, cmap='inferno_r')
                
                axis = next(axes)
                _map = get_map(axis)
                axis.set_title(f'{title} (decade 2)')
                scatter_map(axis, mx, my, df[df['decade'] == 2][t], cb_min=_min,
                    cb_max=_max, size_func=lambda series: 15, cmap='inferno_r')
                
                decadal_diff = df[df['decade'] == 2][t] - df[df['decade'] == 1][t]
                axis = next(axes)
                _map = get_map(axis)
                axis.set_title(f'{title} (d2 - d1)')
                scatter_map(axis, mx, my, decadal_diff, cb_min=-np.abs(decadal_diff).max(),
                    cb_max=np.abs(decadal_diff).max(), size_func=lambda series: 15, cmap='RdYlBu')
            else:
                if show_s and i == 0:
                    axis.set_title(f'(s = {args.s_steps * args.delta_s}) {title}')
                else:
                    axis.set_title(title)
                scatter_map(axis, mx, my, df[t], cb_min=_min,
                    cb_max=_max, size_func=lambda series: 15, cmap='inferno_r')
        plt.show()
    
if __name__ == '__main__':
    main()