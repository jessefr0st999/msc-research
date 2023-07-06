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
from unami_2009_helpers import prepare_model_df, estimate_params, \
    build_scheme, plot_results, get_x_domain, deseasonalise_x, detrend_x

PERIOD = 24 * 365
BOM_DAILY_PATH = 'data_unfused/bom_daily'
FUSED_DAILY_PATH = 'data/fused_upsampled'
START_DATE = datetime(2000, 1, 1)
PLOT_LOCATIONS = {
    (-37.75, 145.5): 'Melbourne',
    (-12.75, 131.5): 'Darwin',
    (-17.75, 140.5): 'NW Qld',
    (-28.75, 153.5): 'Lismore',
    (-25.75, 133.5): 'Central Australia',
    (-33.25, 151.5): 'Central Coast',
    (-31.75, 116.5): 'Perth',
    (-17.75, 122.5): 'Broome',
    (-35.25, 138.5): 'Adelaide',
    (-42.75, 147.5): 'Hobart',
    (-27.75, 152.5): 'Brisbane',
    (-16.75, 145.5): 'Cairns',
}
PLOT_LOCATIONS = None
if PLOT_LOCATIONS:
    mpl.rcParams.update({'axes.titlesize': 10})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_folder', default=None)
    parser.add_argument('--dataset', default='fused')
    parser.add_argument('--prec_inc', type=float, default=0.5)
    parser.add_argument('--x_steps', type=int, default=50)
    parser.add_argument('--t_steps', type=int, default=50)
    parser.add_argument('--s_steps', type=int, default=100)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--delta_s', type=float, default=10)
    parser.add_argument('--use_existing', action='store_true', default=False)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--np_params', action='store_true', default=False) # TODO: implement
    parser.add_argument('--np_bcs', action='store_true', default=False)
    parser.add_argument('--shrink_x_proportion', type=float, default=None)
    parser.add_argument('--shrink_x_quantile', type=float, default=None)
    parser.add_argument('--shrink_x_mixed', action='store_true', default=False)
    parser.add_argument('--year_cycles', type=int, default=1)
    args = parser.parse_args()
    
    full_loc_list = []
    prec_series_list = []
    # prec_series_list receives data sequentially by location
    # For decadal, each location is added as a pair with decade 1 (2) data at even (odd) entries
    if args.dataset == 'fused':
        # Select random locations from the fused dataset
        prec_df, _, _ = prepare_df('data/precipitation', 'FusedData.csv', 'prec')
        num_samples = args.num_samples if args.num_samples else 1391
        _prec_df = prec_df.T.sample(num_samples)
        for loc, row in _prec_df.iterrows():
            full_loc_list.append(loc)
            prec_series_list.append(row)
    elif args.dataset == 'fused_daily':
        num_samples = args.num_samples if args.num_samples else 1391
        pathnames = [path.name for path in os.scandir(FUSED_DAILY_PATH)]
        pathnames = np.random.choice(pathnames, num_samples, replace=False)
        for pathname in pathnames:
            prec_df = pd.read_csv(f'{FUSED_DAILY_PATH}/{pathname}', index_col=0)
            prec_series = pd.Series(prec_df.values[:, 0], index=pd.DatetimeIndex(prec_df.index))
            loc = ast.literal_eval(prec_df.columns[0])
            full_loc_list.append(loc)
            prec_series_list.append(prec_series)
    elif args.dataset == 'bom_daily':
        # Select random files (each corresponding to a location) from the BOM daily dataset
        info_df = pd.read_csv('bom_info.csv', index_col=0, converters={0: ast.literal_eval})
        filenames = set(info_df.sample(args.num_samples)['filename']) if args.num_samples else None
        for i, path in enumerate(os.scandir(BOM_DAILY_PATH)):
            if path.is_file():
                if args.num_samples and path.name not in filenames:
                    continue
                prec_df = pd.read_csv(f'{BOM_DAILY_PATH}/{path.name}')
                prec_df = prec_df.dropna(subset=['Rain'])
                prec_df.index = pd.DatetimeIndex(prec_df['Date'])
                loc = (-prec_df.iloc[0]['Lat'], prec_df.iloc[0]['Lon'])
                prec_series = pd.Series(prec_df['Rain']).dropna().loc['2000-01-01':]
                prec_series.name = loc
                full_loc_list.append(loc)
                prec_series_list.append(prec_series)
            
    n = args.x_steps
    m = args.t_steps
    z = args.s_steps
    delta_t = PERIOD / m
    last_s_median_list = []
    last_s_std_list = []
    beta_list = []
    kappa_list = []
    psi_list = []
    last_pdf_list = []
    last_cdf_list = []
    mean_list = []
    q1_list = []
    median_list = []
    q3_list = []
    mode_list = []
    mode_pdf_list = []
    j2_list = []
    x_inf_list = []
    x_sup_list = []
    trend_list = []
    # Fit the parameters then solve the DE for each prec series
    # Extract parameter values and useful statistics from the DE solution
    s_mesh = np.linspace(0, z * args.delta_s, z + 1)
    loc_list = []
    # Slice the DE solution at evenly-spaced times throughout the year
    t_indices = [floor(i * m / 12) for i in range(12)]
    # Sample with x close to x_inf, middle and close to x_sup
    x_indices = [
        floor(n / 10),
        floor(n / 5),
        floor(n / 2),
        # floor(9*n / 10),
    ]
    if args.read_folder:
        calculated_locations = []
        for path in os.scandir(f'unami_results/{args.read_folder}'):
            if path.is_file():
                calculated_locations.append(path.name.split('_')[2].split('.npy')[0])
    for s, prec_series in enumerate(prec_series_list):
        loc = full_loc_list[s]
        if PLOT_LOCATIONS and loc not in PLOT_LOCATIONS:
            continue
        if args.read_folder:
            if str(loc) in calculated_locations:
                filename = f'u_array_{str(loc)}.npy'
                print(f'Reading u_array from {filename}...')
                u_array = np.load(f'unami_results/{args.read_folder}/{filename}')
            else:
                continue
        loc_list.append(loc)
        model_df = prepare_model_df(prec_series, args.prec_inc)
        param_func = estimate_params(model_df, PERIOD, shift_zero=True)
        x_deseasonalised = deseasonalise_x(model_df, param_func)
        x_detrended = detrend_x(x_deseasonalised, polynomial=1)
        trend = x_deseasonalised - x_detrended
        x_data = np.array(model_df['x'])
        x_inf_list.append({0: x_data.min()})
        x_sup_list.append({0: x_data.max()})
        trend_list.append({0: trend.iloc[-1] - trend.iloc[0]})
        _last_s_median = {}
        _last_s_std = {}
        _beta = {}
        _kappa = {}
        _psi = {}
        _last_pdf = {}
        _last_cdf = {}
        _mean = {}
        _q1 = {}
        _median = {}
        _q3 = {}
        _mode = {}
        _mode_pdf = {}
        t_mesh = np.linspace(0, args.year_cycles * PERIOD, args.year_cycles * m + 1)
        x_inf, x_sup = get_x_domain(model_df['x'], args.shrink_x_proportion,
            args.shrink_x_quantile, loc if args.shrink_x_mixed else None)
        scheme_output = build_scheme(param_func, t_mesh, n, m, args.delta_s,
            delta_t, args.np_bcs, x_inf=x_inf, x_sup=x_sup)
        if args.np_bcs:
            M_mats, G_mats, H_mats = scheme_output
        else:
            A_mat, G_mats = scheme_output
        u_vecs = [np.ones(m * (n - 1))]
        print(f'Series {s} / {len(prec_series_list)}: solving linear systems:')
        if not args.read_folder and args.np_bcs:
            u_array = np.zeros((z + 1, args.year_cycles * m, n - 1))
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
                            b_vec = G_mats[t_index] @ u_array[i - 1, j, :] + H_mats[t_index] @ u_array[i, j - 1, :]
                            u_array[i, j, :] = spsolve(M_mats[t_index], b_vec)
                        except IndexError:
                            t_index -= m
        elif not args.read_folder:
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
        u_last_cycle = u_array[:, -m:, :]
        # The mean gives expected value of s such that X_s exits the [x_inf, x_sup] domain
        # This requires solving with high enough s that a steady state probability is reached
        # as well as high enough t such that (u, x) falls into a steady cycle
        # TODO: extract PDF value at the mean and median
        # TODO: extract number of year cycles required for steady state
        for j in t_indices:
            _last_s_median[j] = np.median(1 - u_last_cycle[-1, j, :])
            _last_s_std[j] = np.std(1 - u_last_cycle[-1, j, :])
            _beta[j] = param_func('beta', j * PERIOD / m)
            _kappa[j] = param_func('kappa', j * PERIOD / m)
            _psi[j] = param_func('psi', j * PERIOD / m, np.median(x_data))
            for k in x_indices:
                cdf = 1 - u_last_cycle[:, j, k]
                pdf = [(cdf[i + 1] - cdf[i]) / args.delta_s for i in range(len(cdf) - 1)]
                _last_cdf[(j, k)] = cdf[-1]
                # This non-zero indicates the DE should be solved again but for higher s
                _last_pdf[(j, k)] = pdf[-1]
                _mean[(j, k)] = None if cdf[-1] == 0 else \
                    np.sum([s_mesh[i] * p * args.delta_s / cdf[-1] for i, p in enumerate(pdf)])
                _q1[(j, k)] = None if cdf[-1] < 0.25 else \
                    np.argwhere(cdf >= 0.25)[0, 0] * args.delta_s
                _median[(j, k)] = None if cdf[-1] < 0.5 else \
                    np.argwhere(cdf >= 0.5)[0, 0] * args.delta_s
                _q3[(j, k)] = None if cdf[-1] < 0.75 else \
                    np.argwhere(cdf >= 0.75)[0, 0] * args.delta_s
                _mode[(j, k)] = np.argmax(pdf) * args.delta_s
                _mode_pdf[(j, k)] = np.max(pdf)
        last_s_median_list.append(_last_s_median)
        last_s_std_list.append(_last_s_std)
        last_cdf_list.append(_last_cdf)
        last_pdf_list.append(_last_pdf)
        mean_list.append(_mean)
        q1_list.append(_q1)
        median_list.append(_median)
        q3_list.append(_q3)
        mode_list.append(_mode)
        mode_pdf_list.append(_mode_pdf)
        beta_list.append(_beta)
        kappa_list.append(_kappa)
        psi_list.append(_psi)
        j2_list.append({0: (u_last_cycle[-1, :, :] ** 2).sum(axis=(0, 1)) / (n - 1) / m})
        if args.plot:
            plot_results(u_array, x_data, t_mesh, n, m, z, args.delta_s, delta_t,
                param_func, start_date=model_df['t'].iloc[0] if args.np_bcs else None,
                non_periodic=False, x_inf=x_inf, x_sup=x_sup,
                title=f'{PLOT_LOCATIONS[loc]} {str(loc)}' if PLOT_LOCATIONS else str(loc))
        if args.read_folder is None:
            np.save(f'unami_results/u_array_{str(loc)}.npy', u_array)

    last_s_median_df = pd.DataFrame(last_s_median_list, index=loc_list)
    last_s_std_df = pd.DataFrame(last_s_std_list, index=loc_list)
    beta_df = pd.DataFrame(beta_list, index=loc_list)
    kappa_df = pd.DataFrame(kappa_list, index=loc_list)
    psi_df = pd.DataFrame(psi_list, index=loc_list)
    last_cdf_df = pd.DataFrame(last_cdf_list, index=loc_list)
    last_pdf_df = pd.DataFrame(last_pdf_list, index=loc_list)
    mean_df = pd.DataFrame(mean_list, index=loc_list)
    q1_df = pd.DataFrame(q1_list, index=loc_list)
    median_df = pd.DataFrame(median_list, index=loc_list)
    q3_df = pd.DataFrame(q3_list, index=loc_list)
    mode_df = pd.DataFrame(mode_list, index=loc_list)
    mode_pdf_df = pd.DataFrame(mode_pdf_list, index=loc_list)
    j2_df = pd.DataFrame(j2_list, index=loc_list)
    x_inf_df = pd.DataFrame(x_inf_list, index=loc_list)
    x_sup_df = pd.DataFrame(x_sup_list, index=loc_list)
    trend_df = pd.DataFrame(trend_list, index=loc_list)
    lats, lons = list(zip(*loc_list))

    def df_min(df):
        return df.min().min()
    def df_max(df):
        return df.max().max()
    for df, label, _min, _max in [
        (x_inf_df, 'x_inf', df_min(x_inf_df), df_max(x_inf_df)),
        (x_sup_df, 'x_sup', df_min(x_sup_df), df_max(x_sup_df)),
        (x_sup_df - x_inf_df, 'x_sup - x_inf', df_min(x_sup_df - x_inf_df), df_max(x_sup_df - x_inf_df)),
        (trend_df, 'linear trend', -df_max(np.abs(trend_df)), df_max(np.abs(trend_df))),
        (j2_df, 'j2', 0, 1),
    ]:
        figure, axis = plt.subplots(1)
        _map = get_map(axis)
        mx, my = _map(lons, lats)
        axis.set_title(label)
        scatter_map(axis, mx, my, df[0], cb_min=_min,
            cb_max=_max, size_func=lambda series: 15, cmap='RdYlBu_r')
        plt.show()
    for df, label, _min, _max in [
        (last_s_median_df, 'median probability at final s', 0, 1),
        (last_s_std_df, 'std probability at final s', 0, df_max(last_s_std_df)),
        (beta_df, 'beta', df_min(beta_df), df_max(beta_df)),
        (kappa_df, 'kappa', df_min(kappa_df), df_max(kappa_df)),
        (psi_df, 'psi at x median', df_min(psi_df), df_max(psi_df)),
    ]:
        figure, axes = plt.subplots(3, 4, layout='compressed')
        axes = iter(axes.flatten())
        # For each analysed time, plot the results
        for i, j in enumerate(df.columns):
            axis = next(axes)
            _map = get_map(axis)
            mx, my = _map(lons, lats)
            scatter_map(axis, mx, my, df[j], cb_min=_min,
                cb_max=_max, size_func=lambda series: 15, cmap='inferno_r')
            days = int(j) * delta_t / 24
            date_part = (datetime(2000, 1, 1) + timedelta(days=days)).strftime('%b %d')
            axis.set_title(f'{label} at t = {date_part}')
        plt.show()
    for df, label, _min, _max in [
        (last_cdf_df, 'final value of cdf', 0, 1),
        (last_pdf_df, 'final value of pdf', 0, df_max(last_pdf_df)),
        (mean_df, 'mean of pdf', 0, df_max(mean_df)),
        (q1_df, '0.25 quantile of pdf', 0, df_max(q1_df)),
        (median_df, 'median of pdf', 0, df_max(median_df)),
        (q3_df, '0.75 quantile of pdf', 0, df_max(q3_df)),
        (mode_df, 'mode of pdf', 0, df_max(mode_df)),
        (mode_pdf_df, 'pdf value at mode', 0, df_max(mode_pdf_df)),
    ]:
        for k in x_indices:
            figure, axes = plt.subplots(3, 4, layout='compressed')
            axes = iter(axes.flatten())
            # For each analysed time and x-value, plot the results
            for i, (j, _k) in enumerate(df.columns):
                if _k != k:
                    continue
                axis = next(axes)
                _map = get_map(axis)
                mx, my = _map(lons, lats)
                scatter_map(axis, mx, my, df[(j, k)], cb_min=_min,
                    cb_max=_max, size_func=lambda series: 15, cmap='inferno_r')
                days = int(j) * delta_t / 24
                date_part = (datetime(2000, 1, 1) + timedelta(days=days)).strftime('%b %d')
                axis.set_title(f'{label} at t = {date_part}, x = x_inf + {k} * delta_x')
        plt.show()
    
if __name__ == '__main__':
    main()