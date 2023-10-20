import ast
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from helpers import get_map, scatter_map
from unami_2009_helpers import prepare_model_df, calculate_param_coeffs, get_x_domain

# CALCULATE = True
CALCULATE = False
PERIOD = 24 * 365
PREC_INC = 0.2
FUSED_DAILY_PATH = 'data/fused_upsampled'
# DECADE = 1
DECADE = 2
# DECADE = None

file_label = 'nsrp'
if DECADE == 1:
    file_label += '_d1'
elif DECADE == 2:
    file_label += '_d2'

if CALCULATE:
    pathnames = []
    for path in os.scandir(FUSED_DAILY_PATH):
        if not path.name.startswith('fused_daily_nsrp'):
            continue
        pathnames.append(path.name)

    loc_list = []
    beta_coeffs_list = []
    kappa_coeffs_list = []
    psi_coeffs_list = []
    x_inf_list = []
    x_sup_list = []
    for pathname in pathnames:
        prec_df = pd.read_csv(f'{FUSED_DAILY_PATH}/{pathname}', index_col=0)
        prec_series = pd.Series(prec_df.values[:, 0], index=pd.DatetimeIndex(prec_df.index))
        if DECADE == 1:
            prec_series = prec_series[:'2011-03-31']
        elif DECADE == 2:
            prec_series = prec_series['2011-04-01':]
        loc_list.append(ast.literal_eval(prec_df.columns[0]))
        model_df = prepare_model_df(prec_series, PREC_INC)
        beta_hats = calculate_param_coeffs(model_df, PERIOD, shift_zero=True)
        beta_coeffs_list.append(beta_hats['beta'])
        kappa_coeffs_list.append(beta_hats['kappa'])
        psi_coeffs_list.append(beta_hats['psi'])
        x_inf, x_sup = get_x_domain(model_df['x'], quantile=0.005)
        x_inf_list.append(x_inf)
        x_sup_list.append(x_sup)
    pd.DataFrame(beta_coeffs_list, index=loc_list)\
        .to_csv(f'beta_coeffs_fused_daily_{file_label}.csv')
    pd.DataFrame(kappa_coeffs_list, index=loc_list)\
        .to_csv(f'kappa_coeffs_fused_daily_{file_label}.csv')
    pd.DataFrame(psi_coeffs_list, index=loc_list)\
        .to_csv(f'psi_coeffs_fused_daily_{file_label}.csv')
    pd.DataFrame(x_inf_list, index=loc_list)\
        .to_csv(f'x_inf_fused_daily_{file_label}.csv')
    pd.DataFrame(x_sup_list, index=loc_list)\
        .to_csv(f'x_sup_fused_daily_{file_label}.csv')

# NSRP daily upsampled
beta_coeffs_df_nsrp = pd.read_csv(f'beta_coeffs_fused_daily_{file_label}.csv',
    index_col=0, converters={0: ast.literal_eval})
kappa_coeffs_df_nsrp = pd.read_csv(f'kappa_coeffs_fused_daily_{file_label}.csv',
    index_col=0, converters={0: ast.literal_eval})
psi_coeffs_df_nsrp = pd.read_csv(f'psi_coeffs_fused_daily_{file_label}.csv',
    index_col=0, converters={0: ast.literal_eval})
x_inf_df_nsrp = pd.read_csv(f'x_inf_fused_daily_{file_label}.csv',
    index_col=0, converters={0: ast.literal_eval})
x_sup_df_nsrp = pd.read_csv(f'x_sup_fused_daily_{file_label}.csv',
    index_col=0, converters={0: ast.literal_eval})
loc_list = x_inf_df_nsrp.index.values
lats, lons = list(zip(*loc_list))

# Save with lats and lons in separate columns for R analysis
np.savetxt(f'array_beta_coeffs_{file_label}.csv', np.column_stack(
    (lats, lons, beta_coeffs_df_nsrp)), delimiter=',')
np.savetxt(f'array_kappa_coeffs_{file_label}.csv', np.column_stack(
    (lats, lons, kappa_coeffs_df_nsrp)), delimiter=',')
np.savetxt(f'array_psi_coeffs_{file_label}.csv', np.column_stack(
    (lats, lons, psi_coeffs_df_nsrp)), delimiter=',')
np.savetxt(f'array_x_inf_{file_label}.csv', np.column_stack(
    (lats, lons, x_inf_df_nsrp)), delimiter=',')
np.savetxt(f'array_x_sup_{file_label}.csv', np.column_stack(
    (lats, lons, x_sup_df_nsrp)), delimiter=',')

# NSRP daily upsampled (SAR-corrected)
beta_coeffs_df_nsrp_corr = pd.read_csv(f'corrected_beta_coeffs_{file_label}.csv')
kappa_coeffs_df_nsrp_corr = pd.read_csv(f'corrected_kappa_coeffs_{file_label}.csv')
psi_coeffs_df_nsrp_corr = pd.read_csv(f'corrected_psi_coeffs_{file_label}.csv')
x_inf_df_nsrp_corr = pd.read_csv(f'corrected_x_inf_{file_label}.csv')
x_sup_df_nsrp_corr = pd.read_csv(f'corrected_x_sup_{file_label}.csv')

def plot_series(axis, series, title, cb_min, cb_max, cmap='inferno_r',
        size_func=lambda x: 25):
    _map = get_map(axis)
    mx, my = _map(lons, lats)
    axis.set_title(title)
    scatter_map(axis, mx, my, series, size_func=size_func,
        cb_min=cb_min, cb_max=cb_max, cmap=cmap)
    
# x_inf and x_sup
figure, axes = plt.subplots(2, 3, layout='compressed')
axes = iter(axes.flatten())
x_inf_df_nsrp_corr.columns = x_inf_df_nsrp.columns
x_inf_df_nsrp_corr.index = x_inf_df_nsrp.index
x_inf_min = np.min([x_inf_df_nsrp.min(), x_inf_df_nsrp_corr.min()])
x_inf_max = np.max([x_inf_df_nsrp.max(), x_inf_df_nsrp_corr.max()])
for series, title in [
    (x_inf_df_nsrp, 'x_inf nsrp'),
    (x_inf_df_nsrp_corr, 'x_inf nsrp (corrected)'),
]:
    plot_series(next(axes), series, title,
        cb_min=x_inf_min, cb_max=x_inf_max)
series = x_inf_df_nsrp - x_inf_df_nsrp_corr
plot_series(next(axes), np.abs(series), f'x_inf nsrp diff',
    cb_min=0, cb_max=np.quantile(np.abs(series), 0.99))

x_sup_df_nsrp_corr.columns = x_sup_df_nsrp.columns
x_sup_df_nsrp_corr.index = x_sup_df_nsrp.index
x_sup_min = np.min([x_sup_df_nsrp.min(), x_sup_df_nsrp_corr.min()])
x_sup_max = np.max([x_sup_df_nsrp.max(), x_sup_df_nsrp_corr.max()])
for series, title in [
    (x_sup_df_nsrp, 'x_sup nsrp'),
    (x_sup_df_nsrp_corr, 'x_sup nsrp (corrected)'),
]:
    plot_series(next(axes), series, title,
        cb_min=x_sup_min, cb_max=x_sup_max)
series = x_sup_df_nsrp - x_sup_df_nsrp_corr
plot_series(next(axes), np.abs(series), f'x_sup nsrp diff',
    cb_min=0, cb_max=np.quantile(np.abs(series), 0.99))
plt.show()

# Beta
beta_minima = [np.min([
    beta_coeffs_df_nsrp.iloc[:, i].min(),
    beta_coeffs_df_nsrp_corr.iloc[:, i].min(),
]) for i in range(7)]
beta_maxima = [np.max([
    beta_coeffs_df_nsrp.iloc[:, i].max(),
    beta_coeffs_df_nsrp_corr.iloc[:, i].max(),
]) for i in range(7)]
for series, title in [
    (beta_coeffs_df_nsrp, 'beta_coeffs nsrp'),
    (beta_coeffs_df_nsrp_corr, 'beta_coeffs nsrp (corrected)'),
]:
    figure, axes = plt.subplots(2, 4, layout='compressed')
    axes = iter(axes.flatten())
    for i in range(7):
        plot_series(next(axes), series.iloc[:, i], f'{title} {i}',
            cb_min=beta_minima[i], cb_max=beta_maxima[i])
    next(axes).axis('off')
    plt.show()
figure, axes = plt.subplots(2, 4, layout='compressed')
axes = iter(axes.flatten())
beta_coeffs_df_nsrp_corr.columns = beta_coeffs_df_nsrp.columns
beta_coeffs_df_nsrp_corr.index = beta_coeffs_df_nsrp.index
series = beta_coeffs_df_nsrp - beta_coeffs_df_nsrp_corr
for i in range(7):
    _max = np.quantile(np.abs(series.iloc[:, i]), 0.99)
    # plot_series(next(axes), series.iloc[:, i], f'beta_coeffs nsrp diff {i}',
    #     cb_min=-_max, cb_max=_max, cmap='RdYlBu_r')
    plot_series(next(axes), np.abs(series.iloc[:, i]),
        f'beta_coeffs nsrp diff {i}', cb_min=0, cb_max=_max)
next(axes).axis('off')
plt.show()

# Kappa
kappa_minima = [np.min([
    kappa_coeffs_df_nsrp.iloc[:, i].min(),
    kappa_coeffs_df_nsrp_corr.iloc[:, i].min(),
]) for i in range(5)]
kappa_maxima = [np.max([
    kappa_coeffs_df_nsrp.iloc[:, i].max(),
    kappa_coeffs_df_nsrp_corr.iloc[:, i].max(),
]) for i in range(5)]
for series, title in [
    (kappa_coeffs_df_nsrp, 'kappa_coeffs nsrp'),
    (kappa_coeffs_df_nsrp_corr, 'kappa_coeffs nsrp (corrected)'),
]:
    figure, axes = plt.subplots(2, 3, layout='compressed')
    # figure, axes = plt.subplots(1, 5, layout='compressed')
    axes = iter(axes.flatten())
    for i in range(5):
        plot_series(next(axes), series.iloc[:, i], f'{title} {i}',
            cb_min=kappa_minima[i], cb_max=kappa_maxima[i],
            size_func=lambda x: 25)
    next(axes).axis('off')
    plt.show()
figure, axes = plt.subplots(2, 3, layout='compressed')
axes = iter(axes.flatten())
kappa_coeffs_df_nsrp_corr.columns = kappa_coeffs_df_nsrp.columns
kappa_coeffs_df_nsrp_corr.index = kappa_coeffs_df_nsrp.index
series = kappa_coeffs_df_nsrp - kappa_coeffs_df_nsrp_corr
for i in range(5):
    _max = np.quantile(np.abs(series.iloc[:, i]), 0.99)
    # plot_series(next(axes), series.iloc[:, i], f'kappa_coeffs nsrp diff {i}',
    #     cb_min=-_max, cb_max=_max, cmap='RdYlBu_r')
    plot_series(next(axes), np.abs(series.iloc[:, i]),
        f'kappa_coeffs nsrp diff {i}', cb_min=0, cb_max=_max,
        size_func=lambda x: 25)
next(axes).axis('off')
plt.show()

# Psi
psi_minima = [np.min([
    psi_coeffs_df_nsrp.iloc[:, i].min(),
    psi_coeffs_df_nsrp_corr.iloc[:, i].min(),
]) for i in range(15)]
psi_maxima = [np.max([
    psi_coeffs_df_nsrp.iloc[:, i].max(),
    psi_coeffs_df_nsrp_corr.iloc[:, i].max(),
]) for i in range(15)]
for series, title in [
    (psi_coeffs_df_nsrp, 'psi_coeffs nsrp'),
    (psi_coeffs_df_nsrp_corr, 'psi_coeffs nsrp (corrected)'),
]:
    figure, axes = plt.subplots(3, 5, layout='compressed')
    axes = iter(axes.flatten())
    for i in range(15):
        plot_series(next(axes), series.iloc[:, i], f'{title} {i}',
            cb_min=psi_minima[i], cb_max=psi_maxima[i],
            size_func=lambda x: 10)
    plt.show()
figure, axes = plt.subplots(3, 5, layout='compressed')
axes = iter(axes.flatten())
psi_coeffs_df_nsrp_corr.columns = psi_coeffs_df_nsrp.columns
psi_coeffs_df_nsrp_corr.index = psi_coeffs_df_nsrp.index
series = psi_coeffs_df_nsrp - psi_coeffs_df_nsrp_corr
for i in range(15):
    _max = np.quantile(np.abs(series.iloc[:, i]), 0.99)
    # plot_series(next(axes), series.iloc[:, i], f'psi_coeffs nsrp diff {i}',
    #     cb_min=-_max, cb_max=_max, cmap='RdYlBu_r')
    plot_series(next(axes), np.abs(series.iloc[:, i]),
        f'psi_coeffs nsrp diff {i}', cb_min=0, cb_max=_max,
        size_func=lambda x: 10)
plt.show()
