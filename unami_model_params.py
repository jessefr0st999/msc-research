import ast
import os

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from helpers import get_map, scatter_map, configure_plots
from unami_2009_helpers import prepare_model_df, calculate_param_coeffs

CALCULATE = True
# CALCULATE = False
PERIOD = 24 * 365
PREC_INC = 0.5
FUSED_DAILY_PATH = 'data/fused_upsampled'
DATASET = 'fused_daily_orig'
# DATASET = 'fused_daily_nsrp'

if CALCULATE:
    pathnames = []
    for path in os.scandir(FUSED_DAILY_PATH):
        if DATASET == 'fused_daily_nsrp' and not path.name.startswith('fused_daily_nsrp'):
            continue
        if DATASET == 'fused_daily' and not path.name.endswith('it_3000.csv'):
            continue
        pathnames.append(path.name)

    loc_list = []
    beta_coeffs_list = []
    kappa_coeffs_list = []
    psi_coeffs_list = []
    for pathname in pathnames:
        prec_df = pd.read_csv(f'{FUSED_DAILY_PATH}/{pathname}', index_col=0)
        prec_series = pd.Series(prec_df.values[:, 0], index=pd.DatetimeIndex(prec_df.index))
        loc_list.append(ast.literal_eval(prec_df.columns[0]))
        model_df = prepare_model_df(prec_series, PREC_INC)
        beta_hats = calculate_param_coeffs(model_df, PERIOD, shift_zero=True)
        beta_coeffs_list.append(beta_hats['beta'])
        kappa_coeffs_list.append(beta_hats['kappa'])
        psi_coeffs_list.append(beta_hats['psi'])
    pd.DataFrame(beta_coeffs_list, index=loc_list)\
        .to_csv(f'beta_coeffs_{DATASET}.csv')
    pd.DataFrame(kappa_coeffs_list, index=loc_list)\
        .to_csv(f'kappa_coeffs_{DATASET}.csv')
    pd.DataFrame(psi_coeffs_list, index=loc_list)\
        .to_csv(f'psi_coeffs_{DATASET}.csv')

# Original daily upsampled
beta_coeffs_df_orig = pd.read_csv('beta_coeffs_fused_daily_orig.csv',
    index_col=0, converters={0: ast.literal_eval})
kappa_coeffs_df_orig = pd.read_csv('kappa_coeffs_fused_daily_orig.csv',
    index_col=0, converters={0: ast.literal_eval})
psi_coeffs_df_orig = pd.read_csv('psi_coeffs_fused_daily_orig.csv',
    index_col=0, converters={0: ast.literal_eval})
# NSRP daily upsampled
beta_coeffs_df_nsrp = pd.read_csv('beta_coeffs_fused_daily_nsrp.csv',
    index_col=0, converters={0: ast.literal_eval})
kappa_coeffs_df_nsrp = pd.read_csv('kappa_coeffs_fused_daily_nsrp.csv',
    index_col=0, converters={0: ast.literal_eval})
psi_coeffs_df_nsrp = pd.read_csv('psi_coeffs_fused_daily_nsrp.csv',
    index_col=0, converters={0: ast.literal_eval})
# Original daily upsampled (SAR-corrected)
beta_coeffs_df_orig_corr = pd.read_csv('corrected_beta_coeffs_orig.csv')
kappa_coeffs_df_orig_corr = pd.read_csv('corrected_kappa_coeffs_orig.csv')
psi_coeffs_df_orig_corr = pd.read_csv('corrected_psi_coeffs_orig.csv')
# NSRP daily upsampled (SAR-corrected)
beta_coeffs_df_nsrp_corr = pd.read_csv('corrected_beta_coeffs_nsrp.csv')
kappa_coeffs_df_nsrp_corr = pd.read_csv('corrected_kappa_coeffs_nsrp.csv')
psi_coeffs_df_nsrp_corr = pd.read_csv('corrected_psi_coeffs_nsrp.csv')
loc_list = beta_coeffs_df_nsrp.index.values
lats, lons = list(zip(*loc_list))

# Save with lats and lons in separate columns for R analysis
np.savetxt('array_beta_coeffs_orig.csv', np.column_stack(
    (lats, lons, beta_coeffs_df_orig)), delimiter=',')
np.savetxt('array_kappa_coeffs_orig.csv', np.column_stack(
    (lats, lons, kappa_coeffs_df_orig)), delimiter=',')
np.savetxt('array_psi_coeffs_orig.csv', np.column_stack(
    (lats, lons, psi_coeffs_df_orig)), delimiter=',')
np.savetxt('array_beta_coeffs_nsrp.csv', np.column_stack(
    (lats, lons, beta_coeffs_df_nsrp)), delimiter=',')
np.savetxt('array_kappa_coeffs_nsrp.csv', np.column_stack(
    (lats, lons, kappa_coeffs_df_nsrp)), delimiter=',')
np.savetxt('array_psi_coeffs_nsrp.csv', np.column_stack(
    (lats, lons, psi_coeffs_df_nsrp)), delimiter=',')

def plot_series(axis, series, title, cb_min, cb_max):
    _map = get_map(axis)
    mx, my = _map(lons, lats)
    axis.set_title(title)
    scatter_map(axis, mx, my, series, size_func=lambda x: 15,
        cb_min=cb_min, cb_max=cb_max)

# Beta
beta_minima = [np.min([
    beta_coeffs_df_orig.iloc[:, i].min(),
    beta_coeffs_df_nsrp.iloc[:, i].min(),
    beta_coeffs_df_orig_corr.iloc[:, i].min(),
    beta_coeffs_df_nsrp_corr.iloc[:, i].min(),
]) for i in range(7)]
beta_maxima = [np.max([
    beta_coeffs_df_orig.iloc[:, i].max(),
    beta_coeffs_df_nsrp.iloc[:, i].max(),
    beta_coeffs_df_orig_corr.iloc[:, i].max(),
    beta_coeffs_df_nsrp_corr.iloc[:, i].max(),
]) for i in range(7)]
for series, title in [
    (beta_coeffs_df_orig, 'beta_coeffs original'),
    (beta_coeffs_df_nsrp, 'beta_coeffs nsrp'),
    (beta_coeffs_df_orig_corr, 'beta_coeffs original (corrected)'),
    (beta_coeffs_df_nsrp_corr, 'beta_coeffs nsrp (corrected)'),
]:
    figure, axes = plt.subplots(2, 4, layout='compressed')
    axes = iter(axes.flatten())
    for i in range(7):
        plot_series(next(axes), series.iloc[:, i], f'{title} {i}',
            cb_min=beta_minima[i], cb_max=beta_maxima[i])
    next(axes).axis('off')
    plt.show()

# Kappa
kappa_minima = [np.min([
    kappa_coeffs_df_orig.iloc[:, i].min(),
    kappa_coeffs_df_nsrp.iloc[:, i].min(),
    kappa_coeffs_df_orig_corr.iloc[:, i].min(),
    kappa_coeffs_df_nsrp_corr.iloc[:, i].min(),
]) for i in range(5)]
kappa_maxima = [np.max([
    kappa_coeffs_df_orig.iloc[:, i].max(),
    kappa_coeffs_df_nsrp.iloc[:, i].max(),
    kappa_coeffs_df_orig_corr.iloc[:, i].max(),
    kappa_coeffs_df_nsrp_corr.iloc[:, i].max(),
]) for i in range(5)]
for series, title in [
    (kappa_coeffs_df_orig, 'kappa_coeffs original'),
    (kappa_coeffs_df_nsrp, 'kappa_coeffs nsrp'),
    (kappa_coeffs_df_orig_corr, 'kappa_coeffs original (corrected)'),
    (kappa_coeffs_df_nsrp_corr, 'kappa_coeffs nsrp (corrected)'),
]:
    figure, axes = plt.subplots(2, 3, layout='compressed')
    axes = iter(axes.flatten())
    for i in range(5):
        plot_series(next(axes), series.iloc[:, i], f'{title} {i}',
            cb_min=kappa_minima[i], cb_max=kappa_maxima[i])
    next(axes).axis('off')
    plt.show()

# Psi
psi_minima = [np.min([
    psi_coeffs_df_orig.iloc[:, i].min(),
    psi_coeffs_df_nsrp.iloc[:, i].min(),
    psi_coeffs_df_orig_corr.iloc[:, i].min(),
    psi_coeffs_df_nsrp_corr.iloc[:, i].min(),
]) for i in range(15)]
psi_maxima = [np.max([
    psi_coeffs_df_orig.iloc[:, i].max(),
    psi_coeffs_df_nsrp.iloc[:, i].max(),
    psi_coeffs_df_orig_corr.iloc[:, i].max(),
    psi_coeffs_df_nsrp_corr.iloc[:, i].max(),
]) for i in range(15)]
for series, title in [
    (psi_coeffs_df_orig, 'psi_coeffs original'),
    (psi_coeffs_df_nsrp, 'psi_coeffs nsrp'),
    (psi_coeffs_df_orig_corr, 'psi_coeffs original (corrected)'),
    (psi_coeffs_df_nsrp_corr, 'psi_coeffs nsrp (corrected)'),
]:
    figure, axes = plt.subplots(3, 5, layout='compressed')
    axes = iter(axes.flatten())
    for i in range(15):
        plot_series(next(axes), series.iloc[:, i], f'{title} {i}',
            cb_min=psi_minima[i], cb_max=psi_maxima[i])
    plt.show()