import argparse

import pandas as pd
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt

from helpers import configure_plots, get_map, scatter_map

DATA_DIR = 'data/precipitation'
PREC_DATA_FILE = f'{DATA_DIR}/FusedData.csv'
LOCATIONS_FILE = f'{DATA_DIR}/Fused.Locations.csv'
locations_df = pd.read_csv(LOCATIONS_FILE)
lats = locations_df['Lat']
lons = locations_df['Lon']

# TODO: check/verify this
def complex_pca(X):
    X_centred = np.array(X - X.mean(axis=0))
    cov = X_centred.conj().T @ X_centred / X.shape[0]
    _, stds, pcs = np.linalg.svd(cov, hermitian=True)
    # evals, evecs = np.linalg.eigh(cov)
    return stds**2, pcs.T

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', default=None)
    parser.add_argument('--pcs_to_plot', type=int, default=4)
    args = parser.parse_args()
    label_size, font_size, show_or_save = configure_plots(args)

    raw_prec_df = pd.read_csv(PREC_DATA_FILE)
    raw_prec_df.columns = pd.to_datetime(raw_prec_df.columns, format='D%Y.%m')
    prec_df = pd.concat([locations_df, raw_prec_df], axis=1)
    prec_df = prec_df.set_index(['Lat', 'Lon']).T
    prec_df_complex = prec_df.apply(hilbert, axis=0)

    vars, pcs = complex_pca(prec_df_complex)
    prop_vars = vars / vars.sum()
    figure, axes = plt.subplots(1, 2, layout='compressed')
    axes[0].plot(list(range(2, 11)), prop_vars[1 : 10])
    axes[0].set_title('PC2 to PC10 explained variance proportion')
    axes[1].plot(list(range(1, len(prop_vars) + 1)), np.log10(prop_vars))
    axes[1].set_title('Log explained variance proportion')
    show_or_save(figure, f'complex_pca_explained_var.png')

    pcs_spatial_phase = np.angle(pcs)
    pcs_spatial_amp = np.abs(pcs)
    pcs_time = np.array(prec_df_complex @ pcs)
    pcs_time_phase = np.angle(pcs_time)
    pcs_time_amp = np.abs(pcs_time)
    
    for i in range(args.pcs_to_plot):
        if i % 2 == 0:
            figure, axes = plt.subplots(2, 4, layout='compressed')
            axes = iter(axes.flatten())
        percent_var = np.round(100 * prop_vars[i], 2)
        axis = next(axes)
        _map = get_map(axis)
        mx, my = _map(lons, lats)
        scatter_map(axis, mx, my, pcs_spatial_phase[:, i], cb_fs=label_size, cmap='RdYlBu_r')
        axis.set_title(f'PC{i + 1} ({percent_var}%) spatial phase')
        
        axis = next(axes)
        _map = get_map(axis)
        scatter_map(axis, mx, my, pcs_spatial_amp[:, i], cb_fs=label_size, cmap='RdYlBu_r')
        axis.set_title(f'PC{i + 1} ({percent_var}%) spatial amplitude')
        
        axis = next(axes)
        axis.plot(pcs_time_phase[:, i])
        axis.set_title(f'PC{i + 1} ({percent_var}%) temporal phase')
        
        axis = next(axes)
        axis.plot(pcs_time_amp[:, i])
        axis.set_title(f'PC{i + 1} ({percent_var}%) temporal amplitude')
        if i % 2 == 1:
            show_or_save(figure, f'complex_pc_{i}_pc_{i + 1}.png')

main()