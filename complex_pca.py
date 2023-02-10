import argparse
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt

from helpers import configure_plots, get_map, scatter_map, prepare_df

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
    parser.add_argument('--data_dir', default='data/precipitation')
    parser.add_argument('--data_file', default='FusedData.csv')
    parser.add_argument('--pcs_to_plot', type=int, default=4)
    args = parser.parse_args()
    label_size, font_size, show_or_save = configure_plots(args)

    dataset = 'prec' if args.data_file == 'FusedData.csv' else args.data_file.split('_')[0]
    df, lats, lons = prepare_df(args.data_dir, args.data_file, dataset)
    plot_aus = True if dataset == 'prec' else False

    df_complex = df.apply(hilbert, axis=0)
    vars_file_name = f'data/cpca/{dataset}_cpca_vars.pkl'
    pcs_file_name = f'data/cpca/{dataset}_cpca_pcs.pkl'
    if Path(vars_file_name).is_file() and Path(pcs_file_name).is_file():
        with open(vars_file_name, 'rb') as f:
            vars = pickle.load(f)
        with open(pcs_file_name, 'rb') as f:
            pcs = pickle.load(f)
        print(f'Complex PCA data read from Pickle files {vars_file_name} and {pcs_file_name}')
    else:
        vars, pcs = complex_pca(df_complex)
        with open(vars_file_name, 'wb') as f:
            pickle.dump(vars, f)
        with open(pcs_file_name, 'wb') as f:
            pickle.dump(pcs, f)
        print(f'Complex PCA data saved to Pickle files {vars_file_name} and {pcs_file_name}')
   
    prop_vars = vars / vars.sum()
    figure, axes = plt.subplots(1, 2, layout='compressed')
    axes[0].plot(list(range(2, 11)), prop_vars[1 : 10])
    axes[0].set_title('PC2 to PC10 explained variance proportion')
    axes[1].plot(list(range(1, len(prop_vars) + 1)), np.log10(prop_vars))
    axes[1].set_title('Log explained variance proportion')
    show_or_save(figure, f'{dataset}_complex_pca_explained_var.png')

    pcs_spatial_phase = np.angle(pcs)
    pcs_spatial_amp = np.abs(pcs)
    pcs_time = np.array(df_complex @ pcs)
    pcs_time_phase = np.angle(pcs_time)
    pcs_time_amp = np.abs(pcs_time)
    
    lats, lons = zip(*df.columns)
    for i in range(args.pcs_to_plot):
        if i % 2 == 0:
            figure, axes = plt.subplots(2, 4, layout='compressed')
            axes = iter(axes.flatten())
        percent_var = np.round(100 * prop_vars[i], 2)
        axis = next(axes)
        _map = get_map(axis, plot_aus)
        mx, my = _map(lons, lats)
        scatter_map(axis, mx, my, pcs_spatial_phase[:, i], cb_fs=label_size, cmap='RdYlBu_r')
        axis.set_title(f'PC{i + 1} ({percent_var}%) spatial phase')
        
        axis = next(axes)
        _map = get_map(axis, plot_aus)
        scatter_map(axis, mx, my, pcs_spatial_amp[:, i], cb_fs=label_size, cmap='RdYlBu_r')
        axis.set_title(f'PC{i + 1} ({percent_var}%) spatial amplitude')
        
        axis = next(axes)
        axis.plot(pcs_time_phase[:, i])
        axis.set_title(f'PC{i + 1} ({percent_var}%) temporal phase')
        
        axis = next(axes)
        axis.plot(pcs_time_amp[:, i])
        axis.set_title(f'PC{i + 1} ({percent_var}%) temporal amplitude')
        if i % 2 == 1:
            show_or_save(figure, f'{dataset}_complex_pc_{i}_pc_{i + 1}.png')

if __name__ == '__main__':
    start = datetime.now()
    main()
    print(f'Total time elapsed: {datetime.now() - start}')