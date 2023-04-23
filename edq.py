import argparse
from datetime import datetime
import pickle
from math import ceil

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from helpers import configure_plots, get_map, scatter_map

# Note that intra-annual analysis may be affected for 2000 and 2022, as these
# years lack data for all months
YEARS = list(range(2000, 2022 + 1))
DATA_DIR = 'data/precipitation'
PREC_DATA_FILE = f'{DATA_DIR}/FusedData.csv'
LOCATIONS_FILE = f'{DATA_DIR}/Fused.Locations.csv'

def edq(series_df, series_name, level=0):
    print('\n' + f'Calculating EDQ series "{series_name}" for level {round(level, 2)}')
    i = 0
    def edq_score(q):
        score = 0
        for t, q_t in enumerate(q):
            z_t = series_df.iloc[t, :]
            z_ge_q = (z_t >= q_t).astype(int)
            z_le_q = (z_t <= q_t).astype(int)
            weighted_diffs = np.abs(z_t - q_t) * (level * z_ge_q + (1 - level) * z_le_q)
            score += weighted_diffs.sum()
        nonlocal i
        if i % 100 == 0:
            print(f'{i} / {series_df.shape[1]}')
        i += 1
        return score
    scores = series_df.apply(edq_score, axis=0)
    return series_df.loc[:, scores.idxmin()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', default=None)
    parser.add_argument('--calculate_edq', action='store_true', default=False)
    parser.add_argument('--num_levels', type=int, default=11)
    args = parser.parse_args()
    label_size, font_size, show_or_save = configure_plots(args)

    edq_levels = np.linspace(0, 1, args.num_levels)
    raw_prec_df = pd.read_csv(PREC_DATA_FILE)
    raw_prec_df.columns = pd.to_datetime(raw_prec_df.columns, format='D%Y.%m')
    locations_df = pd.read_csv(LOCATIONS_FILE)
    prec_df = pd.concat([locations_df, raw_prec_df], axis=1)
    prec_df = prec_df.set_index(['Lat', 'Lon'])
    prec_df = prec_df.T

    lats = locations_df['Lat']
    lons = locations_df['Lon']

    prec_edq_series = [
        'prec',
        'prec_summer',
        'prec_autumn',
        'prec_winter',
        'prec_spring',
    ]
    prec_edq_files = [f'{DATA_DIR}/edq_{series_name}.pkl' for series_name in prec_edq_series]
    prec_edq_dfs = []
    if args.calculate_edq:
        # Calculate and save the EDQ dimension-reduced series
        prec_dfs = [
            prec_df,
            prec_df.loc[[_dt.month in [12, 1, 2] for _dt in prec_df.index], :],
            prec_df.loc[[_dt.month in [3, 4, 5] for _dt in prec_df.index], :],
            prec_df.loc[[_dt.month in [6, 7, 8] for _dt in prec_df.index], :],
            prec_df.loc[[_dt.month in [9, 10, 11] for _dt in prec_df.index], :],
        ]
        for i, df in enumerate(prec_dfs):
            prec_edq = [edq(df, prec_edq_series[i], p) for p in edq_levels]
            filename = prec_edq_files[i]
            with open(filename, 'wb') as f:
                pickle.dump(prec_edq, f)
                print(f'EDQ data saved to file {filename}!')
            prec_edq_dfs.append(prec_edq)
    else:
        for filename in prec_edq_files:
            with open(filename, 'rb') as f:
                prec_edq_dfs.append(pickle.load(f))
                print(f'EDQ data read from file {filename}')

    for i, df in enumerate(prec_edq_dfs):
        series_name = prec_edq_series[i]
        # First plot the map showing the locations of the representative series
        locations = [q.name for q in df]
        labels = [f'{loc}, p = {round(edq_levels[i], 2)}' \
            for i, loc in enumerate(locations)]
        lats, lons = zip(*locations)
        figure, axis = plt.subplots(1)
        _map = get_map(axis)
        mx, my = _map(lons, lats)
        axis.scatter(mx, my, c='red', s=50)
        for i, label in enumerate(labels):
            axis.annotate(label, (mx[i], my[i]))
        axis.set_title(f'{series_name} EDQ series')
        show_or_save(figure, f'edq_{series_name}_loc_map.png')

        # Then plot each series itself in another figure
        _prec_edq = [prec_df.loc[:, loc] for loc in locations]
        figure, axes = plt.subplots(ceil((len(locations)) / 3), 3, layout='compressed')
        axes = iter(axes.flatten())
        for i, prec_q in enumerate(_prec_edq):
            axis = next(axes)
            d1_av = prec_q.iloc[:11].mean()
            d2_av = prec_q.iloc[11:].mean()
            print()
            print(labels[i])
            print(f'{series_name} decade 1 average: {round(d1_av, 4)},'
                f' decade 2 average: {round(d2_av, 4)},'
                f' change: {round(d2_av - d1_av, 4)}')
            axis.plot(prec_q, c='green')
            axis.set_title(f'{series_name} {labels[i]}')
        show_or_save(figure, f'edq_{series_name}_loc_series.png')

if __name__ == '__main__':
    main()