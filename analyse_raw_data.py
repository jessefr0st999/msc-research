import argparse
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from helpers import get_map

DATA_DIR = 'data/precipitation'
IMAGES_DIR = 'images'
DATA_FILE = f'{DATA_DIR}/FusedData.csv'
LOCATIONS_FILE = f'{DATA_DIR}/Fused.Locations.csv'

iqr_func = lambda series: np.quantile(series, 0.75) - \
    np.quantile(series, 0.25)
size_func = lambda series: [500 * n**2 for n in series]
cmap = 'RdYlGn_r'

def save_or_show(figure, filename, save=False):
    plt.tight_layout()
    if save:
        figure.set_size_inches(32, 18)
        plt.savefig(f'{IMAGES_DIR}/{filename}', bbox_inches='tight')
        print(f'Plot saved to file {IMAGES_DIR}/{filename}!')
    else:
        plt.show()

def prepare_indexed_df(input_df, new_index='date'):
    input_df.columns = pd.to_datetime(input_df.columns, format='D%Y.%m')
    df_locations = pd.read_csv(LOCATIONS_FILE)
    df = pd.concat([df_locations, input_df], axis=1)
    df = df.set_index(['Lat', 'Lon'])
    df = df.stack().reset_index()
    df = df.rename(columns={'level_2': 'date', 0: 'prec', 'Lat': 'lat', 'Lon': 'lon'})
    df = df.set_index(new_index)
    return df

def scatterplot(axis, mx, my, series):
    axis.scatter(mx, my, c=series, cmap=cmap,
        s=size_func((series - np.min(series)) / np.max(series)))
    norm = mpl.colors.Normalize(vmin=np.min(series), vmax=np.max(series))
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_fig', action='store_true', default=False)
    args = parser.parse_args()

    raw_df = pd.read_csv(DATA_FILE)
    df = prepare_indexed_df(raw_df, ['date', 'lat', 'lon'])

    date_group = df.groupby('date').prec
    date_mean = date_group.mean()
    date_std = date_group.std()
    date_skew = date_group.skew()
    date_kurtosis = date_group.apply(pd.Series.kurt)
    date_min = date_group.min()
    date_max = date_group.max()
    date_median = date_group.median()
    date_iqr = date_group.apply(iqr_func)
    dates = date_mean.index.values

    figure, axes = plt.subplots(4, 1)
    axes = axes.flatten()
    axes[0].plot(dates, date_mean, '-')
    axes[0].set_title('date_mean')
    axes[1].plot(dates, date_std, '-')
    axes[1].set_title('date_std')
    axes[2].plot(dates, date_skew, '-')
    axes[2].set_title('date_skew')
    axes[3].plot(dates, date_kurtosis, '-')
    axes[3].set_title('date_kurtosis')
    save_or_show(figure, 'stats_date_1.png', args.save_fig)

    figure, axes = plt.subplots(4, 1)
    axes = axes.flatten()
    axes[0].plot(dates, date_min, '-')
    axes[0].set_title('date_min')
    axes[1].plot(dates, date_max, '-')
    axes[1].set_title('date_max')
    axes[2].plot(dates, date_median, '-')
    axes[2].set_title('date_median')
    axes[3].plot(dates, date_iqr, '-')
    axes[3].set_title('date_iqr')
    save_or_show(figure, 'stats_date_2.png', args.save_fig)

    loc_group = df.groupby(['lat', 'lon']).prec
    loc_mean = loc_group.mean()
    loc_std = loc_group.std()
    loc_skew = loc_group.skew()
    loc_kurtosis = loc_group.apply(pd.Series.kurt)
    loc_min = loc_group.min()
    loc_max = loc_group.max()
    loc_median = loc_group.median()
    loc_iqr = loc_group.apply(iqr_func)
    lats, lons = zip(*loc_mean.index.values)

    figure, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    mean_map = get_map(axes[0])
    std_map = get_map(axes[1])
    skew_map = get_map(axes[2])
    kurtosis_map = get_map(axes[3])
    mx, my = mean_map(lons, lats)
    scatterplot(axes[0], mx, my, loc_mean)
    scatterplot(axes[1], mx, my, loc_std)
    scatterplot(axes[2], mx, my, loc_skew)
    scatterplot(axes[3], mx, my, loc_kurtosis)
    axes[0].set_title('loc_mean')
    axes[1].set_title('loc_std')
    axes[2].set_title('loc_skew')
    axes[3].set_title('loc_kurtosis')
    save_or_show(figure, 'stats_location_1.png', args.save_fig)

    figure, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    min_map = get_map(axes[0])
    max_map = get_map(axes[1])
    medianmap = get_map(axes[2])
    iqr_map = get_map(axes[3])
    scatterplot(axes[0], mx, my, loc_min)
    scatterplot(axes[1], mx, my, loc_max)
    scatterplot(axes[2], mx, my, loc_median)
    scatterplot(axes[3], mx, my, loc_iqr)
    axes[0].set_title('loc_min')
    axes[1].set_title('loc_max')
    axes[2].set_title('loc_median')
    axes[3].set_title('loc_iqr')
    save_or_show(figure, 'stats_location_2.png', args.save_fig)

main()