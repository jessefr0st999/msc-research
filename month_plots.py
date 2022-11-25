from datetime import date
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from helpers import get_map, prepare_indexed_df

DATA_DIR = 'data/precipitation'
DATA_FILE = f'{DATA_DIR}/FusedData.csv'
LOCATIONS_FILE = f'{DATA_DIR}/Fused.Locations.csv'
# Blue, orange, green, red, purple, brown, pink, light yellow, dark yellow, teal, black, grey
COLOURS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#ffff00', '#bcbd22', '#17becf', 'black', 'gray']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--title', default='melbourne')
    parser.add_argument('--lat', type=float, default=-37.75)
    parser.add_argument('--lon', type=float, default=144.5)
    parser.add_argument('--output_folder', default=None)
    parser.add_argument('--all_locations', action='store_true', default=False)
    args = parser.parse_args()

    raw_df = pd.read_csv(DATA_FILE)
    locations_df = pd.read_csv(LOCATIONS_FILE)

    if args.all_locations:
        # Plot a time series for each month, averaged over all locations
        df = prepare_indexed_df(raw_df, locations_df, new_index=['date', 'lat', 'lon'])
        series = df.groupby('date').prec.mean()
        figure, plot_axis = plt.subplots(1, 1)
    else:
        # Plot a time series for each month at the given (lat, lon) location
        # Also plot the given location on a map of Australia
        df = prepare_indexed_df(raw_df, locations_df, new_index='date')
        series = df[(df['lon'] == args.lon) & (df['lat'] == args.lat)]
        figure, (map_axis, plot_axis) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 2.5]})
        lons = locations_df['Lon']
        lats = locations_df['Lat']
        node_colours = ['black'] * len(lons)
        node_sizes = [5] * len(lons)
        location_row = locations_df[(lons == args.lon) & (lats == args.lat)].index[0]
        node_colours[location_row] = 'red'
        node_sizes[location_row] = 100
        mx, my = get_map(map_axis)(lons, lats)
        map_axis.scatter(mx, my, c=node_colours, s=node_sizes)
        map_axis.set_title(f'{args.lat}, {args.lon}', fontsize=20)

    for i, m in enumerate(range(1, 12 + 1)):
        month_series = series.loc[series.index.month == m]
        if not args.all_locations:
            month_series = month_series['prec']
        years = [dt.year for dt in month_series.index]
        plot_axis.plot(years, month_series, c=COLOURS[i], linewidth=2.5)
    plot_axis.legend(
        labels=[date(1900, m, 1).strftime('%B') for m in range(1, 12 + 1)],
        loc='upper left',
        fontsize=20,
    )

    if args.output_folder:
        filename = (f'{args.output_folder}/month_plots_{args.title}_lat_{str(args.lat).replace(".", "p")}'
            f'_lon_{str(args.lon).replace(".", "p")}.png')
        print(f'Saving graph plot to file {filename}')
        figure.set_size_inches(32, 18)
        plt.savefig(filename)
    else:
        plt.show()

if __name__ == '__main__':
    main()