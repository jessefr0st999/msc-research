from datetime import datetime, date
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from helpers import get_map, prepare_indexed_df

DATA_DIR = 'data/precipitation'
DATA_FILE = f'{DATA_DIR}/FusedData.csv'
LOCATIONS_FILE = f'{DATA_DIR}/Fused.Locations.csv'
COLOURS = [
    'red',
    'orange',
    'yellow',
    'lightgreen',
    'green',
    'cyan',
    'blue',
    'purple',
    'brown',
    'black',
    'gray',
    'magenta',
]

location_info = [
    {'title': 'adelaide', 'lat': -35.25, 'lon': 138.5},
    {'title': 'alice_springs', 'lat': -23.75, 'lon': 133.5},
    {'title': 'brisbane', 'lat': -27.25, 'lon': 152.5},
    {'title': 'broome', 'lat': -17.75, 'lon': 122.5},
    {'title': 'cairns', 'lat': -16.75, 'lon': 145.5},
    {'title': 'canberra', 'lat': -35.25, 'lon': 149.5},
    {'title': 'darwin', 'lat': -12.75, 'lon': 130.5},
    {'title': 'hobart', 'lat': -42.75, 'lon': 147.5},
    {'title': 'kalgoorlie', 'lat': -30.75, 'lon': 121.5},
    {'title': 'melbourne', 'lat': -37.75, 'lon': 144.5},
    {'title': 'mount_isa', 'lat': -20.75, 'lon': 139.5},
    {'title': 'perth', 'lat': -31.75, 'lon': 116.5},
    {'title': 'sydney', 'lat': -33.75, 'lon': 150.5},
    {'title': 'tennant_creek', 'lat': -19.25, 'lon': 135.5},
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', default=None)
    args = parser.parse_args()

    raw_df = pd.read_csv(DATA_FILE)
    locations_df = pd.read_csv(LOCATIONS_FILE)
    df = prepare_indexed_df(raw_df, locations_df, new_index='date')

    # Plot the average value over each month for several key locations
    # Also plot the key locations on a map of Australia
    figure, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 2.5]})
    axes = axes.flatten()
    lons = locations_df['Lon']
    lats = locations_df['Lat']
    node_colours = ['black'] * len(lons)
    node_sizes = [5] * len(lons)
    for info in location_info:
        location_row = locations_df[(lons == info['lon']) & (lats == info['lat'])].index[0]
        node_colours[location_row] = 'red'
        node_sizes[location_row] = 100
    mx, my = get_map(axes[0])(lons, lats)
    axes[0].scatter(mx, my, c=node_colours, s=node_sizes)

    for j, info in enumerate(location_info):
        loc_df = df[(df['lon'] == info['lon']) & (df['lat'] == info['lat'])]
        for i, m in enumerate(range(1, 12 + 1)):
            month_series = loc_df.loc[loc_df.index.month == m]
            axes[1].scatter(j, np.mean(month_series['prec']), c=COLOURS[i])
    axes[1].legend(
        labels=[date(1900, m, 1).strftime('%B') for m in range(1, 12 + 1)],
        loc='upper left',
    )
    plt.xticks(list(range(0, len(location_info))),
        [info['title'] for info in location_info], rotation='vertical')

    if args.output_folder:
        filename = (f'{args.output_folder}/month_dots.png')
        print(f'Saving graph plot to file {filename}')
        figure.set_size_inches(24, 13.5)
        plt.savefig(filename)
    else:
        plt.show()

if __name__ == '__main__':
    main()