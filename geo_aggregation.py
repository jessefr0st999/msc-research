import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic

from helpers import get_map, scatter_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calculate_loc_agg_df', action='store_true', default=False)
    parser.add_argument('--dataset', default='prec')
    parser.add_argument('--lat_period', type=int, default=5)
    parser.add_argument('--lon_period', type=int, default=5)
    # Default offset for precipitation to ensure Tasmania gets a point
    parser.add_argument('--lat_offset', type=int, default=0)
    parser.add_argument('--lon_offset', type=int, default=1)
    args = parser.parse_args()

    if args.dataset == 'sst':
        df = pd.read_pickle('data/sst/sst_df_2021_01_2022_03.pkl')
        output_folder = 'data/sst'
    else: # prec
        raw_df = pd.read_csv('data/precipitation/FusedData.csv')
        raw_df.columns = pd.to_datetime(raw_df.columns, format='D%Y.%m')
        locations_df = pd.read_csv('data/precipitation/Fused.Locations.csv')
        df = pd.concat([locations_df, raw_df], axis=1)
        df = df.set_index(['Lat', 'Lon']).T
        output_folder = 'data/precipitation'
    
    file_name = f'{output_folder}/{args.dataset}_locations_df_agg.pkl'
    if Path(file_name).is_file():
        print(f'locations_df_agg read from pickle file {file_name}')
        locations_df_agg: pd.DataFrame = pd.read_pickle(file_name)
    else:
        # Determine the locations to be kept in the aggregated dataframe
        locations_df = pd.DataFrame(list(df.columns.values), columns=['lat', 'lon'])
        lats_unique = np.unique(locations_df['lat'])
        lons_unique = np.unique(locations_df['lon'])
        lats_agg_unique = []
        lons_agg_unique = []
        for i, lat in enumerate(lats_unique):
            if (i + args.lat_offset) % args.lat_period == 0:
                lats_agg_unique.append(lat)
        for i, lon in enumerate(lons_unique):
            if (i + args.lon_offset) % args.lon_period == 0:
                lons_agg_unique.append(lon)
        locations_agg_ind = locations_df.apply(lambda row:
            (row['lat'] in lats_agg_unique) & \
                (row['lon'] in lons_agg_unique), axis=1)
        locations_df_agg = locations_df[locations_agg_ind]
        lats_agg = locations_df_agg['lat']
        lons_agg = locations_df_agg['lon']
        locations_df_agg.to_pickle(file_name)
        print(f'locations_df_agg saved to {file_name}')

    # The SST dataset is very large; just use the subset of points themselves
    # rather than doing a mean aggregation
    if args.dataset == 'sst':
        locations_df_agg_tuples = [tuple(v) for v in locations_df_agg.values]
        file_name = f'{output_folder}/{args.dataset}_df_agg.pkl'
        df_agg = df[locations_df_agg_tuples]
        df_agg.to_pickle(file_name)
        print(f'df_agg saved to {file_name}')
        return
    
    # Assign all other locations to the closest kept location and take
    # the mean over each kept location
    locations_df_agg_tuples = [tuple(v) for v in locations_df_agg.values]
    agg_map = {t: df[[t]] for t in locations_df_agg_tuples}
    for i, loc in enumerate(df.columns):
        if i % 100 == 0:
            print(f'{i} / {df.shape[1]}')
        if loc in locations_df_agg_tuples:
            continue
        distances = [geodesic(loc, agg_loc) for agg_loc in locations_df_agg_tuples]
        closest_agg_loc = locations_df_agg_tuples[np.argmin(distances)]
        agg_map[closest_agg_loc][loc] = df[loc]
    # Print the number of locations assigned to each kept location
    for t in agg_map:
        print(t, agg_map[t].shape[1])
    df_agg = df * 0
    for t in agg_map:
        df_agg[t] = agg_map[t].mean(axis=1)
    df_agg = df_agg.loc[:, (df_agg != 0).any(axis=0)]
    file_name = f'{output_folder}/{args.dataset}_df_agg.pkl'
    df_agg.to_pickle(file_name)
    print(f'df_agg saved to {file_name}')

    figure, axes = plt.subplots(2, 2)
    axes = iter(axes.flatten())
    
    # Plot the kept (red) vs discarded (blue) lat/lon values
    axis = next(axes)
    axis.plot(lons_unique, 'o', c='blue')
    lons_agg_indices = []
    for i, lon in enumerate(lons_unique):
        if lon in lons_agg_unique:
            lons_agg_indices.append(i)
    axis.plot(lons_agg_indices, lons_agg_unique, 'o', c='red')

    axis = next(axes)
    axis.plot(np.sort(lats_unique), 'o', c='blue')
    lats_agg_indices = []
    for i, lat in enumerate(np.sort(lats_unique)):
        if lat in lats_agg_unique:
            lats_agg_indices.append(i)
    axis.plot(lats_agg_indices, lats_agg_unique, 'o', c='red')

    # Plot all vs kept locations
    axis = next(axes)
    _map = get_map(axis)
    mx, my = _map(locations_df['lon'], locations_df['lat'])
    scatter_map(axis, mx, my, [1 for _ in locations_df['lon']], show_cb=False,
        cmap='binary_r', size_func=lambda x: 10)

    axis = next(axes)
    _map = get_map(axis)
    mx, my = _map(lons_agg, lats_agg)
    scatter_map(axis, mx, my, [1 for _ in lons_agg], show_cb=False, cmap='binary_r',
        size_func=lambda x: 10)
        
    plt.show()

if __name__ == '__main__':
    main()