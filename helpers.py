import pickle

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def lat_lon_bounds(lat, lon, region='default'):
    if region == 'aus':
        # Indian and south Pacific oceans
        global_lon_bounds = lon < -90 or lon > 30
        global_lat_bounds = lat >= -60 and lat <= 25
        red_sea_bounds = lat > 13 and lat < 30 and lon > 32 and lon < 44
        gulf_of_mexico_bounds = lat > 18 and lat < 30 and lon > -100 and lon < -81
        return global_lon_bounds and global_lat_bounds and \
            not (red_sea_bounds or gulf_of_mexico_bounds)
    if region == 'all':
        return True
    return lat >= -60 and lat <= 60

def prepare_df(data_dir, data_file, dataset):
    # TODO: pre-process precipitation as per other datasets
    if dataset == 'prec':
        raw_df = pd.read_csv(f'{data_dir}/{data_file}')
        raw_df.columns = pd.to_datetime(raw_df.columns, format='D%Y.%m')
        locations_df = pd.read_csv(f'{data_dir}/Fused.Locations.csv')
        df = pd.concat([locations_df, raw_df], axis=1).set_index(['Lat', 'Lon']).T
        lats = locations_df['Lat']
        lons = locations_df['Lon']
    else:
        df = pd.read_pickle(f'{data_dir}/{data_file}')
        df.index = pd.to_datetime(df.index, format='D%Y.%m')
        columns_to_keep = set()
        for lat, lon in df.columns:
            # if lat_lon_bounds(lat, lon, 'aus'):
            if lat_lon_bounds(lat, lon):
                columns_to_keep.add((lat, lon))
        # with open(f'ocean_locations_{dataset}.pkl', 'rb') as f:
        #     ocean_locations = set(pickle.load(f))
        # columns_to_keep &= ocean_locations
        df = df[list(columns_to_keep)]
        lats, lons = zip(*df.columns)
    return df, lats, lons

# TODO: replace this with above
def prepare_indexed_df(raw_df, locations_df, month=None, new_index='date'):
    raw_df.columns = pd.to_datetime(raw_df.columns, format='D%Y.%m')
    df = pd.concat([locations_df, raw_df], axis=1)
    df = df.set_index(['Lat', 'Lon'])
    if month:
        df = df.loc[:, [c.month == month for c in df.columns]]
    df = df.stack().reset_index()
    df = df.rename(columns={'level_2': 'date', 0: 'prec', 'Lat': 'lat', 'Lon': 'lon'})
    df = df.set_index(new_index)
    return df

def link_str_to_adjacency(link_str_df: pd.DataFrame, edge_density=None,
        threshold=None, lag_bool_df=None):
    if edge_density:
        threshold = np.quantile(link_str_df, 1 - edge_density)
        print(f'Fixed edge density {edge_density} gives threshold {threshold}')
    _link_str_df = link_str_df.copy() if lag_bool_df is None \
        else link_str_df * lag_bool_df
    if not _link_str_df.index.equals(link_str_df.columns):
        n, m = link_str_df.shape
        # Given link_str_df is n x m, rearrange it into a
        # (n + m) x (n + m) matrix with blocks as below:
        # [[ 0_n   LS  ]
        #  [ LS^T  O_m ]]
        link_str_array_square = np.concatenate((
            np.concatenate((np.zeros((n, n)), _link_str_df), axis=1),
            np.concatenate((_link_str_df.T, np.zeros((m, m))), axis=1),
        ), axis=0)
        _index = pd.MultiIndex.from_tuples([*_link_str_df.index.values,
            *_link_str_df.columns.values], names=['lat', 'lon'])
        _link_str_df = pd.DataFrame(link_str_array_square, index=_index, columns=_index)
    adjacency = _link_str_df * 0
    adjacency[_link_str_df >= threshold] = 1
    if not edge_density and lag_bool_df is None:
        _edge_density = np.sum(np.sum(adjacency)) / adjacency.size
        print(f'Fixed threshold {threshold} gives edge density {_edge_density}')
    return adjacency

def read_link_str_df(filename: str):
    if filename.endswith('pkl'):
        link_str_df = pd.read_pickle(filename)
    else: # Assume CSV
        link_str_df = pd.read_csv(filename, index_col=[0, 1], header=[0, 1])
    link_str_df.columns = [link_str_df.columns.get_level_values(i).astype(float) \
        for i in range(len(link_str_df.columns.levels))]
    return link_str_df
            
def configure_plots(args):
    label_size = 20 if args.output_folder else 10
    font_size = 20 if args.output_folder else 10
    mpl.rcParams['xtick.labelsize'] = label_size
    mpl.rcParams['ytick.labelsize'] = label_size
    mpl.rcParams.update({'font.size': font_size})
    def show_or_save(figure, filename):
        if args.output_folder:
            figure.set_size_inches(32, 18)
            plt.savefig(f'{args.output_folder}/{filename}', bbox_inches='tight')
            print(f'Plot saved to file {args.output_folder}/{filename}!')
        else:
            plt.show()
    return label_size, font_size, show_or_save


def get_map(axis=None, region='aus'):
    # Default: world
    kwargs = dict(
        lat_ts=0,
        resolution='l',
        suppress_ticks=True,
        ax=axis,
    )
    if region == 'aus':
        kwargs |= dict(
            projection='merc',
            llcrnrlon=110,
            llcrnrlat=-45,
            urcrnrlon=155,
            urcrnrlat=-10,
        )
    elif region == 'aus_oceans':
        kwargs |= dict(
            projection='merc',
            llcrnrlon=25,
            llcrnrlat=-65,
            urcrnrlon=180,
            urcrnrlat=20,
        )
    _map = Basemap(**kwargs)
    _map.drawcountries(linewidth=1)
    _map.drawstates(linewidth=0.2)
    _map.drawcoastlines(linewidth=1)
    return _map

def scatter_map(axis, mx, my, series, cb_min=None, cb_max=None, cmap='inferno_r',
        size_func=None, show_cb=True, cb_fs=10):
    series = np.array(series)
    if cb_min is None:
        cb_min = np.min(series)
    if cb_max is None:
        cb_max = np.max(series)
    norm = mpl.colors.Normalize(vmin=cb_min, vmax=cb_max)
    if size_func is None:
        size_func = lambda series: 50
    axis.scatter(mx, my, c=series, norm=norm, cmap=cmap, s=size_func(series))
    if show_cb:
        plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis)\
            .ax.tick_params(labelsize=cb_fs)

def file_region_type(file_name: str):
    noaa_file_tags = ['slp', 'temp', 'humidity', 'omega', 'pw']
    for tag in noaa_file_tags:
        if tag in file_name:
            if 'prec' in file_name:
                return 'aus_oceans'
            else:
                return 'world'
    return 'aus'