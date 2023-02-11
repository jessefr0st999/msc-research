import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
from rioxarray.raster_array import RasterArray

import pandas as pd
import xarray as xr
import numpy as np

DATA_DIR = 'data/noaa'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', default='data.nc')
    parser.add_argument('--dataset', default='slp')
    args = parser.parse_args()
    
    def save_df(df, name):
        data_file = (f'{DATA_DIR}/{name}'
            f'_{pd.to_datetime(df.index.values[0]).strftime("%Y_%m")}'
            f'_{pd.to_datetime(df.index.values[-1]).strftime("%Y_%m")}.pkl')
        df.to_pickle(data_file)
        print(f'Pre-processed {name} dataframe saved to file {data_file}')

    def map_lon(lon):
        return lon - 360 if lon > 180 else lon

    # NCEP-NCAR reanalysis data
    df: pd.DataFrame = xr.open_dataset(f'{DATA_DIR}/{args.data_file}',
        decode_times=False).to_dataframe()

    def multi_level(df, measure, name):
        # NOTE: this takes quite a while (~10 mins), hence save to pickle
        file_name = f'{DATA_DIR}/{name}_df.pkl'
        if Path(file_name).is_file():
            df = pd.read_pickle(file_name)
        else:
            df = df.reset_index().pivot_table(values=measure, index=['time', 'level'],
                columns=['lat', 'lon'])
            df.to_pickle(file_name)
        for level, _df in df.groupby('level'):
            _df = _df.droplevel('level')
            _df.index = _df.index.map(lambda dt: datetime(1800, 1, 1) + relativedelta(hours=int(dt)))
            _df.columns = _df.columns.map(lambda lat_lon: (lat_lon[0], map_lon(lat_lon[1])))
            save_df(_df, f'{name}_level_{int(level)}')
        return df

    if args.dataset == 'slp':
        df = df.reset_index().pivot_table(values='pressure', index='T', columns=['X', 'Y'])
        df.index = df.index.map(lambda dt: datetime(1960, 1, 1) + relativedelta(months=int(dt)))
        df.columns = df.columns.rename(('lon', 'lat')).swaplevel(0, 1)
    elif args.dataset == 'omega':
        _ = multi_level(df, 'omega', 'omega')
        return
    elif args.dataset == 'humidity':
        _ = multi_level(df, 'rhum', 'humidity')
        return
    elif args.dataset == 'temp':
        # Resample temperature onto the locations for the other variables
        temp_xarray = xr.open_dataset(f'{DATA_DIR}/{args.data_file}',
            decode_times=False)
        temp_xarray = temp_xarray.drop_vars('time_bnds')
        temp_xarray['tmax'] = temp_xarray['tmax'].sel(level=2).drop('level')
        temp_xarray = temp_xarray.drop_dims('level')
        temp_xarray = temp_xarray.to_array()
        pw_xarray = xr.open_dataset(f'{DATA_DIR}/pr_wtr.eatm.mon.mean.nc', decode_times=False)
        temp_array_new = (pw_xarray.drop_vars('time_bnds').to_array()[0, :, :, :] * 0)\
            .rename({'lat': 'y', 'lon': 'x'}).to_numpy()
        for i, dt in enumerate(pw_xarray.coords['time']):
            temp_xarray_slice = temp_xarray.sel(time=dt)
            pw_xarray_slice = pw_xarray.sel(time=dt)
            pw_xarray_slice.rio._crs = 'epsg:4326'
            xr_base = RasterArray(temp_xarray_slice)
            xr_base._crs = 'epsg:4326'
            temp_array_new[i, :, :] = xr_base.reproject_match(pw_xarray_slice).to_numpy()
            if i % 25 == 0:
                print(f'{i} / {len(pw_xarray.coords["time"])}')
        df = pw_xarray.to_dataframe().reset_index()\
            .pivot_table(values='pr_wtr', index='time', columns=['lat', 'lon'])
        df.index = df.index.map(lambda dt: datetime(1800, 1, 1) + relativedelta(hours=int(dt)))
        df = pd.DataFrame(np.flip(temp_array_new, axis=1).reshape(529, 10512),
            index=df.index, columns=df.columns)
        df -= 273.15
    elif args.dataset == 'pw':
        df = df.reset_index().pivot_table(values='pr_wtr', index='time',
            columns=['lat', 'lon'])
        df.index = df.index.map(lambda dt: datetime(1800, 1, 1) + relativedelta(hours=int(dt)))
    df.columns = df.columns.map(lambda lat_lon: (lat_lon[0], map_lon(lat_lon[1])))
    save_df(df, args.dataset)

if __name__ == '__main__':
    start = datetime.now()
    main()
    print(f'Total time elapsed: {datetime.now() - start}')