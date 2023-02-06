import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
import xarray as xr

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

    # NCEP-NCAR reanalysis data
    df: pd.DataFrame = xr.open_dataset(f'{DATA_DIR}/{args.data_file}',
        decode_times=False).to_dataframe()

    if args.dataset == 'slp':
        df = df.reset_index().pivot_table(values='pressure', index='T', columns=['X', 'Y'])
        df.index = df.index.map(lambda dt: datetime(1960, 1, 1) + relativedelta(months=int(dt)))
        df.columns = df.columns.rename(('lon', 'lat')).swaplevel(0, 1)
        df.columns = df.columns.map(lambda lat_lon: (lat_lon[0], lat_lon[1] - 180))
        save_df(df, 'slp')
    elif args.dataset == 'omega':
        # NOTE: this takes quite a while (~10 mins)
        df = df.reset_index().pivot_table(values='omega', index=['time', 'level'],
            columns=['lat', 'lon'])
        for level, _df in df.groupby('level'):
            _df = _df.droplevel('level')
            _df.index = _df.index.map(lambda dt: datetime(1800, 1, 1) + relativedelta(hours=int(dt)))
            _df.columns = _df.columns.map(lambda lat_lon: (lat_lon[0], lat_lon[1] - 180))
            save_df(_df, f'omega_level_{int(level)}')
    elif args.dataset == 'temp_max':
        df = df.reset_index().pivot_table(values='tmax', index=['time', 'level'],
            columns=['lat', 'lon'])
        df = df.droplevel('level')
        df.index = df.index.map(lambda dt: datetime(1800, 1, 1) + relativedelta(hours=int(dt)))
        df.columns = df.columns.map(lambda lat_lon: (lat_lon[0], lat_lon[1] - 180))
        df -= 273.15
        save_df(df, 'temp_max')

if __name__ == '__main__':
    start = datetime.now()
    main()
    print(f'Total time elapsed: {datetime.now() - start}')