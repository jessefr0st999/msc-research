import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path

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

    # NCAR data (15N to 90N only)
    # data_file = f'{DATA_DIR}/ds010.1.20000100.20221231'
    # df = pd.read_fwf(data_file, index_col=[0])
    # df = df.drop(labels='90N', axis=1)
    # df.index = pd.to_datetime(df.index, format='%Y/%m')
    # def loc_str_to_lat_lon(loc_str):
    #     loc_list = loc_str.split(',')
    #     lat = int(loc_list[1][: -1])
    #     lon = int(loc_list[0][: -1]) - 180
    #     return (lat, lon)
    # df.columns = pd.Index([loc_str_to_lat_lon(c) for c in df.columns])

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
            _df.columns = _df.columns.map(lambda lat_lon: (lat_lon[0], lat_lon[1] - 180))
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
        df = df.reset_index().pivot_table(values='tmax', index=['time', 'level'],
            columns=['lat', 'lon'])
        df = df.droplevel('level')
        df.index = df.index.map(lambda dt: datetime(1800, 1, 1) + relativedelta(hours=int(dt)))
        df -= 273.15
    elif args.dataset == 'pr_water':
        df = df.reset_index().pivot_table(values='pr_wtr', index='time',
            columns=['lat', 'lon'])
        df.index = df.index.map(lambda dt: datetime(1800, 1, 1) + relativedelta(hours=int(dt)))
    df.columns = df.columns.map(lambda lat_lon: (lat_lon[0], lat_lon[1] - 180))
    save_df(df, args.dataset)

if __name__ == '__main__':
    start = datetime.now()
    main()
    print(f'Total time elapsed: {datetime.now() - start}')