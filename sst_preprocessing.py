import glob
import argparse
from datetime import datetime

import pandas as pd
import xarray

RAW_DATA_DIR = 'data/sst_raw'
DATA_DIR = 'data/sst'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calculate_loc_filter', action='store_true', default=False)
    parser.add_argument('--start_year', type=int, default=2021)
    parser.add_argument('--end_year', type=int, default=2022)
    parser.add_argument('--end_month', type=int, default=3)
    parser.add_argument('--lat_gran', type=float, default=0.125)
    parser.add_argument('--lon_gran', type=float, default=0.125)
    args = parser.parse_args()

    loc_filter_file = f'{DATA_DIR}/loc_filter.pkl'
    if args.calculate_loc_filter:
        # Read in one file and extract the (lat, lon) co-ordinates
        data_file = f'{RAW_DATA_DIR}/20210101120000-C3S-L4_GHRSST-SSTdepth-OSTIA-GLOB_ICDR2.1-v02.0-fv01.0.nc'
        print(f'Reading data from file {data_file}')
        sample_df: pd.DataFrame = xarray.open_dataset(data_file).to_dataframe()
        sample_df = sample_df['analysed_sst'].dropna().unstack(0).droplevel(0)
        # Filter these based on the specified granularity
        lat_gran = 0.125
        lon_gran = 0.125
        def filter_func(lat_lon):
            lat, lon = lat_lon[0]
            return (lat / lat_gran == round(lat / lat_gran)) and \
                (lon / lon_gran == round(lon / lon_gran))
        loc_filter = pd.DataFrame(sample_df.index).apply(filter_func, axis=1)
        loc_filter.to_pickle(loc_filter_file)
        print(f'Location filter saved to file {loc_filter_file}')
    else:
        loc_filter: pd.DataFrame = pd.read_pickle(loc_filter_file)
        print(f'Location filter read from file {loc_filter_file}')
    loc_filter_indices = loc_filter[loc_filter].index

    # Using the indices based on the filter, now read in all the files
    sst_df = None
    month_dt_span = []
    for y in range(args.start_year, args.end_year + 1):
        for m in range(1, 12 + 1):
            month_dt_span.append(datetime(y, m, 1))
            if y == args.end_year and m >= args.end_month:
                break
    for month_dt in month_dt_span:
        month_df = None
        file_prefix = month_dt.strftime('%Y%m')
        for filename in glob.glob(f'{RAW_DATA_DIR}/{file_prefix}*'):
            print(f'Reading data from file {filename}')
            day_df: pd.DataFrame = xarray.open_dataset(filename).to_dataframe()
            # Unstack takes the most time, so apply the filter just before it
            day_row = day_df['analysed_sst'].dropna()[loc_filter_indices].unstack(0).droplevel(0).T
            if month_df is None:
                month_df = day_row
            else:
                month_df = pd.concat([month_df, day_row])
        month_row = month_df.mean(axis=0).to_frame().T
        month_row.index = [month_dt]
        if sst_df is None:
            sst_df = month_row
        else:
            sst_df = pd.concat([sst_df, month_row])
        print(f'Row calculated for {month_dt.strftime("%b %Y")}')
    # Save the resulting dataframe to pickle
    data_file = (f'{DATA_DIR}/sst_df_{month_dt_span[0].strftime("%Y_%m")}'
        f'_{month_dt_span[-1].strftime("%Y_%m")}.pkl')
    sst_df.to_pickle(data_file)
    print(f'Pre-processed SST dataframe saved to file {data_file}')

if __name__ == '__main__':
    start = datetime.now()
    main()
    print(f'Total time elapsed: {datetime.now() - start}')