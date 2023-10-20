import os
import ast
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from helpers import get_map, scatter_map, prepare_df, configure_plots

PATH = 'data_unfused/bom_daily'
START_DATE = datetime(2000, 1, 1)

def main():
    loc_list = []
    info_list = []
    for i, path in enumerate(os.scandir(PATH)):
        if i % 100 == 0:
            print(i)
        if i == 10000:
            break
        if path.is_file():
            prec_df = pd.read_csv(f'{PATH}/{path.name}')
            loc = (prec_df.iloc[0]['Lon'], -prec_df.iloc[0]['Lat'])
            # if loc[1] < -13.5:
            #     continue
            # if loc[0] < 145:
            #     continue
            prec_df = prec_df.dropna(subset=['Rain'])
            prec_df = prec_df[prec_df['Year'] >= START_DATE.year]
            if prec_df.empty:
                continue
            start = datetime.strptime(prec_df.iloc[0]['Date'], '%Y-%m-%d')
            end = datetime.strptime(prec_df.iloc[-1]['Date'], '%Y-%m-%d')
            if end.year < 2019 or prec_df.shape[0] < 6000:
                continue
            info_list.append({
                'start': (start - START_DATE).days,
                'end': (end - START_DATE).days,
                'mean': prec_df['Rain'].mean(),
                'median': prec_df['Rain'].median(),
                'count': prec_df.shape[0],
                'count_zero': prec_df.shape[0] - np.count_nonzero(prec_df['Rain']),
                'filename': path.name,
            })
            loc_list.append(loc)
    info_df = pd.DataFrame(info_list, index=loc_list)
    info_df.to_csv('bom_info.csv')

    info_df = pd.read_csv('bom_info.csv', index_col=0, converters={0: ast.literal_eval})
    # print(info_df)
    # print(info_df.shape[0], info_df['count'].mean(), info_df['count'].median())
    # exit()
    lons, lats = list(zip(*list(info_df.index.values)))
    figure, axes = plt.subplots(2, 2, layout='compressed')
    axes = iter(axes.flatten())
    for column in info_df.columns:
        if column == 'filename':
            continue
        axis = next(axes)
        _map = get_map(axis)
        mx, my = _map(lons, lats)
        axis.set_title(column)
        scatter_map(axis, mx, my, info_df[column], cb_min=info_df[column].min(),
            cb_max=info_df[column].max(), size_func=lambda series: 20, cmap='inferno_r')
    plt.show()
    
if __name__ == '__main__':
    main()