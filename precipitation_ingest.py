from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

DATABASE_URI = 'postgresql+psycopg2://postgres:secret@localhost:5432/msc_research'

DATA_FILE = 'data/precipitation/FusedData.csv'
LOCATIONS_FILE = 'data/precipitation/Fused.Locations.csv'
DATAFRAME_FILE = 'data/precipitation/dataframe_drop_75.pkl'

YEARS = range(2000, 2022)
MONTHS = range(1, 13)

DROP_RATIO = 0
LOOKBACK_MONTHS = 12
NORM_CORR_THRESHOLD = 0.9

def build_network(df: pd.DataFrame):
    norm_corr = pd.DataFrame(index=[df['lat'], df['lon']], columns=[df['lat'], df['lon']])
    df = df.reset_index().set_index(['lat', 'lon'])

    # TODO: optimise this
    for i, idx1 in enumerate(df.index):
        if i % 25 == 0:
            print(f'{i} / {len(df.index)} points correlated')
        for j, idx2 in enumerate(df.index[i + 1:]):
            seq_1 = df.loc[idx1, 'prec_seq']
            seq_2 = df.loc[idx2, 'prec_seq']
            seq_1 = (seq_1 - np.mean(seq_1)) / np.std(seq_1)
            seq_2 = (seq_2 - np.mean(seq_2)) / np.std(seq_2)
            seq_1 /= len(seq_1)
            norm_corr.loc[idx1, idx2] = np.correlate(seq_1, seq_2)[0]

    # norm_corr is upper triangular
    norm_corr = norm_corr.add(norm_corr.T, fill_value=0).fillna(0)
    adjacency = norm_corr
    adjacency[adjacency >= NORM_CORR_THRESHOLD] = 1
    adjacency[adjacency < NORM_CORR_THRESHOLD] = 0
    graph = nx.from_numpy_matrix(adjacency.values)
    graph = nx.relabel_nodes(graph, dict(enumerate(adjacency.columns)))

    graph_map = Basemap(
        projection='merc',
        llcrnrlon=110,
        llcrnrlat=-45,
        urcrnrlon=155,
        urcrnrlat=-10,
        lat_ts=0,
        resolution='l',
        suppress_ticks=True,
    )
    df = df.reset_index()
    mx, my = graph_map(df['lon'], df['lat'])
    pos = {}
    for i, elem in enumerate(adjacency.index):
        pos[elem] = (mx[i], my[i])

    nx.draw_networkx_nodes(G=graph, pos=pos, nodelist=graph.nodes(),
        node_color='r', alpha=0.8,
        node_size=[2 * adjacency[location].sum() for location in graph.nodes()])
    nx.draw_networkx_edges(G=graph, pos=pos, edge_color='g',
        alpha=0.2, arrows=False)
    graph_map.drawcountries(linewidth=3)
    graph_map.drawstates(linewidth=0.2)
    graph_map.drawcoastlines(linewidth=3)
    plt.tight_layout()
    plt.savefig('./map_1.png', format='png', dpi=300)
    plt.show()

def main():
    if Path(DATAFRAME_FILE).is_file():
        print('Reading dataframe from pickle file')
        df: pd.DataFrame = pd.read_pickle(DATAFRAME_FILE)
        df.to_csv('data/precipitation/dataframe.csv')
    else:
        print('Reading data from raw files')
        df_data = pd.read_csv(DATA_FILE)
        df_data.columns = pd.to_datetime(df_data.columns, format='D%Y.%m')
        df_locations = pd.read_csv(LOCATIONS_FILE)
        df = pd.concat([df_locations, df_data], axis=1)

        # Remove a random subset of locations from dataframe for quicker testing
        if DROP_RATIO > 0:
            np.random.seed(10)
            _floor = lambda n: int(n // 1)
            drop_indices = np.random.choice(df.index, _floor(DROP_RATIO * len(df)), replace=False)
            df = df.drop(drop_indices)

        df = df.set_index(['Lat', 'Lon'])
        def seq_func(row: pd.Series):
            def func(date, start_date, r: pd.Series):
                sequence = r[(r.index <= date) & (r.index > start_date)]
                return list(sequence) if len(sequence) == LOOKBACK_MONTHS else None
            vec_func = np.vectorize(func, excluded=['r'])
            start_dates = [d - relativedelta(months=LOOKBACK_MONTHS) for d in row.index]
            _row = vec_func(date=row.index, start_date=start_dates, r=row)
            return pd.Series(_row, index=row.index)
        start = datetime.now()
        print('Constructing sequences...')
        df = df.apply(seq_func, axis=1)
        print(f'Sequences constructed; time elapsed: {datetime.now() - start}')
        df = df.stack().reset_index()
        df = df.rename(columns={'level_2': 'date', 0: 'prec_seq', 'Lat': 'lat', 'Lon': 'lon'})
        df = df.set_index('date')
        print('Saving data to pickle file')
        df.to_pickle(DATAFRAME_FILE)

    for y in YEARS:
        for m in MONTHS:
            dt = datetime(y, m, 1)
            try:
                location_data = df.loc[dt]
                # Skip unless the sequence based on the specified lookback time is available
                if location_data['prec_seq'].isnull().values.any():
                    location_data = None
            except KeyError:
                location_data = None
            if location_data is not None:
                build_network(location_data)
                return

if __name__ == '__main__':
    main()