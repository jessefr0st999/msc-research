from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# TODO: convert these to command line flags
DATABASE_URI = 'postgresql+psycopg2://postgres:secret@localhost:5432/msc_research'
DROP_PCT = 85
SAVE_LINKS = 0
SAVE_PRECIPITATION = 1

DATA_FILE = 'data/precipitation/FusedData.csv'
LOCATIONS_FILE = 'data/precipitation/Fused.Locations.csv'
PREC_DATA_FILE = f'data/precipitation/dataframe_drop_{DROP_PCT}.pkl'
LINK_STR_FILE = f'data/precipitation/link_str_drop_{DROP_PCT}.pkl'
CORRS_FILE = f'data/precipitation/corrs_drop_{DROP_PCT}.pkl'

YEARS = range(2000, 2022)
MONTHS = range(1, 13)

AVG_LOOKBACK_MONTHS = 12
LAG_MONTHS = 6
LINK_STR_THRESHOLD = 2.9

def build_network(df: pd.DataFrame):
    if Path(LINK_STR_FILE).is_file():
        print('Reading link strength data from pickle file')
        link_str_df: pd.DataFrame = pd.read_pickle(LINK_STR_FILE)
        corrs_df: pd.DataFrame = pd.read_pickle(CORRS_FILE)
    else:
        print('Calculating link strength data')
        start = datetime.now()
        link_str_df = pd.DataFrame(index=[df['lat'], df['lon']], columns=[df['lat'], df['lon']])
        corrs_df = pd.DataFrame(index=[df['lat'], df['lon']], columns=[df['lat'], df['lon']])
        df = df.reset_index().set_index(['lat', 'lon'])

        # TODO: optimise this
        for i, idx1 in enumerate(df.index):
            if i % 25 == 0:
                print(f'{i} / {len(df.index)} points correlated')
            for j, idx2 in enumerate(df.index[i + 1:]):
                seq_1 = df.loc[idx1, 'prec_seq']
                seq_2 = df.loc[idx2, 'prec_seq']
                # Calculate covariance matrix from both unlagged and lagged sequences
                seq_1_list = [seq_1[LAG_MONTHS :]]
                seq_2_list = [seq_2[LAG_MONTHS :]]
                for i in range(-1, -1 - LAG_MONTHS, -1):
                    seq_1_offset = seq_1[i + LAG_MONTHS : i]
                    seq_2_offset = seq_2[i + LAG_MONTHS : i]
                    seq_1_list.append(seq_1_offset)
                    seq_2_list.append(seq_2_offset)
                # seq_list contains [seq_1_L0, seq_1_L1, ..., seq_2_L0, seq_2_L1, ...]
                seq_list = [*seq_1_list, *seq_2_list]
                # Covariance matrix will have 2 * (LAG_MONTHS + 1) rows and columns
                cov_mat = np.corrcoef(seq_list)
                # Extract coefficients corresponding to
                # (seq_1_L0, seq_2_L0), (seq_1_L0, seq_2_L1), (seq_1_L0, seq_2_L2), ...
                # and (seq_2_L0, seq_1_L1), (seq_2_L0, seq_1_L2), ...
                corrs = np.abs([*cov_mat[LAG_MONTHS + 1 :, 0], *cov_mat[LAG_MONTHS + 1, 1 : LAG_MONTHS + 1]])
                corrs_df.loc[idx1, idx2] = corrs
                # Calculate link strength from correlations as documented
                link_str = (np.max(corrs) - np.mean(corrs)) / np.std(corrs)
                link_str_df.loc[idx1, idx2] = link_str
        print(f'Correlations and link strengths calculated; time elapsed: {datetime.now() - start}')

        # link_str_df is upper triangular
        link_str_df = link_str_df.add(link_str_df.T, fill_value=0).fillna(0)
        if SAVE_LINKS:
            print('Saving link strength data to pickle file')
            link_str_df.to_pickle(LINK_STR_FILE)
            print('Saving correlation data to pickle file')
            corrs_df.to_pickle(CORRS_FILE)

    adjacency = pd.DataFrame(0, columns=link_str_df.columns, index=link_str_df.index)
    adjacency[link_str_df >= LINK_STR_THRESHOLD] = 1
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
        node_size=[adjacency[location].sum() for location in graph.nodes()])
    nx.draw_networkx_edges(G=graph, pos=pos, edge_color='g',
        alpha=0.2, arrows=False)
    graph_map.drawcountries(linewidth=3)
    graph_map.drawstates(linewidth=0.2)
    graph_map.drawcoastlines(linewidth=3)
    plt.tight_layout()
    plt.savefig('./map_1.png', format='png', dpi=300)
    plt.show()

def main():
    if Path(PREC_DATA_FILE).is_file():
        print('Reading precipitation data from pickle file')
        df: pd.DataFrame = pd.read_pickle(PREC_DATA_FILE)
    else:
        print('Reading precipitation data from raw files')
        df_data = pd.read_csv(DATA_FILE)
        df_data.columns = pd.to_datetime(df_data.columns, format='D%Y.%m')
        df_locations = pd.read_csv(LOCATIONS_FILE)
        df = pd.concat([df_locations, df_data], axis=1)

        # Remove a random subset of locations from dataframe for quicker testing
        if DROP_PCT > 0:
            np.random.seed(10)
            _floor = lambda n: int(n // 1)
            drop_indices = np.random.choice(df.index, _floor(0.01 * DROP_PCT * len(df)), replace=False)
            df = df.drop(drop_indices)

        df = df.set_index(['Lat', 'Lon'])
        def seq_func(row: pd.Series):
            def func(start_date, end_date, r: pd.Series):
                sequence = r[(r.index > start_date) & (r.index <= end_date)]
                return list(sequence) if len(sequence) == AVG_LOOKBACK_MONTHS + LAG_MONTHS else None
            vec_func = np.vectorize(func, excluded=['r'])
            start_dates = [d - relativedelta(months=AVG_LOOKBACK_MONTHS + LAG_MONTHS) for d in row.index]
            end_dates = [d for d in row.index]
            _row = vec_func(start_date=start_dates, end_date=end_dates, r=row)
            return pd.Series(_row, index=row.index)
        start = datetime.now()
        print('Constructing sequences...')
        df = df.apply(seq_func, axis=1)
        print(f'Sequences constructed; time elapsed: {datetime.now() - start}')
        df = df.stack().reset_index()
        df = df.rename(columns={'level_2': 'date', 0: 'prec_seq', 'Lat': 'lat', 'Lon': 'lon'})
        df = df.set_index('date')
        if SAVE_PRECIPITATION:
            print('Saving precipitation data to pickle file')
            df.to_pickle(PREC_DATA_FILE)

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