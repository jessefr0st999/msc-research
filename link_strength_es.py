from datetime import datetime, timedelta
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx

from link_strength_corr import prepare_graph_plot

DATA_DIR = 'data/precipitation'
OUTPUTS_DIR = 'data/outputs'
DATA_FILE = f'{DATA_DIR}/FusedData.csv'
LOCATIONS_FILE = f'{DATA_DIR}/Fused.Locations.csv'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='sync')
    parser.add_argument('--end_year', type=int, default=2022)
    parser.add_argument('--end_month', type=int, default=3)
    parser.add_argument('--lookback_months', '--lm', type=int, default=None)
    # tau_max for ES, del_tau for ECA
    parser.add_argument('--tau', type=int, default=12)
    parser.add_argument('--edge_density', type=float, default=0.005)
    parser.add_argument('--col_quantile', '--cq', type=float, default=0.95)
    parser.add_argument('--df_quantile', '--dq', type=float, default=0.95)
    parser.add_argument('--save_graph_folder', default=None)
    parser.add_argument('--save_links', action='store_true', default=False)
    args = parser.parse_args()

    dot_to_p = lambda num: str(num).replace('.', 'p')
    month_str = str(args.end_month) if args.end_month >= 10 else f'0{args.end_month}'
    base_filename = (f'event_{args.method}_colq_{dot_to_p(args.col_quantile)}_dfq_{dot_to_p(args.df_quantile)}'
        f'_tau_{args.tau}_{args.end_year}_{month_str}_lm_{str(args.lookback_months)}')
    event_data_file = f'{OUTPUTS_DIR}/{base_filename}.csv'
    if Path(event_data_file).is_file():
        print(f'Reading event data from file {event_data_file}')
        event_sync_df = pd.read_csv(event_data_file, index_col=[0, 1], header=[0, 1])
    else:
        df = pd.read_csv(DATA_FILE)
        df.columns = pd.to_datetime(df.columns, format='D%Y.%m')
        locations_df = pd.read_csv(LOCATIONS_FILE)
        df = pd.concat([locations_df, df], axis=1)
        df = df.set_index(['Lat', 'Lon'])
        df = df.T
        if args.lookback_months:
            end_months = args.end_year * 12 + args.end_month
            df = df.loc[datetime((end_months - args.lookback_months) // 12,
                    (end_months  - args.lookback_months) % 12 + 1, 1)
                : datetime(args.end_year, args.end_month, 1)]

        event_df = pd.DataFrame(0, columns=df.columns, index=df.index)
        event_df[df > np.quantile(df, args.df_quantile)] = 1
        print(f'Number of extreme events over whole dataframe: {event_df.to_numpy().sum()}')
        if args.col_quantile:
            extreme_column_events_df = pd.DataFrame(0, columns=df.columns, index=df.index)
            for c in df.columns:
                extreme_column_events_df[c][df[c] > np.quantile(df[c], args.col_quantile)] = 1
            print(f'Number of extreme events over columns: {extreme_column_events_df.to_numpy().sum()}')
            event_df &= extreme_column_events_df
            print(f'Number of merged extreme events: {event_df.to_numpy().sum()}')
        event_array = np.array(event_df)

        n = event_array.shape[1]
        event_sync_array = np.zeros((n, n))
        for i in range(0, n):
            if i % 100 == 0:
                print(f'{i} / {n}')
            for j in range(i + 1, n):
                if args.method == 'sync':
                    event_sync_array[i, j], event_sync_array[j, i] = event_sync(
                        event_array[:, i], event_array[:, j], taumax=args.tau)
                elif args.method == 'ca':
                    # Use trigger rather than precursor
                    _, event_sync_array[i, j], _, event_sync_array[j, i] = event_ca(
                        event_array[:, i], event_array[:, j], delT=args.tau)
                else:
                    raise ValueError(f'Invalid event method: {args.method}')

        event_sync_df = pd.DataFrame(event_sync_array, columns=event_df.columns,
            index=event_df.columns)
        event_data_file = f'{OUTPUTS_DIR}/{base_filename}.csv'
        print(f'Saving event data to file {event_data_file}')
        event_sync_df.to_csv(event_data_file)

    # Symmetric ES/ECA matrix
    sym_event_array = np.array(event_sync_df) + np.array(event_sync_df).T
    sym_event_df = pd.DataFrame(sym_event_array, columns=event_sync_df.columns,
        index=event_sync_df.columns).fillna(0)
    threshold = np.quantile(sym_event_df, 1 - args.edge_density)
    print(f'Fixed edge density {args.edge_density} gives threshold {threshold}')
    adjacency = pd.DataFrame(0, columns=event_sync_df.columns,
        index=event_sync_df.columns)
    adjacency[sym_event_df > threshold] = 1
    graph = nx.from_numpy_array(adjacency.values)
    graph = nx.relabel_nodes(graph, dict(enumerate(adjacency.columns)))
    locations_df = pd.read_csv(LOCATIONS_FILE)
    prepare_graph_plot(graph, adjacency, locations_df['Lon'], locations_df['Lat'],
        plot=not args.save_graph_folder)
    if args.save_graph_folder:
        plt.gcf().set_size_inches(32, 18)
        base_filename = (f'event_{args.method}_colq_{dot_to_p(args.col_quantile)}_dfq_{dot_to_p(args.df_quantile)}'
            f'_tau_{args.tau}_{args.end_year}_{month_str}_lm_{str(args.lookback_months)}')
        title = (f'Event {args.method}: {datetime(args.end_year, args.end_month, 1).strftime("%B")}'
            f' {args.end_year}, ({args.lookback_months} lookback months)')
        plt.title(title)
        filename = f'{args.save_graph_folder}/{base_filename}_ed_{dot_to_p(args.edge_density)}.png'
        print(f'Saving graph plot to file {filename}')
        plt.savefig(filename)

def event_sync(seq_1, seq_2, taumax):
    '''Calculates the directed event synchronization from two event
    series X and Y.

    :type seq_1: 1D Numpy array
    :arg seq_1: Event series containing '0's and '1's
    :type seq_2: 1D Numpy array
    :arg seq_2: Event series containing '0's and '1's
    :rtype: list
    :return: [Event synchronization XY, Event synchronization YX]

    Reference:
    J.F. Donges, J. Heitzig, B. Beronov, M. Wiedermann, J. Runge, Q.-Y. Feng,
    L. Tupikina, V. Stolbova, R.V. Donner, N. Marwan, H.A. Dijkstra,
    and J. Kurths, "Unified functional network and nonlinear time series analysis
    for complex systems science: The pyunicorn package"
    '''
    # Get time indices (type boolean or simple '0's and '1's)
    ex = np.array(np.where(seq_1), dtype=np.int8)
    ey = np.array(np.where(seq_2), dtype=np.int8)
    # Number of events
    lx = ex.shape[1]
    ly = ey.shape[1]
    if lx == 0 or ly == 0:              # Division by zero in output
        return np.nan, np.nan
    if lx in [1, 2] or ly in [1, 2]:    # Too few events to calculate
        return 0., 0.
    # Array of distances
    dstxy2 = 2 * (np.repeat(ex[:, 1:-1].T, ly-2, axis=1)
                    - np.repeat(ey[:, 1:-1], lx-2, axis=0))
    # Dynamical delay
    diffx = np.diff(ex)
    diffy = np.diff(ey)
    diffxmin = np.minimum(diffx[:, 1:], diffx[:, :-1])
    diffymin = np.minimum(diffy[:, 1:], diffy[:, :-1])
    tau2 = np.minimum(np.repeat(diffxmin.T, ly-2, axis=1),
                        np.repeat(diffymin, lx-2, axis=0))
    tau2 = np.minimum(tau2, 2 * taumax)
    # Count equal time events and synchronised events
    eqtime = dstxy2.size - np.count_nonzero(dstxy2)

    # Calculate boolean matrices of coincidences
    Axy = (dstxy2 > 0) * (dstxy2 <= tau2)
    Ayx = (dstxy2 < 0) * (dstxy2 >= -tau2)

    # Loop over coincidences and determine number of double counts
    # by checking at least one event of the pair is also coincided
    # in other direction
    countxydouble = countyxdouble = 0

    for i, j in np.transpose(np.where(Axy)):
        countxydouble += np.any(Ayx[i, :]) or np.any(Ayx[:, j])
    for i, j in np.transpose(np.where(Ayx)):
        countyxdouble += np.any(Axy[i, :]) or np.any(Axy[:, j])

    # Calculate counting quantities and subtract half of double countings
    countxy = np.sum(Axy) + 0.5 * eqtime - 0.5 * countxydouble
    countyx = np.sum(Ayx) + 0.5 * eqtime - 0.5 * countyxdouble

    norm = np.sqrt((lx-2) * (ly-2))
    return countxy / norm, countyx / norm

def event_ca(seq_1, seq_2, delT, tau=0, ts1=None, ts2=None):
    '''Event coincidence analysis:
    Returns the precursor and trigger coincidence rates of two event series
    X and Y.

    :type seq_1: 1D Numpy array
    :arg seq_1: Event series containing '0's and '1's
    :type seq_2: 1D Numpy array
    :arg seq_2: Event series containing '0's and '1's
    :arg delT: coincidence interval width
    :arg int tau: lag parameter
    :rtype: list
    :return: [Precursor coincidence rate XY, Trigger coincidence rate XY,
          Precursor coincidence rate YX, Trigger coincidence rate YX]

    Reference:
    J.F. Donges, J. Heitzig, B. Beronov, M. Wiedermann, J. Runge, Q.-Y. Feng,
    L. Tupikina, V. Stolbova, R.V. Donner, N. Marwan, H.A. Dijkstra,
    and J. Kurths, "Unified functional network and nonlinear time series analysis
    for complex systems science: The pyunicorn package"
    '''

    # Count events that cannot be coincided due to tau and delT
    if not (tau == 0 and delT == 0):
        # Start of seq_1
        n11 = np.count_nonzero(seq_1[:tau+delT])
        # End of seq_1
        n12 = np.count_nonzero(seq_1[-(tau+delT):])
        # Start of seq_2
        n21 = np.count_nonzero(seq_2[:tau+delT])
        # End of seq_2
        n22 = np.count_nonzero(seq_2[-(tau+delT):])
    else:
        # Instantaneous coincidence
        n11, n12, n21, n22 = 0, 0, 0, 0
    # Get time indices
    if ts1 is None:
        e1 = np.where(seq_1)[0]
    else:
        e1 = ts1[seq_1]
    if ts2 is None:
        e2 = np.where(seq_2)[0]
    else:
        e2 = ts2[seq_2]
    del seq_1, seq_2, ts1, ts2
    # Number of events
    l1 = len(e1)
    l2 = len(e2)
    try:
        # Array of all interevent distances
        dst = (np.array([e1]*l2).T - np.array([e2]*l1))
        # Count coincidences with array slicing
        prec12 = np.count_nonzero(np.any(((dst - tau >= 0)
                                        * (dst - tau <= delT))[n11:, :],
                                        axis=1))
        trig12 = np.count_nonzero(np.any(((dst - tau >= 0)
                                        * (dst - tau <= delT))
                                        [:, :dst.shape[1]-n22],
                                        axis=0))
        prec21 = np.count_nonzero(np.any(((-dst - tau >= 0)
                                        * (-dst - tau <= delT))[:, n21:],
                                        axis=0))
        trig21 = np.count_nonzero(np.any(((-dst - tau >= 0)
                                        * (-dst - tau <= delT))
                                        [:dst.shape[0]-n12, :],
                                        axis=1))
    except (ValueError, IndexError):
        return (0, 0, 0, 0)
    # Normalisation and output
    return (np.float32(prec12)/(l1-n11), np.float32(trig12)/(l2-n22),
            np.float32(prec21)/(l2-n21), np.float32(trig21)/(l1-n12))

if __name__ == '__main__':
    start = datetime.now()
    main()
    print(f'Total time elapsed: {datetime.now() - start}')