import networkx as nx
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from networkx.algorithms import community
from mpl_toolkits.basemap import Basemap

def read_link_str_df(filename):
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

def get_map(axis=None):
    _map = Basemap(
        projection='merc',
        llcrnrlon=110,
        llcrnrlat=-45,
        urcrnrlon=155,
        urcrnrlat=-10,
        lat_ts=0,
        resolution='l',
        suppress_ticks=True,
        ax=axis,
    )
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

def average_degree(graph: nx.Graph):
    return 2 * len(graph.edges) / len(graph.nodes)

# TODO: Make this faster
def shortest_path_and_eccentricity(graph: nx.Graph):
    shortest_path_lengths = nx.shortest_path_length(graph)
    ecc_by_node = {}
    average_spl_by_node = {}
    for source_node, target_nodes in shortest_path_lengths:
        ecc_by_node[source_node] = np.max(list(target_nodes.values()))
        for point, spl in target_nodes.items():
            if spl == 0:
                target_nodes[point] = len(graph.nodes)
        average_spl_by_node[source_node] = np.average(list(target_nodes.values()))
    return average_spl_by_node, ecc_by_node

def partitions(graph: nx.Graph):
    lcc_graph = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
    return {
        'louvain': [p for p in community.louvain_communities(lcc_graph)],
        'greedy_modularity': [p for p in community.greedy_modularity_communities(lcc_graph)],
        'asyn_lpa': [p for p in community.asyn_lpa_communities(lcc_graph, seed=0)],
        # This one causes the script to hang for an ~1000 edge graph
        # 'girvan_newman': [p for p in community.girvan_newman(lcc_graph)],
    }

def modularity(graph: nx.Graph, communities):
    return community.modularity(graph, communities=communities)

# TODO: implement
def global_average_link_distance(graph: nx.Graph):
    return None

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