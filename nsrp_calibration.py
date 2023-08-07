from datetime import datetime
import itertools
import argparse
import os
import ast
import pickle

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy

from helpers import get_map, scatter_map, configure_plots
from upsample_nsrp import simulate_nsrp

BOM_DAILY_PATH = 'data_unfused/bom_daily'

def compare_statistics(s1, s2):
    timescales = ['2D', '4D', '7D', '14D', '1M', '2M', '3M']
    score = 0
    if sum(s1) == 0 and sum(s2) == 0:
        return 0
    def diff(stat_func, timescale):
        s1_agg = s1.groupby(pd.Grouper(freq=timescale)).sum()
        s2_agg = s2.groupby(pd.Grouper(freq=timescale)).sum()
        s1_stat = stat_func(s1_agg)
        s2_stat = stat_func(s2_agg)
        if s1_stat == 0 and s2_stat == 0:
            return 0
        return np.abs(s1_stat - s2_stat) / max(np.abs([s1_stat, s2_stat]))
    for timescale in timescales:
        score += diff(np.var, timescale)
        score += diff(np.mean, timescale)
        score += diff(np.median, timescale)
        score += diff(scipy.stats.skew, timescale)
    return score

def calculate_best_params(prec_series, param_combs, simulations, plot=False):
    scores = [np.zeros((len(param_combs), simulations)) for _ in range(13)]
    for i, (nsi, anc, act, acd) in enumerate(param_combs):
        if i % 10 == 0:
            print(f'{i} / {len(param_combs)}')
        num_storms_rule = lambda x: int(1 + min(x // nsi, 4))
        for j in range(simulations):
            sim_series = simulate_nsrp(prec_series, num_storms_rule, anc, act, acd)
            if sim_series is None:
                for month in range(13):
                    scores[month][i, j] = np.nan
                continue
            if plot:
                figure, axis = plt.subplots(1)
                axis.plot(prec_series, 'mo-', label='actual')
                axis.plot(sim_series, 'go-', label='simulated')
                axis.set_xlabel('t')
                axis.set_ylabel('prec')
                axis.legend()
                plt.show()
            scores[0][i, j] = compare_statistics(sim_series, prec_series)
            for month in range(1, 13):
                sim_series_m = sim_series.loc[[t.month == month for t in sim_series.index]]
                prec_series_m = prec_series.loc[[t.month == month for t in prec_series.index]]
                scores[month][i, j] = compare_statistics(sim_series_m, prec_series_m)
    return scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--last_values', type=int, default=None)
    parser.add_argument('--simulations', type=int, default=1)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.seed:
        np.random.seed(args.seed)

    # Select random files (each corresponding to a location) from the BOM daily dataset
    info_df = pd.read_csv('bom_info.csv', index_col=0, converters={0: ast.literal_eval})
    filenames = set(info_df.sample(args.num_samples)['filename']) if args.num_samples else None
    for path in os.scandir(BOM_DAILY_PATH):
        if not path.is_file() or (args.num_samples and path.name not in filenames):
            continue
        prec_df = pd.read_csv(f'{BOM_DAILY_PATH}/{path.name}')
        prec_df = prec_df.dropna(subset=['Rain'])
        prec_df.index = pd.DatetimeIndex(prec_df['Date'])
        loc = (-prec_df.iloc[0]['Lat'], prec_df.iloc[0]['Lon'])
        # Scan series outside of the south-east
        # if loc[0] < -28.5 and loc[1] > 141.5:
        #     continue
        results_df_file = f'nsrp_results/score_agg_v2_{path.name}_{str(loc)}.csv'
        results_pkl_file = f'nsrp_results/score_v2_{path.name}_{str(loc)}.pkl'
        prec_series = pd.Series(prec_df['Rain']).dropna().loc['2000-04-01':]
        if args.last_values:
            prec_series = prec_series[-args.last_values:]
        if np.count_nonzero(prec_series) / len(prec_series) < 0.25:
            continue
        print(path.name, loc, os.path.isfile(results_df_file),
            np.count_nonzero(prec_series) / len(prec_series))
        if os.path.isfile(results_df_file):
            continue

        # Parameter sweep to match statistics
        # NOTE: First sweep
        # num_storms_increment_span = np.linspace(20, 60, 5)
        # av_num_cells_span = np.linspace(1, 4, 4)
        # av_cell_time_span = np.linspace(0.5, 3.5, 4)
        # av_cell_disp_span = [1]
        # NOTE: Second sweep
        num_storms_increment_span = np.linspace(15, 24, 4)
        av_num_cells_span = np.linspace(0.6, 1.5, 4)
        av_cell_time_span = np.linspace(0.25, 1, 4)
        av_cell_disp_span = [1]
        param_combs = list(itertools.product(*[num_storms_increment_span,
            av_num_cells_span, av_cell_time_span, av_cell_disp_span]))
        start = datetime.now()
        scores = calculate_best_params(prec_series, param_combs, args.simulations)
        print(f'Time elapsed: {datetime.now() - start}')

        with open(results_pkl_file, 'wb') as f:
            pickle.dump(scores, f)
        best_scores = []
        best_scores_std = []
        best_params = []
        for month in range(13):
            scores_mean = np.nanmean(scores[month], axis=1)
            scores_std = np.nanstd(scores[month], axis=1)
            best_comb_index = np.nanargmin(scores_mean)
            best_scores.append(scores_mean[best_comb_index])
            best_scores_std.append(scores_std[best_comb_index])
            best_params.append(param_combs[best_comb_index])
        df_rows = []
        for i, (params, score, score_std) in enumerate(zip(
                best_params, best_scores, best_scores_std)):
            if params:
                nsi, anc, act, acd = params
            else:
                nsi, anc, act, acd = None, None, None, None
            df_rows.append({
                'month': i,
                'num_storms_increment': nsi,
                'av_num_cells': anc,
                'av_cell_time': act,
                'av_cell_disp': acd,
                'best_score': score,
                'best_score_std': score_std,
            })
        results_df = pd.DataFrame.from_records(df_rows)
        results_df.to_csv(results_df_file)

    # Analyse best params across all series
    nsi_counts = [{} for _ in range(13)]
    anc_counts = [{} for _ in range(13)]
    act_counts = [{} for _ in range(13)]
    plot_data = [[] for _ in range(13)]
    loc_list = []
    for path in os.scandir('nsrp_results'):
        if not path.name.startswith('score_agg_v2'):
            continue
        results_df = pd.read_csv(f'nsrp_results/{path.name}')
        loc_list.append(ast.literal_eval(path.name.split('(')[1].split(')')[0]))
        for i in range(13):
            try:
                nsi = int(results_df['num_storms_increment'].iloc[i])
                anc = float(results_df['av_num_cells'].iloc[i])
                act = float(results_df['av_cell_time'].iloc[i])
            except ValueError:
                continue
            plot_data[i].append([nsi, anc, act])
            nsi_counts[i][nsi] = nsi_counts[i][nsi] + 1 if nsi in nsi_counts[i] else 1
            anc_counts[i][anc] = anc_counts[i][anc] + 1 if anc in anc_counts[i] else 1
            act_counts[i][act] = act_counts[i][act] + 1 if act in act_counts[i] else 1
    for i in range(13):
        nsi_fractions = {j: round(nsi_counts[i][j] / sum(nsi_counts[i].values()), 3)
            for j in sorted(list(nsi_counts[i].keys()))}
        anc_fractions = {j: round(anc_counts[i][j] / sum(anc_counts[i].values()), 3)
            for j in sorted(list(anc_counts[i].keys()))}
        act_fractions = {j: round(act_counts[i][j] / sum(act_counts[i].values()), 3)
            for j in sorted(list(act_counts[i].keys()))}
        print()
        print(f'month: {i}')
        print(nsi_fractions)
        print(anc_fractions)
        print(act_fractions)

    series_names = ['num_storms_increment', 'av_num_cells', 'av_cell_time']
    lats, lons = list(zip(*loc_list))
    for m in range(13):
        figure, axes = plt.subplots(1, 3, layout='compressed')
        axes = iter(axes.flatten())
        for i in range(3):
            axis = next(axes)
            _map = get_map(axis)
            mx, my = _map(lons, lats)
            series = [x[i] for x in plot_data[m]]
            axis.set_title(f'month {m}: {series_names[i]}')
            scatter_map(axis, mx, my, series, cb_min=np.min(series), cb_max=np.max(series))
        plt.show()

if __name__ == '__main__':
    main()