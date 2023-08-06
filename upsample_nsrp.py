from datetime import datetime
from dateutil.relativedelta import relativedelta
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

from helpers import prepare_df
from upsample_by_day import get_event_prob_functions

BOM_DAILY_PATH = 'data_unfused/bom_daily'

# t_max = 31
# def inhomogeneous_poisson_process(intensity, t_max):
#     arrival_times = []
#     t = 0
#     while t < t_max:
#         dt = np.random.exponential(scale=1 / intensity(t))
#         t += dt
#         if t <= t_max:
#             arrival_times.append(t)
#     return np.array(arrival_times)
# pdf, norm = get_event_pdf(20, 30, 10)
# arrival_times = inhomogeneous_poisson_process(lambda t: pdf(t / t_max), t_max)

def nsrp_simulation(month_prec, inv_cdf, start_date, previous_cells,
        num_storms_rule, av_num_cells, av_cell_time, av_cell_disp):
    daily_series = pd.Series(0, index=np.arange(start_date,
        start_date + relativedelta(months=1), dtype='datetime64[D]'))
    previous_cells_shifted = [(
        front - (start_date - (start_date - relativedelta(months=1))).days,
        duration, intensity,
    ) for (front, duration, intensity) in previous_cells]

    # NOTE: day, storm fronts and cell fronts are all zero-indexed
    def generate_storm_fronts():
        return [inv_cdf(x) for x in np.random.rand(num_storms_rule(month_prec))]
    
    def generate_cell_fronts(storm_fronts):
        cell_fronts = []
        for storm_front in storm_fronts:
            # Must be minimum of 1 cell within the month
            num_cells = np.random.poisson(av_num_cells) + 1
            cell_fronts.extend(np.random.exponential(av_cell_disp, num_cells) + storm_front)
        return cell_fronts
    
    # Reject storm fronts if:
    # - None are at least a day within the month
    # - Any are within av_cell_disp + av_cell_time of any others
    # - The previous month has a storm towards its end
    def storm_fronts_invalid(fronts):
        day_within_month = False
        for i, x in enumerate(fronts):
            if x < len(daily_series) - 2:
                day_within_month = True
            other_fronts = np.delete(fronts, i)
            if len(other_fronts) > 0 and np.abs(other_fronts - x).min() < av_cell_disp + av_cell_time:
                return True
            for shifted_front, duration, intensity in previous_cells_shifted:
                if shifted_front + duration > x:
                    return True
        return not day_within_month
    
    # Reject cell fronts if none contain any rainfall at least half a day within the month
    def cell_fronts_invalid(fronts):
        for x in fronts:
            if x < len(daily_series) - 1.5:
                return False
        return True

    storm_fronts = generate_storm_fronts()
    attempts = 0
    while storm_fronts_invalid(storm_fronts):
        storm_fronts = generate_storm_fronts()
        if attempts == 100:
            # print(storm_fronts)
            # exit()
            return None, None
        attempts += 1
    storm_fronts = np.sort(storm_fronts)

    cell_fronts = generate_cell_fronts(storm_fronts)
    attempts = 0
    while cell_fronts_invalid(cell_fronts):
        cell_fronts = generate_cell_fronts(storm_fronts)
        if attempts == 100:
            return None, None
        attempts += 1
    cell_fronts = np.sort(cell_fronts)

    cell_durations = np.random.exponential(av_cell_time, len(cell_fronts))
    # Not parameterised as intensities get scaled to respect monthly sum
    cell_intensities = np.random.exponential(1, len(cell_fronts))
    cells = list(zip(cell_fronts, cell_durations, cell_intensities))

    # for front, duration, intensity in zip(cell_fronts, cell_durations, cell_intensities):
    #     print(front, round(duration, 4), round(intensity, 4))
    # Incorporate the previous and current month's cells into the current month's rainfall
    for day in range(len(daily_series)):
        rainfall = 0
        for front, duration, intensity in [*previous_cells_shifted, *cells]:
            # Cells are ordered by time, so go to next day if a front is ahead of current day
            if front > day + 1:
                break
            if front < day:
                overlap_time = min(front + duration - day, 1)
            else:
                overlap_time = min(duration, day + 1 - front)
            if overlap_time > 0:
                # print(day, round(front, 3), round(duration, 3), round(overlap_time, 3))
                rainfall += overlap_time * intensity
        daily_series.iloc[day] = rainfall
    return daily_series, cells

def simulate_nsrp(series, *params):
    monthly_sums = series.groupby(pd.Grouper(freq='1M')).sum()
    previous_cells = []
    sim_series = pd.Series(dtype='float64')
    for i in range(1, len(monthly_sums) - 1):
        start_date = datetime(monthly_sums.index[i].year, monthly_sums.index[i].month, 1)
        if monthly_sums[i] == 0:
            month_sim = pd.Series(0, index=np.arange(start_date,
                start_date + relativedelta(months=1), dtype='datetime64[D]'))
        else:
            _, _, inv_cdf, _ = get_event_prob_functions(monthly_sums.index[i].day,
                monthly_sums[i - 1], monthly_sums[i], monthly_sums[i + 1])
            month_sim, previous_cells = nsrp_simulation(monthly_sums[i], inv_cdf,
                start_date, previous_cells, *params)
            if month_sim is None:
                return None
            # Rescale simulated series to match monthly sum
            # TODO: Consider doing this a nicer way
            month_sim = month_sim * monthly_sums[i] / sum(month_sim)
        # if sum(month_sim) == 0:
        # if start_date.year == 2017 and start_date.month == 6:
        #     print(monthly_sums[i])
        #     print(params[0](monthly_sums[i]))
        #     print(month_sim)
        #     print(previous_cells)
        #     exit()
        sim_series = pd.concat([sim_series, month_sim])
    return sim_series

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
        # print(timescale, diff(np.mean, timescale), diff(np.median, timescale),
        #     diff(np.var, timescale), diff(scipy.stats.skew, timescale))
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
            # print((nsi, anc, act, acd), round(score, 2), round(best_scores[0], 2))
            for month in range(1, 13):
                sim_series_m = sim_series.loc[[t.month == month for t in sim_series.index]]
                prec_series_m = prec_series.loc[[t.month == month for t in prec_series.index]]
                scores[month][i, j] = compare_statistics(sim_series_m, prec_series_m)
                # if np.isnan(scores[month][i, j]):
                #     print(month, i, j)
                #     print(prec_series_m)
                #     print(sim_series_m)
                #     print(sum(sim_series_m))
                #     exit()
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
        ##################################################################
        # prec_df = pd.read_csv(f'{BOM_DAILY_PATH}/BOMDaily086213')
        # results_df_file = 'nsrp_results/score_agg_BOMDaily086213_(-38.38, 144.9).csv'
        # results_pkl_file = 'nsrp_results/score_BOMDaily086213_(-38.38, 144.9).pkl'
        # with open(results_pkl_file, 'rb') as f:
        #     scores = pickle.load(f)
        # print(scores)
        # exit()
        ##################################################################
        prec_df = prec_df.dropna(subset=['Rain'])
        prec_df.index = pd.DatetimeIndex(prec_df['Date'])
        loc = (-prec_df.iloc[0]['Lat'], prec_df.iloc[0]['Lon'])
        results_df_file = f'nsrp_results/score_agg_{path.name}_{str(loc)}.csv'
        results_pkl_file = f'nsrp_results/score_{path.name}_{str(loc)}.pkl'
        prec_series = pd.Series(prec_df['Rain']).dropna().loc['2000-04-01':]
        if args.last_values:
            prec_series = prec_series[-args.last_values:]
        if np.count_nonzero(prec_series) / len(prec_series) < 0.25:
            continue
        print(path.name, loc, os.path.isfile(results_df_file),
            np.count_nonzero(prec_series) / len(prec_series))
        # if os.path.isfile(results_df_file):
        #     continue

        # Parameter sweep to match statistics
        # num_storms_increment_span = np.linspace(10, 70, 7)
        # av_num_cells_span = np.linspace(1, 4, 7)
        # av_cell_time_span = np.linspace(0.5, 3.5, 7)
        num_storms_increment_span = np.linspace(20, 60, 5)
        av_num_cells_span = np.linspace(1, 4, 4)
        av_cell_time_span = np.linspace(0.5, 3.5, 4)
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
            # print(scores[month])
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
        # print(results_df)
        # exit()

    # Analyse best params across all series
    # nsi_counts = [{} for _ in range(13)]
    # anc_counts = [{} for _ in range(13)]
    # act_counts = [{} for _ in range(13)]
    # for path in os.scandir('nsrp_results'):
    #     results_df = pd.read_csv(f'nsrp_results/{path.name}')
    #     for i in range(13):
    #         try:
    #             nsi = int(results_df['num_storms_increment'].iloc[i])
    #             anc = float(results_df['av_num_cells'].iloc[i])
    #             act = float(results_df['av_cell_time'].iloc[i])
    #         except ValueError:
    #             continue
    #         nsi_counts[i][nsi] = nsi_counts[i][nsi] + 1 if nsi in nsi_counts[i] else 1
    #         anc_counts[i][anc] = anc_counts[i][anc] + 1 if anc in anc_counts[i] else 1
    #         act_counts[i][act] = act_counts[i][act] + 1 if act in act_counts[i] else 1
    # for i in range(13):
    #     nsi_fractions = {j: round(nsi_counts[i][j] / sum(nsi_counts[i].values()), 3)
    #         for j in sorted(list(nsi_counts[i].keys()))}
    #     anc_fractions = {j: round(anc_counts[i][j] / sum(anc_counts[i].values()), 3)
    #         for j in sorted(list(anc_counts[i].keys()))}
    #     act_fractions = {j: round(act_counts[i][j] / sum(act_counts[i].values()), 3)
    #         for j in sorted(list(act_counts[i].keys()))}
    #     print()
    #     print(f'month: {i}')
    #     print(nsi_fractions)
    #     print(anc_fractions)
    #     print(act_fractions)

    # TODO: For each param, either pick one value or take some sort of spatial approach
    # to picking, then synthesise daily series with best-determined params
    def calculate_and_save(loc, plot=False):
        pass

    # prec_df, _, _ = prepare_df('data/precipitation', 'FusedData.csv', 'prec')
    # prec_df.index = pd.DatetimeIndex(prec_df.index)
    # for i, loc in enumerate(prec_df.columns):
    #     print(f'{i} / {len(prec_df.columns)}: {loc}')
    #     calculate_and_save(loc)

if __name__ == '__main__':
    main()