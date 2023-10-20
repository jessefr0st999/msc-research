from datetime import datetime
from dateutil.relativedelta import relativedelta
import argparse
from scipy import integrate

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from helpers import prepare_df

# Best params as calculated in nsrp_calibration.py
NSRP_PARAMS = (
    lambda x: int(1 + min(x // 15, 4)), # rule for number of storms
    1.5, # average number of cells per storm
    1, # average cell duration in days
    1, # average cell displacement from storm front in days
)
ATTEMPTS_LIMIT = 10

def get_event_prob_functions(days_in_month, p, c, n):
    def next_func(t):
        return c + (n - c) * t
    def prev_func(t):
        return c + (p - c) * (1 - t)
    if (p >= c and n > c) or (p > c and n >= c) or (p <= c and n < c) or (p < c and n <= c):
        alpha = (c - p) / (2*c - p - n)
        norm = integrate.quad(prev_func, 0, alpha)[0] + integrate.quad(next_func, alpha, 1)[0]
        def pdf(t):
            return prev_func(t) / norm if t < alpha else next_func(t) / norm
    elif c == p and c == n:
        norm = 1
        def pdf(t):
            return 1
    elif abs(c - p) >= abs(c - n):
        norm = integrate.quad(prev_func, 0, 1)[0]
        def pdf(t):
            return prev_func(t) / norm
    elif abs(c - n) > abs(c - p):
        norm = integrate.quad(next_func, 0, 1)[0]
        def pdf(t):
            return next_func(t) / norm
    def cdf(t):
        return integrate.quad(pdf, 0, t)[0]
    inv_cdf_list = []
    for i in range(days_in_month):
        inv_cdf_list.append(cdf((i + 1) / days_in_month))
    def inv_cdf(p):
        for i, q in enumerate(inv_cdf_list):
            if q >= p:
                return i
    return pdf, cdf, inv_cdf, norm, prev_func, next_func

def nsrp_simulation(month_prec, inv_cdf, start_date, previous_cells,
        num_storms_rule, av_num_cells, av_cell_time, av_cell_disp):
    daily_series = pd.Series(0, index=np.arange(start_date,
        start_date + relativedelta(months=1), dtype='datetime64[D]'))
    previous_cells_shifted = [(
        front - (start_date - (start_date - relativedelta(months=1))).days,
        duration, intensity, storm_front,
    ) for (front, duration, intensity, storm_front) in previous_cells]

    # NOTE: day, storm fronts and cell fronts are all zero-indexed
    def generate_storm_fronts():
        return [inv_cdf(x) for x in np.random.rand(num_storms_rule(month_prec))]
    
    def generate_cell_fronts(storm_fronts):
        cell_fronts = []
        cell_storm_fronts = []
        for storm_front in storm_fronts:
            # Must be minimum of 1 cell within the month
            num_cells = np.random.poisson(av_num_cells) + 1
            cell_fronts.extend(np.random.exponential(av_cell_disp, num_cells) + storm_front)
            cell_storm_fronts.extend([storm_front] * num_cells)
        return cell_fronts, cell_storm_fronts
    
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
            for shifted_front, duration, intensity, storm_front in previous_cells_shifted:
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
            return None, None
        attempts += 1
    storm_fronts = np.sort(storm_fronts)

    cell_fronts, cell_storm_fronts = generate_cell_fronts(storm_fronts)
    attempts = 0
    while cell_fronts_invalid(cell_fronts):
        cell_fronts, cell_storm_fronts = generate_cell_fronts(storm_fronts)
        if attempts == 100:
            return None, None
        attempts += 1
    cell_fronts_sort_index = np.argsort(cell_fronts)
    cell_fronts = np.array(cell_fronts)[cell_fronts_sort_index]
    cell_storm_fronts = np.array(cell_storm_fronts)[cell_fronts_sort_index]

    cell_durations = np.random.exponential(av_cell_time, len(cell_fronts))
    # Not parameterised as intensities get scaled to respect monthly sum
    cell_intensities = np.random.exponential(1, len(cell_fronts))
    cells = list(zip(cell_fronts, cell_durations, cell_intensities, cell_storm_fronts))

    # Incorporate the previous and current month's cells into the current month's rainfall
    for day in range(len(daily_series)):
        rainfall = 0
        for front, duration, intensity, storm_front in [*previous_cells_shifted, *cells]:
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
            # pdf, cdf, inv_cdf, norm, prev_func, next_func = \
            #     get_event_prob_functions(monthly_sums.index[i].day,
            #         monthly_sums[i - 1], monthly_sums[i], monthly_sums[i + 1])
            # pdf, cdf, inv_cdf, norm, prev_func, next_func = \
            #     get_event_prob_functions(monthly_sums.index[i].day,
            #         10, 200, 100)
            pdf, cdf, inv_cdf, norm, prev_func, next_func = \
                get_event_prob_functions(monthly_sums.index[i].day,
                    10, 200, 100)
            month_sim, previous_cells = nsrp_simulation(monthly_sums[i], inv_cdf,
                start_date, previous_cells, *params)
            if month_sim is None:
                return None
            # Rescale simulated series to match monthly sum
            month_sim = month_sim * monthly_sums[i] / sum(month_sim)
            # if start_date.year == 2022 and start_date.month == 2:
            #     for c in previous_cells:
            #         print(c)
            #     fronts, durations, intensities, storm_fronts = zip(*previous_cells)
            #     intensities = np.array(intensities) * monthly_sums[i] / sum(intensities)
            #     cell_ends = [f + d for f, d in zip(fronts, durations)]
            #     colours = ['r', 'g', 'b', 'k', 'm', 'c']
            #     storm_to_colour = {s: colours[i] for i, s in enumerate(np.unique(storm_fronts))}
            #     cell_colours = [storm_to_colour[s] for s in storm_fronts]

                # figure, axis = plt.subplots(1)
                # axis.hlines(intensities, fronts, cell_ends, cell_colours)
                # axis.vlines(fronts, 0, intensities, cell_colours, 'dotted')
                # axis.vlines(cell_ends, 0, intensities, cell_colours, 'dotted')
                # for storm, colour in storm_to_colour.items():
                #     axis.plot(storm, 0, color=colour, marker='o')
                # axis.set_xlabel('time (days into month)')
                # axis.set_xlim([0, 29])
                # plt.show()

                # figure, axis = plt.subplots(1)
                # month_sim.index = range(1, monthly_sums.index[i].day + 1)
                # axis.hlines(intensities, fronts, cell_ends, 'gray')
                # axis.vlines(fronts, 0, intensities, 'gray', 'dotted')
                # axis.vlines(cell_ends, 0, intensities, 'gray', 'dotted')
                # axis.set_xlabel('time (days into month)')
                # axis.set_xlim([0, 29])
                # axis.plot(month_sim, 'ro-')
                # plt.show()
                # exit()

                # figure, axes = plt.subplots(1, 2)
                # axes = iter(axes.flatten())
                # axis = next(axes)
                # # figure, axis = plt.subplots(1)
                # t_vec = np.linspace(0, 1, 1001)
                # t_days_vec = t_vec * monthly_sums.index[i].day
                # # axis.plot(t_days_vec, [0 for t in t_vec])
                # axis.plot(t_days_vec, [prev_func(t) for t in t_vec], 'b-', label='y1')
                # axis.plot(t_days_vec, [next_func(t) for t in t_vec], 'g-', label='y2')
                # axis.legend()
                # axis.set_xlabel('time (days into month)')
                # axis.set_ylabel('prec')
                
                # axis = next(axes)
                # axis.plot(t_days_vec, [prev_func(t) / norm for t in t_vec], 'b-', label='y1')
                # axis.plot(t_days_vec, [next_func(t) / norm for t in t_vec], 'g-', label='y2')
                # axis.plot(t_days_vec, [pdf(t) for t in t_vec], 'r-', label='PDF')
                # axis.plot(t_days_vec, [cdf(t) for t in t_vec], 'y-', label='CDF')
                # axis.legend()
                # axis.set_xlabel('time (days into month)')
                # axis.set_ylabel('probability')
                
                # axis = next(axes)
                # p_vec = np.linspace(0, 1, 1001)
                # axis.plot(p_vec, [inv_cdf(p) for p in p_vec], 'y-', label='inverse CDF')
                # axis.legend()
                # axis.set_xlabel('random number in [0, 1]')
                # axis.set_ylabel('time (days into month)')
                # plt.show()
        sim_series = pd.concat([sim_series, month_sim])
    return sim_series

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', default=False)
    parser.add_argument('--lat', type=float, default=-28.75) # Lismore
    parser.add_argument('--lon', type=float, default=153.5) # Lismore
    parser.add_argument('--plot_date', default=None)
    args = parser.parse_args()

    np.random.seed(0)
    prec_df, _, _ = prepare_df('data/precipitation', 'FusedData.csv', 'prec')
    prec_df.index = pd.DatetimeIndex(prec_df.index)

    def calculate_series(loc, save=False, plot=False):
        prec_series = prec_df[loc]
        # Can occasionally fail; try several times before exiting
        attempts = 0
        while True:
            if attempts == ATTEMPTS_LIMIT:
                print(f'No feasible series generated for {str(loc)} after '
                    f'{ATTEMPTS_LIMIT} attempts; exiting.')
                exit()
            upsampled_series = simulate_nsrp(prec_series, *NSRP_PARAMS)
            if upsampled_series is not None:
                break
            attempts += 1
        upsampled_series.name = str(loc)
        if save:
            upsampled_series.to_csv(f'data/fused_upsampled/fused_daily_nsrp_{loc[0]}_{loc[1]}.csv')
        if plot:
            figure, axes = plt.subplots(2, 1)
            axes = iter(axes.flatten())
            axis = next(axes)
            axis.plot(prec_series, 'mo-')
            axis.set_xlabel('t')
            axis.set_ylabel('prec')
            axis.set_title('monthly')

            axis = next(axes)
            axis.plot(upsampled_series, 'g-')
            axis.set_xlabel('t')
            axis.set_ylabel('prec')
            axis.set_title('daily upsampled')
            plt.show()
        return attempts

    if args.all:
        for i, loc in enumerate(prec_df.columns):
            attempts = calculate_series(loc, save=True, plot=False)
            if attempts > 1:
                print(f'{i} / {len(prec_df.columns)}: {loc} ({attempts} attempts)')
            else:
                print(f'{i} / {len(prec_df.columns)}: {loc}')
    else:
        calculate_series((args.lat, args.lon), save=False, plot=True)

if __name__ == '__main__':
    main()