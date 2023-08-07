from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from helpers import prepare_df
from upsample_by_day import get_event_prob_functions

NSRP_PARAMS = (
    lambda x: int(1 + min(x // 15, 4)), # rule for number of storms
    1.5, # average number of cells per storm
    1, # average cell duration in days
    1, # average cell displacement from storm front in days
)

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
        sim_series = pd.concat([sim_series, month_sim])
    return sim_series

def main():
    np.random.seed(0)
    prec_df, _, _ = prepare_df('data/precipitation', 'FusedData.csv', 'prec')
    prec_df.index = pd.DatetimeIndex(prec_df.index)

    def calculate_and_save(loc, plot=False):
        prec_series = prec_df[loc]
        upsampled_series = simulate_nsrp(prec_series, *NSRP_PARAMS)
        upsampled_series.name = str(loc)
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
            axis.plot(upsampled_series, 'go-')
            axis.set_xlabel('t')
            axis.set_ylabel('prec')
            axis.set_title('daily upsampled')
            plt.show()

    # for i, loc in enumerate(prec_df.columns):
    #     print(f'{i} / {len(prec_df.columns)}: {loc}')
    #     calculate_and_save(loc)

    # FUSED_SERIES_KEY = (-12.75, 131.5) # Darwin
    # FUSED_SERIES_KEY = (-37.75, 145.5) # Melbourne
    # FUSED_SERIES_KEY = (-28.75, 153.5) # Lismore
    FUSED_SERIES_KEY = (-25.75, 133.5) # Central Australia
    calculate_and_save(FUSED_SERIES_KEY, True)

if __name__ == '__main__':
    main()