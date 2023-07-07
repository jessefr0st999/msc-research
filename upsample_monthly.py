from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from math import floor

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import integrate
np.random.seed(0)

from helpers import prepare_df

def upsample_month(series, date):
    # if date.year != 2020 or date.month != 4:
    #     return
    c = series.loc[date]
    month_series_index = np.arange(date, date + relativedelta(months=1), timedelta(days=1)).astype(datetime)
    month_series = pd.Series(0, index=month_series_index)

    # If the month's value is small, just assign it all to one day
    if c < 1:
        month_series[np.random.randint(len(month_series))] = c
        return month_series

    # Inverse transform sampling for weighting the days to which values will be added
    current_iloc = series.index.get_loc(date)
    p = c if current_iloc == 0 else series.iloc[current_iloc - 1]
    n = c if current_iloc == len(series) - 1 else series.iloc[current_iloc + 1]
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
    for i in range(len(month_series)):
        inv_cdf_list.append(cdf((i + 1) / len(month_series)))
    def inv_cdf(p):
        for i, q in enumerate(inv_cdf_list):
            if q >= p:
                return i

    # Initialise by evenly distributing the monthly rainfall over a random subset of its days
    non_zero_indices = np.random.choice(len(month_series), 10, replace=False)
    month_series.iloc[non_zero_indices] = c / len(non_zero_indices)
    # month_df = pd.DataFrame(month_series.copy())
    # Weighted drawing based on previous and next months using inverse transform sampling
    def get_i_and_increment():
        base_increment = c / len(non_zero_indices) / 2
        plus_i = inv_cdf(np.random.rand(1)[0])
        minus_i = np.random.randint(len(month_series))
        # If a decrement would result in a negative, change it so that it results in zero
        if month_series.iloc[minus_i] - base_increment < 0:
            increment = month_series.iloc[minus_i]
        else:
            increment = base_increment
        return plus_i, minus_i, increment
    for _ in range(ITERATIONS):
        plus_i, minus_i, increment = get_i_and_increment()
        # Redraw if indices are equal or if the subtracted one is already zero
        while plus_i == minus_i or month_series.iloc[minus_i] == 0:
            plus_i, minus_i, increment = get_i_and_increment()
        try:
            month_series.iloc[plus_i] += increment
            month_series.iloc[minus_i] -= increment
        except TypeError:
            print(month_series)
            print(plus_i)
            print(minus_i)
            figure, axis = plt.subplots(1)
            t_vec = np.linspace(0, 1, 1001)
            axis.plot(t_vec, [next_func(t) / norm for t in t_vec], 'g-')
            axis.plot(t_vec, [prev_func(t) / norm for t in t_vec], 'b-')
            axis.plot(t_vec, [pdf(t) for t in t_vec], 'r-')
            axis.plot(t_vec, [cdf(t) for t in t_vec], 'm-')
            axis.set_xlabel('time')
            axis.set_ylabel('probability')
            plt.show()
            
            figure, axis = plt.subplots(1)
            p_vec = np.linspace(0, 1, 1001)
            axis.plot(p_vec, [inv_cdf(p) for p in p_vec], 'm-')
            axis.set_xlabel('probability')
            axis.set_ylabel('time')
            plt.show()
            exit()
    # month_df['after'] = month_series
    # print(month_df)
    # exit()
    print(date, f'{(month_series >= 1).sum()}/{len(month_series)}', round(month_series.max(), 2))
    return month_series

# Higher iteration count allows for more zero and extreme high precipitations to be generated
ITERATIONS = 3000
def calculate_and_save(loc, plot=False):
    prec_series = prec_df[loc]
    # prec_series = prec_df[loc][-48:]
    upsampled_series_list = []
    for i in range(len(prec_series)):
        upsampled_series_list.append(upsample_month(prec_series, prec_series.index[i]))
    upsampled_series = pd.concat(upsampled_series_list)
    upsampled_series.name = str(loc)
    upsampled_series.to_csv(f'data/fused_upsampled/fused_daily_{loc[0]}_{loc[1]}_it_{ITERATIONS}.csv')
    if plot:
        figure, axes = plt.subplots(2, 1)
        axes = iter(axes.flatten())
        axis = next(axes)
        axis.plot(prec_series, 'mo-')
        axis.set_xlabel('t')
        axis.set_ylabel('prec')

        axis = next(axes)
        axis.plot(upsampled_series, 'go-')
        axis.set_xlabel('t')
        axis.set_ylabel('prec')
        plt.show()

prec_df, _, _ = prepare_df('data/precipitation', 'FusedData.csv', 'prec')
prec_df.index = pd.DatetimeIndex(prec_df.index)

for i, loc in enumerate(prec_df.columns):
    if i < 1073:
        continue
    print(f'{i} / {len(prec_df.columns)}: {loc}')
    calculate_and_save(loc)

# FUSED_SERIES_KEY = (-12.75, 131.5) # Darwin
# FUSED_SERIES_KEY = (-37.75, 145.5) # Melbourne
# FUSED_SERIES_KEY = (-28.75, 153.5) # Lismore
# calculate_and_save(FUSED_SERIES_KEY, True)