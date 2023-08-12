import os
import ast

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from helpers import get_map, scatter_map, prepare_df
from unami_2009_helpers import prepare_model_df

FUSED_DAILY_PATH = 'data/fused_upsampled'
# NSRP_DATASET = False
NSRP_DATASET = True
X_DF_NAME = 'x_df_nsrp.csv' if NSRP_DATASET else 'x_df.csv'
PREC_INC = 0.5
DF_QUANTILES = [0.05, 0.95]
# DF_QUANTILES = [0.1, 0.9]
# MONTH_QUANTILES = [0.05, 0.95]
MONTH_QUANTILES = [0.1, 0.9]
# LOC_QUANTILES = [0.1, 0.9]
LOC_QUANTILES = [0.05, 0.95]

# x_df = None
# i = 0
# for path in os.scandir(FUSED_DAILY_PATH):
#     if NSRP_DATASET and not path.name.startswith('fused_daily_nsrp'):
#         continue
#     if not NSRP_DATASET and not path.name.endswith('it_3000.csv'):
#         continue
#     if i % 25 == 0:
#         print(i)
#     prec_df = pd.read_csv(f'{FUSED_DAILY_PATH}/{path.name}', index_col=0)
#     prec_series = pd.Series(prec_df.values[:, 0], index=pd.DatetimeIndex(prec_df.index))
#     loc = ast.literal_eval(prec_df.columns[0])
#     model_df = prepare_model_df(prec_series, PREC_INC)
#     if x_df is None:
#         x_df = pd.DataFrame(np.array(model_df['x']), columns=[loc], index=pd.DatetimeIndex(model_df['t']))
#     else:
#         x_series = pd.DataFrame(np.array(model_df['x']), columns=[loc], index=pd.DatetimeIndex(model_df['t']))
#         x_df = x_df.join(x_series, how='outer')
#     i += 1
# x_df.to_csv(X_DF_NAME)

x_df = pd.read_csv(X_DF_NAME, index_col=[0])
x_df.index = index=pd.DatetimeIndex(x_df.index)

df_lower_q = np.nanquantile(x_df, DF_QUANTILES[0])
df_upper_q = np.nanquantile(x_df, DF_QUANTILES[1])
extreme_lower_by_df = (x_df < df_lower_q).fillna(0).astype(bool)
extreme_upper_by_df = (x_df > df_upper_q).fillna(0).astype(bool)

def cq_extremes(df_slice, quantile):
    col_extremes = df_slice * 0
    for c in range(df_slice.shape[1]):
        if quantile > 0.5:
            col_extremes.iloc[:, c][df_slice.iloc[:, c] >
                np.nanquantile(df_slice.iloc[:, c], quantile)] = 1
        else:
            col_extremes.iloc[:, c][df_slice.iloc[:, c] <
                np.nanquantile(df_slice.iloc[:, c], quantile)] = 1
    return col_extremes.fillna(0).astype(bool)
extreme_lower_by_loc = cq_extremes(x_df, LOC_QUANTILES[0])
extreme_upper_by_loc = cq_extremes(x_df, LOC_QUANTILES[1])

extreme_lower_by_month = x_df.copy()
extreme_upper_by_month = x_df.copy()
for month in range(1, 13):
    x_m = x_df.loc[[dt.month == month for dt in x_df.index], :]
    month_lower_q = np.nanquantile(x_m, MONTH_QUANTILES[0])
    month_upper_q = np.nanquantile(x_m, MONTH_QUANTILES[1])
    extreme_lower_by_month.loc[[dt.month == month for dt in x_df.index], :] = \
        (x_df < month_lower_q).fillna(0).astype(bool)
    extreme_upper_by_month.loc[[dt.month == month for dt in x_df.index], :] = \
        (x_df > month_upper_q).fillna(0).astype(bool)

print('extreme_lower_by_df', extreme_lower_by_df.sum().sum())
print('extreme_lower_by_loc', extreme_lower_by_loc.sum().sum())
print('extreme_lower_by_month', extreme_lower_by_month.sum().sum())
print('extreme_upper_by_df', extreme_upper_by_df.sum().sum())
print('extreme_upper_by_loc', extreme_upper_by_loc.sum().sum())
print('extreme_upper_by_month', extreme_upper_by_month.sum().sum())
extreme_lower = extreme_lower_by_df & extreme_lower_by_loc & extreme_lower_by_month
extreme_upper = extreme_upper_by_df & extreme_upper_by_loc & extreme_upper_by_month
print('extreme_lower combined', extreme_lower.sum().sum())
print('extreme_upper combined', extreme_upper.sum().sum())

extreme_lower_counts = extreme_lower.sum(axis=0)
extreme_upper_counts = extreme_upper.sum(axis=0)
extreme_lower_quantiles = extreme_lower_counts / x_df.count(axis=0)
extreme_upper_quantiles = 1 - extreme_upper_counts / x_df.count(axis=0)

suffix = 'nsrp' if NSRP_DATASET else 'orig'
extreme_lower_quantiles.to_csv(f'x_lower_quantiles_{suffix}.csv')
extreme_upper_quantiles.to_csv(f'x_upper_quantiles_{suffix}.csv')

figure, axes = plt.subplots(1, 2)
axes = iter(axes.flatten())
axis = next(axes)
_map = get_map(axis)
lats, lons = list(zip(*[ast.literal_eval(c) for c in x_df.columns]))
mx, my = _map(lons, lats)
scatter_map(axis, mx, my, extreme_lower_counts,
    cb_min=0, cb_max=extreme_lower_counts.max())
axis.set_title(f'extreme low (df q = {DF_QUANTILES[0]}, '
    f'month q = {MONTH_QUANTILES[0]}, loc q = {LOC_QUANTILES[0]})')

axis = next(axes)
_map = get_map(axis)
scatter_map(axis, mx, my, extreme_upper_counts,
    cb_min=0, cb_max=extreme_upper_counts.max())
axis.set_title(f'extreme high (df q = {DF_QUANTILES[1]}, '
    f'month q = {MONTH_QUANTILES[1]}, loc q = {LOC_QUANTILES[1]})')
plt.show()