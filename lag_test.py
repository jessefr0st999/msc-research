import numpy as np
from math import pi
import matplotlib.pyplot as plt

lag_months = 3
t = np.linspace(0, 2 * pi, 16 + 1)
x = [
    np.sin(1 * t),
    np.sin(2 * t),
]
y = [
    np.sin(2 * (t - 0)), # same as x[1]
    np.sin(2 * (t - 1 * pi / 8)), # x[1] shifted forward by 1
    np.sin(2 * (t - 2 * pi / 8)), # x[1] shifted forward by 2
    np.sin(2 * (t + 1 * pi / 8)), # x[1] shifted backward by 1
]
n = len(x)
m = len(y)

link_str_array = np.zeros((m, n, 1 + 2 * lag_months))
for i, lag in enumerate(range(-1, -1 - lag_months, -1)):
    unlagged_noaa = [list(l[lag_months :]) for l in np.array(x)]
    unlagged_prec = [list(l[lag_months :]) for l in np.array(y)]
    lagged_noaa = [list(l[lag + lag_months : lag]) for l in np.array(x)]
    lagged_prec = [list(l[lag + lag_months : lag]) for l in np.array(y)]
    combined = [*unlagged_noaa, *lagged_noaa, *unlagged_prec, *lagged_prec]
    cov_mat = np.abs(np.corrcoef(combined))
    # Get the correlations between the unlagged series at both locations
    if i == 0:
        link_str_array[:, :, lag_months] = cov_mat[2*n : 2*n + m, 0 : n]
    # Positive: between lagged x series and unlagged y series
    link_str_array[:, :, lag_months + i + 1] = cov_mat[2*n + m : 2*n + 2*m, 0 : n]
    # Negative: between unlagged x series and lagged y series
    link_str_array[:, :, lag_months - i - 1] = cov_mat[2*n : 2*n + m, n : 2*n]
link_str_array_agg = np.apply_along_axis(np.max, 2, link_str_array)
link_str_max_lags = np.argmax(link_str_array, 2) - lag_months

print(link_str_array)
print(link_str_array_agg)
print(link_str_max_lags)

# expected link_str_max_lags:
# x0, y0: ?
# x0, y1: ?
# x0, y2: ?
# x0, y3: ?
# x1, y0: 0
# x1, y1: -1
# x1, y2: -2
# x1, y3: +1

figure, axes = plt.subplots(2, 2, layout='compressed')
axes = iter(axes.flatten())
for i in range(4):
    axis = next(axes)
    axis.plot(t, y[i], 'r')
    axis.plot(t, x[0], 'g')
    axis.plot(t, x[1], 'b')
plt.show()