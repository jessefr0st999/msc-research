import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
np.random.seed(0)

index = pd.date_range(start='2018-01-01', end='2022-12-31', freq='D')

t_vec = np.array([(t - index.values[0]) / np.timedelta64(1, 'D') \
    for t in index.values])
# test_series = pd.Series(10 + 10 * np.cos(2 * np.pi / 365.25 * t_vec), index=index)
test_series = pd.Series(2 + 2 * np.cos(2 * np.pi / 365.25 * t_vec), index=index)
test_series += np.random.normal(0, 1, len(test_series))
# test_series[330 : 380] += np.abs(np.random.normal(5, 3, 50))
# test_series[830 : 1130] -= np.abs(np.random.normal(3, 3, 300))
test_series[test_series < 0] = 0
test_series.to_csv('data_unfused/test_data.csv', header=False)
print(test_series)

figure, axis = plt.subplots(1)
axis.plot(test_series, 'ro-')
plt.show()