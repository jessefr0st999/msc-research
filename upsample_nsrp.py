import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from helpers import prepare_df
from nsrp_calibration import simulate_nsrp

NSRP_PARAMS = (
    lambda x: int(1 + min(x // 15, 4)), # rule for number of storms
    1.5, # average number of cells per storm
    1, # average cell duration in days
    1, # average cell displacement from storm front in days
)

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
    FUSED_SERIES_KEY = (-37.75, 145.5) # Melbourne
    # FUSED_SERIES_KEY = (-28.75, 153.5) # Lismore
    calculate_and_save(FUSED_SERIES_KEY, True)

if __name__ == '__main__':
    main()