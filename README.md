## Guide for running code in this repository

### Raw precipitation data analysis

For plots and statistical analysis of raw precipitation data, see raw_plots.py in the `slid-research` repository.

Perform k-means clustering on raw precipitation data, varying the number of clusters, and plot (some of the) results. Do this for each decade, for each (month, decade) or for each individual timestamp. Optionally use a constrained k-means algorithm to limit the size of a given cluster.
```sh
python k_means.py --output_folder outputs/kmeans --decadal
python k_means.py --output_folder outputs/kmeans --decadal --constrain 0.3
python k_means.py --output_folder outputs/kmeans --monthly
python k_means.py --output_folder outputs/kmeans --monthly --constrain 0.3
python k_means.py --output_folder outputs/kmeans
python k_means.py --output_folder outputs/kmeans --constrain 0.3
```

Plot the average precipitation for each month at several key locations in Australia:
```sh
python month_dots.py
```

Plot the evolution over time of precipitation for each month at a given (lat, lon) location (defaulting to Sydney) or averaged across all locations (with `all_locations` flag):
```sh
python month_plots.py
python month_plots.py --all_locations
```

### Complex networks analysis of precipitation data

Construct networks from precipitation data based on the following procedure:
- First, prepare a dataframe from raw precipitation values by assigning a sequence of the previous `avg_lookback_months + lag_months` at a given location for a given timestamp.
  - If `month` is specified, only consider precipitation values (and hence build networks) for that given month.
  - If `deseasonalise` is specified, for each location, subtract the mean of precipitation values for the given month and add the mean of all precipitation values.
  - If `exp_kernel` is specified, the sequence is weighted by an exponential kernel to emphasise values closer to the given timestamp.
- Calculate link strength matrix for every possible location pair based on Pearson correlation between the last `avg_lookback_months` values at the given timestamp.
  - If `lag_months` is specified, do this many times by lagging one series by 1 up to `lag_months` and correlating with the other (unlagged) series, then calculating the `(max - mean) / SD` of all resulting correlations.
  - If `no_anti_corr` is specified, reject any negative correlations. Otherwise, consider the absolute value of correlations.
- Use `edge_density` or `link_str_threshold` to define an adjanency matrix based on the link strength matrix, then construct a `networkx` graph object based on this adjacency matrix.
The precipitation dataframe and link strength matrices from the resulting graphs are saved to CSV files.
```sh
python link_strength_corr.py
```

Plot these networks for the last few years:
```sh
python plot_networks.py --output_folder outputs --start_year 2018
python plot_networks.py --output_folder outputs --start_year 2019
python plot_networks.py --output_folder outputs --start_year 2020
python plot_networks.py --output_folder outputs --start_year 2021
```

Calculate the following metrics for graphs built from link strength matrices:
- Average degree
- Transitivity
- Eccentricity
- Average shortest path
- Eigenvector, betweenness and closeness centralities
- Louvain, greedy modularity and asyn_lpa partitions of the largest connected subgraph
- TODO: Global average link distance
Save the metrics to Pickle files.
```sh
python metrics.py
```

Repeat but for networks constructed only with values from July and with a larger lookback window:
```sh
python link_strength_corr.py --alm 120 --month 7
python plot_networks.py --output_folder outputs --link_str_file_tag corr_alm_120 --start_year 2010 --month 7
python metrics.py --link_str_file_tag corr_alm_120 --month 7
```

Instead build link strength matrices using event synchronisation and event coincidence analysis:
```sh
python link_strength_es.py --method sync
python link_strength_es.py --method ca
```

Plot location-dependent network metrics on a map:
```sh
python metrics_map.py --output_folder outputs/metric_maps
```

Plot time evolution of location-independent and averages of location-dependent network metrics as line plots:
```sh
python metrics_series.py
```

Categorise locations in the dataset based on values of their location-dependent network metrics, rather than raw precipitation values.
```sh
python metrics_k_means.py
```

Calculate and plot partitions/communities of graphs yielded by link strength matrices using the following algorithms (see `networkx` documentation for details):
- Louvain (`lv_partitions`)
- Greedy modularity (`gm_partitions`)
- Fluid communities (`af_partitions`; a number of communities must be specified)
- Asynchronous label propagation (`al_partitions`)
```sh
python communities.py
```

### Sea surface temperature data

The SST dataset being analysed can be obtained from https://cds.climate.copernicus.eu/cdsapp#!/dataset/satellite-sea-surface-temperature?tab=form. This dataset is of daily granularity from 1981 to 2022 and is of 0.05° x 0.05° spatial resolution.

Read in raw SST files and pre-process into a new dataframe, with temporal aggregation via monthly averages and spatial downsampling by keeping only points on a 0.125° x 0.125° grid.
```sh
python sst_preprocessing.py --calculate_loc_filter
```

Plot the pre-processed values for the whole world for the most recent value as well as over Australia for the last year:
```sh
python sst_plots.py --output_folder outputs
```