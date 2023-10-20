@REM analysis of raw data
@REM python311 raw_plots.py --output_folder outputs/prec
@REM python311 k_means.py --decadal --output_folder outputs/prec
@REM python311 k_means.py --monthly --output_folder outputs/prec
@REM python311 k_means.py --monthly_averages --output_folder outputs/prec
@REM python311 edq.py --output_folder outputs/prec

@REM DWS, pearson
@REM python311 links.py --decadal
@REM python311 plot_networks.py --link_str_file_tag corr_decadal_lag_0 --output_folder outputs/prec
@REM python311 communities.py --link_str_file link_str_corr_decadal_lag_0_d1.csv --min 4 --max 6 --output_folder outputs/prec
@REM python311 communities.py --link_str_file link_str_corr_decadal_lag_0_d2.csv --min 4 --max 6 --output_folder outputs/prec
@REM python311 communities_by_number.py --link_str_file link_str_corr_decadal_lag_0_d1.csv --output_folder outputs/prec
@REM python311 communities_by_number.py --link_str_file link_str_corr_decadal_lag_0_d2.csv --output_folder outputs/prec
@REM python311 metrics.py --link_str_file_tag corr_decadal_lag_0 --output_folder data/metrics
@REM python311 metrics_map.py --metrics_file_base metrics_corr_decadal_lag_0_ed_0p005 --common_scales --output_folder outputs/prec

@REM DWS, spearman
@REM python311 links.py --decadal --method spearman
@REM python311 plot_networks.py --link_str_file_tag corr_decadal_spearman_lag_0 --output_folder outputs/prec
@REM python311 communities.py --link_str_file link_str_corr_decadal_spearman_lag_0_d1.csv --min 5 --max 7 --output_folder outputs/prec
@REM python311 communities.py --link_str_file link_str_corr_decadal_spearman_lag_0_d2.csv --min 5 --max 7 --output_folder outputs/prec
@REM python311 communities_by_number.py --link_str_file link_str_corr_decadal_spearman_lag_0_d1.csv --output_folder outputs/prec
@REM python311 communities_by_number.py --link_str_file link_str_corr_decadal_spearman_lag_0_d2.csv --output_folder outputs/prec
@REM python311 metrics.py --link_str_file_tag corr_decadal_spearman_lag_0 --output_folder data/metrics
@REM python311 metrics_map.py --metrics_file_base metrics_corr_decadal_spearman_lag_0_ed_0p005 --common_scales --output_folder outputs/prec

@REM DWS, regression gradient
@REM python311 links.py --decadal --method reg
@REM python311 plot_networks.py --link_str_file_tag corr_decadal_reg_lag_0 --output_folder outputs/prec
@REM python311 communities.py --link_str_file link_str_corr_decadal_reg_lag_0_d1.csv --output_folder outputs/prec
@REM python311 communities.py --link_str_file link_str_corr_decadal_reg_lag_0_d2.csv --output_folder outputs/prec
@REM python311 communities_by_number.py --link_str_file link_str_corr_decadal_reg_lag_0_d1.csv --output_folder outputs/prec
@REM python311 communities_by_number.py --link_str_file link_str_corr_decadal_reg_lag_0_d2.csv --output_folder outputs/prec
@REM python311 metrics.py --link_str_file_tag corr_decadal_reg_lag_0 --output_folder data/metrics
@REM python311 metrics_map.py --metrics_file_base metrics_corr_decadal_reg_lag_0_ed_0p005 --common_scales --output_folder outputs/prec

@REM DAMS, pearson
@REM python311 links.py --dms
python311 plot_networks.py --link_str_file_tag corr_dms_lag_0 --output_folder networks_outputs/prec_thesis
@REM python311 communities.py --link_str_file link_str_corr_dms_lag_0_d1.csv --output_folder networks_outputs/prec_thesis
@REM python311 communities.py --link_str_file link_str_corr_dms_lag_0_d2.csv --output_folder networks_outputs/prec_thesis
@REM python311 communities_by_number.py --link_str_file link_str_corr_dms_lag_0_d1.csv --output_folder networks_outputs/prec_thesis
@REM python311 communities_by_number.py --link_str_file link_str_corr_dms_lag_0_d2.csv --output_folder networks_outputs/prec_thesis
@REM python311 metrics.py --link_str_file_tag corr_dms_lag_0 --output_folder data/metrics
@REM python311 metrics_map.py --metrics_file_base metrics_corr_dms_lag_0_ed_0p005 --common_scales --output_folder networks_outputs/prec_thesis

@REM DAMS, spearman
@REM python311 links.py --dms --method spearman
python311 plot_networks.py --link_str_file_tag corr_dms_spearman_lag_0 --output_folder networks_outputs/prec_thesis
@REM python311 communities.py --link_str_file link_str_corr_dms_spearman_lag_0_d1.csv --output_folder networks_outputs/prec_thesis
@REM python311 communities.py --link_str_file link_str_corr_dms_spearman_lag_0_d2.csv --output_folder networks_outputs/prec_thesis
@REM python311 communities_by_number.py --link_str_file link_str_corr_dms_spearman_lag_0_d1.csv --output_folder networks_outputs/prec_thesis
@REM python311 communities_by_number.py --link_str_file link_str_corr_dms_spearman_lag_0_d2.csv --output_folder networks_outputs/prec_thesis
@REM python311 metrics.py --link_str_file_tag corr_dms_spearman_lag_0 --output_folder data/metrics
@REM python311 metrics_map.py --metrics_file_base metrics_corr_dms_spearman_lag_0_ed_0p005 --common_scales --output_folder networks_outputs/prec_thesis

@REM DAMS, regression gradient
@REM python311 links.py --dms --method reg
@REM python311 plot_networks.py --link_str_file_tag corr_dms_reg_lag_0 --output_folder outputs/prec
@REM python311 communities.py --link_str_file link_str_corr_dms_reg_lag_0_d1.csv --output_folder outputs/prec
@REM python311 communities.py --link_str_file link_str_corr_dms_reg_lag_0_d2.csv --output_folder outputs/prec
@REM python311 communities_by_number.py --link_str_file link_str_corr_dms_reg_lag_0_d1.csv --output_folder outputs/prec
@REM python311 communities_by_number.py --link_str_file link_str_corr_dms_reg_lag_0_d2.csv --output_folder outputs/prec
@REM python311 metrics.py --link_str_file_tag corr_dms_reg_lag_0 --output_folder data/metrics
@REM python311 metrics_map.py --metrics_file_base metrics_corr_dms_reg_lag_0_ed_0p005 --common_scales --output_folder outputs/prec

@REM DSMS, pearson
@REM python311 links.py --dms --season summer
@REM python311 links.py --dms --season autumn
@REM python311 links.py --dms --season winter
@REM python311 links.py --dms --season spring
python311 plot_networks.py --link_str_file_tag corr_dms_summer_lag_0 --output_folder networks_outputs/prec_thesis
python311 plot_networks.py --link_str_file_tag corr_dms_autumn_lag_0 --output_folder networks_outputs/prec_thesis
python311 plot_networks.py --link_str_file_tag corr_dms_winter_lag_0 --output_folder networks_outputs/prec_thesis
python311 plot_networks.py --link_str_file_tag corr_dms_spring_lag_0 --output_folder networks_outputs/prec_thesis
@REM python311 metrics.py --link_str_file_tag corr_dms_summer_lag_0 --output_folder data/metrics
@REM python311 metrics.py --link_str_file_tag corr_dms_autumn_lag_0 --output_folder data/metrics
@REM python311 metrics.py --link_str_file_tag corr_dms_winter_lag_0 --output_folder data/metrics
@REM python311 metrics.py --link_str_file_tag corr_dms_spring_lag_0 --output_folder data/metrics
@REM python311 metrics_map.py --metrics_file_base metrics_corr_dms_summer_lag_0_ed_0p005 --common_scales --output_folder networks_outputs/prec_thesis
@REM python311 metrics_map.py --metrics_file_base metrics_corr_dms_autumn_lag_0_ed_0p005 --common_scales --output_folder networks_outputs/prec_thesis
@REM python311 metrics_map.py --metrics_file_base metrics_corr_dms_winter_lag_0_ed_0p005 --common_scales --output_folder networks_outputs/prec_thesis
@REM python311 metrics_map.py --metrics_file_base metrics_corr_dms_spring_lag_0_ed_0p005 --common_scales --output_folder networks_outputs/prec_thesis

@REM DSMS, spearman
@REM python311 links.py --dms --season summer --method spearman
@REM python311 links.py --dms --season autumn --method spearman
@REM python311 links.py --dms --season winter --method spearman
@REM python311 links.py --dms --season spring --method spearman
python311 plot_networks.py --link_str_file_tag corr_dms_summer_spearman_lag_0 --output_folder networks_outputs/prec_thesis
python311 plot_networks.py --link_str_file_tag corr_dms_autumn_spearman_lag_0 --output_folder networks_outputs/prec_thesis
python311 plot_networks.py --link_str_file_tag corr_dms_winter_spearman_lag_0 --output_folder networks_outputs/prec_thesis
python311 plot_networks.py --link_str_file_tag corr_dms_spring_spearman_lag_0 --output_folder networks_outputs/prec_thesis
@REM python311 metrics.py --link_str_file_tag corr_dms_summer_spearman_lag_0 --output_folder data/metrics
@REM python311 metrics.py --link_str_file_tag corr_dms_autumn_spearman_lag_0 --output_folder data/metrics
@REM python311 metrics.py --link_str_file_tag corr_dms_winter_spearman_lag_0 --output_folder data/metrics
@REM python311 metrics.py --link_str_file_tag corr_dms_spring_spearman_lag_0 --output_folder data/metrics
@REM python311 metrics_map.py --metrics_file_base metrics_corr_dms_summer_spearman_lag_0_ed_0p005 --common_scales --output_folder networks_outputs/prec_thesis
@REM python311 metrics_map.py --metrics_file_base metrics_corr_dms_autumn_spearman_lag_0_ed_0p005 --common_scales --output_folder networks_outputs/prec_thesis
@REM python311 metrics_map.py --metrics_file_base metrics_corr_dms_winter_spearman_lag_0_ed_0p005 --common_scales --output_folder networks_outputs/prec_thesis
@REM python311 metrics_map.py --metrics_file_base metrics_corr_dms_spring_spearman_lag_0_ed_0p005 --common_scales --output_folder networks_outputs/prec_thesis

@REM DSMS, regression gradient
@REM python311 links.py --dms --season summer --method reg
@REM python311 links.py --dms --season autumn --method reg
@REM python311 links.py --dms --season winter --method reg
@REM python311 links.py --dms --season spring --method reg
@REM python311 plot_networks.py --link_str_file_tag corr_dms_summer_reg_lag_0 --output_folder outputs/prec
@REM python311 plot_networks.py --link_str_file_tag corr_dms_autumn_reg_lag_0 --output_folder outputs/prec
@REM python311 plot_networks.py --link_str_file_tag corr_dms_winter_reg_lag_0 --output_folder outputs/prec
@REM python311 plot_networks.py --link_str_file_tag corr_dms_spring_reg_lag_0 --output_folder outputs/prec
@REM python311 metrics.py --link_str_file_tag corr_dms_summer_reg_lag_0 --output_folder data/metrics
@REM python311 metrics.py --link_str_file_tag corr_dms_autumn_reg_lag_0 --output_folder data/metrics
@REM python311 metrics.py --link_str_file_tag corr_dms_winter_reg_lag_0 --output_folder data/metrics
@REM python311 metrics.py --link_str_file_tag corr_dms_spring_reg_lag_0 --output_folder data/metrics
@REM python311 metrics_map.py --metrics_file_base metrics_corr_dms_summer_reg_lag_0_ed_0p005 --common_scales --output_folder outputs/prec
@REM python311 metrics_map.py --metrics_file_base metrics_corr_dms_autumn_reg_lag_0_ed_0p005 --common_scales --output_folder outputs/prec
@REM python311 metrics_map.py --metrics_file_base metrics_corr_dms_winter_reg_lag_0_ed_0p005 --common_scales --output_folder outputs/prec
@REM python311 metrics_map.py --metrics_file_base metrics_corr_dms_spring_reg_lag_0_ed_0p005 --common_scales --output_folder outputs/prec

@REM yearly, pearson, 4 years lookback prior to given year (ED 0.005)
@REM use March as the reference month for the yearly networks
@REM python311 links.py --month 3 --alm 60
@REM python311 plot_networks.py --start_year 2001 --month 3 --link_str_file_tag corr_alm_60_lag_0 --output_folder outputs/prec
@REM python311 plot_networks.py --start_year 2012 --month 3 --link_str_file_tag corr_alm_60_lag_0 --output_folder outputs/prec
@REM python311 communities.py --link_str_file link_str_corr_alm_60_lag_0_2012_03.csv --output_folder outputs/prec
@REM python311 communities.py --link_str_file link_str_corr_alm_60_lag_0_2022_03.csv --output_folder outputs/prec
@REM python311 communities_by_number.py --link_str_file link_str_corr_alm_60_lag_0_2012_03.csv --output_folder outputs/prec
@REM python311 communities_by_number.py --link_str_file link_str_corr_alm_60_lag_0_2022_03.csv --output_folder outputs/prec
@REM python311 metrics.py --link_str_file_tag corr_alm_60_lag_0 --output_folder data/metrics
@REM python311 metrics_map.py --month 3 --metrics_file_base metrics_corr_alm_60_lag_0_ed_0p005 --output_folder outputs/prec
@REM python311 metrics_series.py --metrics_file_base metrics_corr_alm_60_lag_0_ed_0p005 --output_folder outputs/prec

@REM yearly, spearman, 4 years lookback prior to given year (ED 0.005)
@REM python311 links.py --month 3 --alm 60 --method spearman
@REM python311 plot_networks.py --start_year 2001 --month 3 --link_str_file_tag corr_alm_60_spearman_lag_0 --output_folder outputs/prec
@REM python311 plot_networks.py --start_year 2012 --month 3 --link_str_file_tag corr_alm_60_spearman_lag_0 --output_folder outputs/prec
@REM python311 communities.py --link_str_file link_str_corr_alm_60_spearman_lag_0_2012_03.csv --output_folder outputs/prec
@REM python311 communities.py --link_str_file link_str_corr_alm_60_spearman_lag_0_2022_03.csv --output_folder outputs/prec
@REM python311 communities_by_number.py --link_str_file link_str_corr_alm_60_spearman_lag_0_2012_03.csv --output_folder outputs/prec
@REM python311 communities_by_number.py --link_str_file link_str_corr_alm_60_spearman_lag_0_2022_03.csv --output_folder outputs/prec
@REM python311 metrics.py --link_str_file_tag corr_alm_60_spearman_lag_0 --output_folder data/metrics
@REM python311 metrics_map.py --month 3 --metrics_file_base metrics_corr_alm_60_spearman_lag_0_ed_0p005 --output_folder outputs/prec
@REM python311 metrics_series.py --metrics_file_base metrics_corr_alm_60_spearman_lag_0_ed_0p005 --output_folder outputs/prec

@REM yearly, regression gradient, 4 years lookback prior to given year (ED 0.005)
@REM python311 links.py --month 3 --alm 60 --method reg
@REM python311 plot_networks.py --start_year 2001 --month 3 --link_str_file_tag corr_alm_60_reg_lag_0 --output_folder outputs/prec
@REM python311 plot_networks.py --start_year 2012 --month 3 --link_str_file_tag corr_alm_60_reg_lag_0 --output_folder outputs/prec
@REM python311 communities.py --link_str_file link_str_corr_alm_60_reg_lag_0_2012_03.csv --output_folder outputs/prec
@REM python311 communities.py --link_str_file link_str_corr_alm_60_reg_lag_0_2022_03.csv --output_folder outputs/prec
@REM python311 communities_by_number.py --link_str_file link_str_corr_alm_60_reg_lag_0_2012_03.csv --output_folder outputs/prec
@REM python311 communities_by_number.py --link_str_file link_str_corr_alm_60_reg_lag_0_2022_03.csv --output_folder outputs/prec
@REM python311 metrics.py --link_str_file_tag corr_alm_60_reg_lag_0 --output_folder data/metrics
@REM python311 metrics_map.py --month 3 --metrics_file_base metrics_corr_alm_60_reg_lag_0_ed_0p005 --output_folder outputs/prec
@REM python311 metrics_series.py --metrics_file_base metrics_corr_alm_60_reg_lag_0_ed_0p005 --output_folder outputs/prec

@REM yearly (ED 0.005)
@REM python311 plot_networks.py --start_year 2001 --month 3 --link_str_file_tag corr_alm_12_lag_0 --ed 0.005 --output_folder outputs/prec
@REM python311 plot_networks.py --start_year 2012 --month 3 --link_str_file_tag corr_alm_12_lag_0 --ed 0.005 --output_folder outputs/prec

@REM yearly (ED 0.01)
@REM python311 links.py --month 3 --alm 12
@REM python311 plot_networks.py --start_year 2001 --month 3 --link_str_file_tag corr_alm_12_lag_0 --ed 0.01 --output_folder outputs/prec
@REM python311 plot_networks.py --start_year 2012 --month 3 --link_str_file_tag corr_alm_12_lag_0 --ed 0.01 --output_folder outputs/prec
@REM python311 communities.py --link_str_file link_str_corr_alm_12_lag_0_2002_03.csv --ed 0.01 --output_folder outputs/prec
@REM python311 communities.py --link_str_file link_str_corr_alm_12_lag_0_2007_03.csv --ed 0.01 --output_folder outputs/prec
@REM python311 communities.py --link_str_file link_str_corr_alm_12_lag_0_2012_03.csv --ed 0.01 --output_folder outputs/prec
@REM python311 communities.py --link_str_file link_str_corr_alm_12_lag_0_2017_03.csv --ed 0.01 --output_folder outputs/prec
@REM python311 communities.py --link_str_file link_str_corr_alm_12_lag_0_2022_03.csv --ed 0.01 --output_folder outputs/prec
@REM python311 communities_by_number.py --link_str_file link_str_corr_alm_12_lag_0_2002_03.csv --ed 0.01 --output_folder outputs/prec
@REM python311 communities_by_number.py --link_str_file link_str_corr_alm_12_lag_0_2007_03.csv --ed 0.01 --output_folder outputs/prec
@REM python311 communities_by_number.py --link_str_file link_str_corr_alm_12_lag_0_2012_03.csv --ed 0.01 --output_folder outputs/prec
@REM python311 communities_by_number.py --link_str_file link_str_corr_alm_12_lag_0_2017_03.csv --ed 0.01 --output_folder outputs/prec
@REM python311 communities_by_number.py --link_str_file link_str_corr_alm_12_lag_0_2022_03.csv --ed 0.01 --output_folder outputs/prec
@REM python311 metrics.py --ed 0.01 --link_str_file_tag corr_alm_12_lag_0 --output_folder data/metrics
@REM python311 metrics_map.py --month 3 --metrics_file_base metrics_corr_alm_12_lag_0_ed_0p01 --output_folder outputs/prec
@REM python311 metrics_series.py --metrics_file_base metrics_corr_alm_12_lag_0_ed_0p01 --output_folder outputs/prec

@REM lag testing
@REM python311 links.py --decadal --lag 6
@REM python311 links.py --month 3 --alm 12 --lag 6
@REM python311 links.py --month 3 --alm 60 --lag 6
@REM python311 plot_networks.py --link_str_file_tag corr_decadal_lag_6 --output_folder outputs/prec
@REM python311 plot_networks_by_max_lag.py --link_str_file link_str_corr_decadal_lag_6_d1 --output_folder outputs/prec
@REM python311 plot_networks_by_max_lag.py --link_str_file link_str_corr_alm_12_lag_6_2022_03 --output_folder outputs/prec
@REM python311 plot_networks_by_max_lag.py --link_str_file link_str_corr_alm_60_lag_6_2022_03 --output_folder outputs/prec
