python build_networks.py --save_precipitation --save_links --avg_lookback_months 48 --lag_months 12 --edge_density 0.015

python network_cat_test.py --link_str_file link_str_alm_48_lag_12_2006_01.pkl --output_folder images/video --edge_density 0.015
python network_cat_test.py --link_str_file link_str_alm_48_lag_12_2007_01.pkl --output_folder images/video --edge_density 0.015
python network_cat_test.py --link_str_file link_str_alm_48_lag_12_2008_01.pkl --output_folder images/video --edge_density 0.015
python network_cat_test.py --link_str_file link_str_alm_48_lag_12_2009_01.pkl --output_folder images/video --edge_density 0.015
python network_cat_test.py --link_str_file link_str_alm_48_lag_12_2010_01.pkl --output_folder images/video --edge_density 0.015
python network_cat_test.py --link_str_file link_str_alm_48_lag_12_2011_01.pkl --output_folder images/video --edge_density 0.015
python network_cat_test.py --link_str_file link_str_alm_48_lag_12_2012_01.pkl --output_folder images/video --edge_density 0.015
python network_cat_test.py --link_str_file link_str_alm_48_lag_12_2013_01.pkl --output_folder images/video --edge_density 0.015
python network_cat_test.py --link_str_file link_str_alm_48_lag_12_2014_01.pkl --output_folder images/video --edge_density 0.015
python network_cat_test.py --link_str_file link_str_alm_48_lag_12_2015_01.pkl --output_folder images/video --edge_density 0.015
python network_cat_test.py --link_str_file link_str_alm_48_lag_12_2016_01.pkl --output_folder images/video --edge_density 0.015
python network_cat_test.py --link_str_file link_str_alm_48_lag_12_2017_01.pkl --output_folder images/video --edge_density 0.015
python network_cat_test.py --link_str_file link_str_alm_48_lag_12_2018_01.pkl --output_folder images/video --edge_density 0.015
python network_cat_test.py --link_str_file link_str_alm_48_lag_12_2019_01.pkl --output_folder images/video --edge_density 0.015
python network_cat_test.py --link_str_file link_str_alm_48_lag_12_2020_01.pkl --output_folder images/video --edge_density 0.015
python network_cat_test.py --link_str_file link_str_alm_48_lag_12_2021_01.pkl --output_folder images/video --edge_density 0.015

python build_video.py --target_folder images/video --video_name communities_ed_0p015_alm_48_lag_12

move images\video\communities_ed_0p015_alm_48_lag_12.mp4 images\communities_ed_0p015_alm_48_lag_12.mp4

del /S /Q images\video\*

@REM python metrics_plot_test.py --save_fig --metrics_file spatial_metrics_0_alm_48_lag_12_ed_0p005.pkl