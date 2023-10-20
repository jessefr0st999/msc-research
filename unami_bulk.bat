@REM python311 unami_2009_bulk.py --dataset fused_daily_nsrp --np_bcs --delta_s 100 --s_steps 250 --year_cycles 150 --save_folder fused_daily_nsrp_x_0p025_s_100_250_np_yc_150 --solution_tol_exit

@REM python311 unami_2009_bulk.py --dataset fused_daily_nsrp --decade 1 --np_bcs --delta_s 100 --s_steps 250 --year_cycles 150 --save_folder fused_daily_nsrp_x_0p025_s_100_250_np_yc_150_d1 --solution_tol_exit

@REM python311 unami_2009_bulk.py --dataset fused_daily_nsrp --decade 2 --np_bcs --delta_s 100 --s_steps 250 --year_cycles 150 --save_folder fused_daily_nsrp_x_0p025_s_100_250_np_yc_150_d2 --solution_tol_exit

@REM python311 unami_2009_bulk.py --dataset fused_daily_nsrp --np_bcs --delta_s 10 --s_steps 1000 --year_cycles 150 --save_folder fused_daily_nsrp_x_0p025_s_10_1000_np_yc_150 --solution_tol_exit --solution_tol 0.1

@REM python311 unami_2009_bulk.py --dataset fused_daily_nsrp --np_bcs --delta_s 100 --s_steps 250 --year_cycles 150 --sar_corrected --save_folder fused_daily_nsrp_sar_x_0p025_s_100_250_np_yc_150 --solution_tol_exit

@REM python311 unami_2009_bulk_metrics.py --read_folder fused_daily_nsrp_x_0p025_s_100_250_np_yc_150
@REM python311 unami_2009_bulk_metrics.py --read_folder fused_daily_nsrp_x_0p025_s_100_250_np_yc_150_d1
@REM python311 unami_2009_bulk_metrics.py --read_folder fused_daily_nsrp_x_0p025_s_100_250_np_yc_150_d2
@REM python311 unami_2009_bulk_metrics.py --read_folder fused_daily_nsrp_x_0p025_s_10_1000_np_yc_150 --delta_s 10 --s_steps 1000
@REM python311 unami_2009_bulk_metrics.py --read_folder fused_daily_nsrp_x_0p025_s_100_250_np_yc_150 --decadal

@REM python311 unami_2009_bulk_metrics.py --read_folder fused_daily_nsrp_sar_x_0p025_s_100_250_np_yc_150



@REM python311 unami_2009_bulk.py --dataset fused_daily_nsrp --np_bcs --delta_s 100 --s_steps 250 --year_cycles 150 --save_folder fused_daily_nsrp_x_0p025_s_100_250_np_yc_150_wy_5 --solution_tol_exit --window_years 5
@REM python311 unami_2009_bulk_metrics.py --read_folder fused_daily_nsrp_x_0p025_s_100_250_np_yc_150_wy_5

@REM python311 unami_2009_bulk.py --dataset fused_daily_nsrp --np_bcs --delta_s 100 --s_steps 250 --year_cycles 150 --sar_corrected --decade 1 --save_folder fused_daily_nsrp_sar_x_0p025_s_100_250_np_yc_150_d1 --solution_tol_exit
@REM python311 unami_2009_bulk.py --dataset fused_daily_nsrp --np_bcs --delta_s 100 --s_steps 250 --year_cycles 150 --sar_corrected --decade 2 --save_folder fused_daily_nsrp_sar_x_0p025_s_100_250_np_yc_150_d2 --solution_tol_exit
@REM python311 unami_2009_bulk_metrics.py --read_folder fused_daily_nsrp_sar_x_0p025_s_100_250_np_yc_150_d1
@REM python311 unami_2009_bulk_metrics.py --read_folder fused_daily_nsrp_sar_x_0p025_s_100_250_np_yc_150_d2
@REM python311 unami_2009_bulk_metrics.py --read_folder fused_daily_nsrp_sar_x_0p025_s_100_250_np_yc_150 --decadal

python311 unami_2009.py --dataset fused_daily_nsrp --np_bcs --delta_s 100 --s_steps 50 --year_cycles 20 --x_quantile 0.005 --lat -28.75 --lon 135.5 --decade 2 --plot
python311 unami_2009.py --dataset fused_daily_nsrp --np_bcs --delta_s 100 --s_steps 50 --year_cycles 20 --x_quantile 0.005 --lat -29.25 --lon 135.5 --decade 2 --plot
python311 unami_2009.py --dataset fused_daily_nsrp --np_bcs --delta_s 100 --s_steps 50 --year_cycles 20 --x_quantile 0.005 --lat -29.75 --lon 135.5 --decade 2 --plot
python311 unami_2009.py --dataset fused_daily_nsrp --np_bcs --delta_s 100 --s_steps 50 --year_cycles 20 --x_quantile 0.005 --lat -30.25 --lon 135.5 --decade 2 --plot
python311 unami_2009.py --dataset fused_daily_nsrp --np_bcs --delta_s 100 --s_steps 50 --year_cycles 20 --x_quantile 0.005 --lat -30.75 --lon 135.5 --decade 2 --plot --prec_inc 0.05

python311 unami_2009_bulk.py --dataset fused_daily_nsrp --decade 2 --np_bcs --delta_s 100 --s_steps 250 --year_cycles 150 --save_folder fused_daily_nsrp_x_0p005_s_100_250_np_yc_150_d2_pi_0p05 --solution_tol_exit --prec_inc 0.05
python311 unami_2009_bulk_metrics.py --read_folder fused_daily_nsrp_x_0p005_s_100_250_np_yc_150_d2_pi_0p05