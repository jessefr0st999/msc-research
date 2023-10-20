@REM Lismore 2022 flood
@REM python311 unami_2009_window.py --dataset fused_daily_nsrp --year_cycles 20 --lat -28.75 --lon 153.5 --start_date "2014-01-01" --end_date "2021-12-01" --target_date "2022-02-01"
@REM python311 unami_2009_window.py --dataset fused_daily_nsrp --year_cycles 20 --lat -28.75 --lon 153.5
python311 unami_2009_window.py --dataset bom_daily --year_cycles 20 --start_date "2014-01-01" --end_date "2021-12-01" --target_date "2022-02-01" --bom_file BOMDaily058070_lismore

@REM NSW 2021 flood (Sydney, Taree, Mount Seaview)
@REM python311 unami_2009_window.py --dataset fused_daily_nsrp --year_cycles 20 --lat -33.25 --lon 150.5 --start_date "2013-01-01" --end_date "2021-01-01" --target_date "2021-03-01"
@REM python311 unami_2009_window.py --dataset fused_daily_nsrp --year_cycles 20 --lat -31.75 --lon 152.5 --start_date "2013-01-01" --end_date "2021-01-01" --target_date "2021-03-01"
@REM python311 unami_2009_window.py --dataset fused_daily_nsrp --year_cycles 20 --lat -31.25 --lon 152.5 --start_date "2013-01-01" --end_date "2021-01-01" --target_date "2021-03-01"
@REM python311 unami_2009_window.py --dataset fused_daily_nsrp --year_cycles 20 --lat -33.25 --lon 150.5
python311 unami_2009_window.py --dataset bom_daily --year_cycles 20 --start_date "2013-01-01" --end_date "2021-01-01" --target_date "2021-03-01" --bom_file BOMDaily060125_port_mac

@REM Queensland 2010/11 flood (Brisbane, Rockhampton, Toowoomba, Longreach)
@REM python311 unami_2009_window.py --dataset fused_daily_nsrp --year_cycles 20 --lat -27.75 --lon 152.5 --start_date "2003-01-01" --end_date "2010-10-01" --target_date "2010-12-01"
@REM python311 unami_2009_window.py --dataset fused_daily_nsrp --year_cycles 20 --lat -23.25 --lon 150.5 --start_date "2003-01-01" --end_date "2010-10-01" --target_date "2010-12-01"
@REM python311 unami_2009_window.py --dataset fused_daily_nsrp --year_cycles 20 --lat -27.75 --lon 151.5 --start_date "2003-01-01" --end_date "2010-10-01" --target_date "2010-12-01"
@REM python311 unami_2009_window.py --dataset fused_daily_nsrp --year_cycles 20 --lat -23.25 --lon 144.5 --start_date "2003-01-01" --end_date "2010-10-01" --target_date "2010-12-01"
@REM python311 unami_2009_window.py --dataset fused_daily_nsrp --year_cycles 20 --lat -27.75 --lon 152.5
python311 unami_2009_window.py --dataset bom_daily --year_cycles 20 --start_date "2003-01-01" --end_date "2010-10-01" --target_date "2010-12-01" --bom_file BOMDaily040237_brisbane

@REM Regional Victoria 2010/11 flood (Shepparton, Gippsland, St Arnaud, Swan Hill, Hamilton)
@REM python311 unami_2009_window.py --dataset fused_daily_nsrp --year_cycles 20 --lat -36.25 --lon 145.5 --start_date "2003-01-01" --end_date "2010-11-01" --target_date "2011-01-01"
@REM python311 unami_2009_window.py --dataset fused_daily_nsrp --year_cycles 20 --lat -38.25 --lon 145.5 --start_date "2003-01-01" --end_date "2010-11-01" --target_date "2011-01-01"
@REM python311 unami_2009_window.py --dataset fused_daily_nsrp --year_cycles 20 --lat -36.75 --lon 143.5 --start_date "2003-01-01" --end_date "2010-11-01" --target_date "2011-01-01"
@REM python311 unami_2009_window.py --dataset fused_daily_nsrp --year_cycles 20 --lat -35.25 --lon 143.5 --start_date "2003-01-01" --end_date "2010-11-01" --target_date "2011-01-01"
@REM python311 unami_2009_window.py --dataset fused_daily_nsrp --year_cycles 20 --lat -37.75 --lon 142.5 --start_date "2003-01-01" --end_date "2010-11-01" --target_date "2011-01-01"
@REM python311 unami_2009_window.py --dataset fused_daily_nsrp --year_cycles 20 --lat -36.25 --lon 145.5
@REM python311 unami_2009_window.py --dataset fused_daily_nsrp --year_cycles 20 --lat -38.25 --lon 145.5
python311 unami_2009_window.py --dataset bom_daily --year_cycles 20 --start_date "2003-01-01" --end_date "2010-11-01" --target_date "2011-01-01" --bom_file BOMDaily085299_gippsland

@REM Townsville 2019 flood
@REM python311 unami_2009_window.py --dataset fused_daily_nsrp --year_cycles 20 --lat -19.25 --lon 146.5 --start_date "2011-01-01" --end_date "2018-11-01" --target_date "2019-01-01"
@REM python311 unami_2009_window.py --dataset fused_daily_nsrp --year_cycles 20 --lat -19.25 --lon 146.5

@REM Townsville 2017 flood (Cyclone Debbie)
@REM python311 unami_2009_window.py --dataset fused_daily_nsrp --year_cycles 20 --lat -19.25 --lon 146.5 --start_date "2009-01-01" --end_date "2016-12-01" --target_date "2017-02-01"

@REM Melbourne 2000s drought
@REM python311 unami_2009_window.py --dataset fused_daily_nsrp --year_cycles 20 --lat -37.75 --lon 145.5

@REM NSW 2017-2019 drought and 2019/20 bushfires (Taree, Tamworth, Dubbo)
@REM python311 unami_2009_window.py --dataset fused_daily_nsrp --year_cycles 20 --lat -31.75 --lon 152.5
@REM python311 unami_2009_window.py --dataset fused_daily_nsrp --year_cycles 20 --lat -31.25 --lon 150.5
@REM python311 unami_2009_window.py --dataset fused_daily_nsrp --year_cycles 20 --lat -32.25 --lon 148.5
