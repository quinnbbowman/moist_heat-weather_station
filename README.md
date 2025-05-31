# moist_heat-weather_station
Code necessary for processing weather station and reanalysis stability and moist heat\

Data processing requires all weather station netcdf files from hadISD is 2 different directories.\
hadisd weather station data is saved in 2 files per station, temperature/surface data and humidity data.\
These should be saved in 2 different directories, humdity in '/WMOHUM' and others in '/WMODATA'.  These should only include tropical weather stations (Between 20&deg; S and &deg; N), and they should use default names as downloaded.

Reanalysis fields are also required\
Surface reanalysis fields (in base .25&deg;x.25&deg; resolution netcdf files) are input in ~20 year files per field, due to download constraints, from 1940 through 2022, from 20&deg; S to 20 &deg;N, across tropics

from 1940 through 1961:\
'2mT,40-61.nc' 2 meter temperature (kelvin)\
'2md_t,40-61.nc' 2 meter dewpoint (kelvin)\
'2msurf_p,40-61.nc' 2 meter surface pressure (pascals)\
'2m_geop,40-61.nc' 2 meter geopotential(m^2/s^2)\


from 1961 through 1981:\
'2mT,61-81.nc' 2 meter temperature (kelvin)\
'2md_t,61-81.nc' 2 meter  dewpoint (kelvin)\
'2msurf_p,61-81.nc' 2 meter surface pressure (pascals)\
'2m_geop,61-81.nc' 2 meter geopotential(m^2/s^2)


from 1981 through 2002:\
'2mT,81-02.nc' 2 meter temperature (kelvin)\
'2md_t,81-02.nc' 2 meter  dewpoint (kelvin)\
'2msurf_p,81-02.nc' 2 meter surface pressure (pascals)\
'2m_geop,81-02.nc' 2 meter geopotential(m^2/s^2)


from 2002 through 2022:\
'2mT,02-22.nc' 2 meter temperature (kelvin)\
'2md_t,02-22.nc' 2 meter  dewpoint (kelvin)\
'2msurf_p,02-22.nc' 2 meter surface pressure (pascals)\
'2m_geop,02-22.nc' 2 meter geopotential(m^2/s^2)


Atmospheric fields are in single files 1940 through 2022, again .25&deg;x.25&deg; resolution from 1940 through 2022, from 20&deg;S to 20&deg;N, across tropics\
'40-22,850hpa_sphum.nc' 850 hpa specific humidity (kg/kg)\
'40-22,850hpa_rhum.nc'  850 hpa relative humidity (percent)\
\
'40-22,500hpa_sphum.nc' 500 hpa specific humidity (kg/kg)\
'40-22,500hpa_rhum.nc' 500 hpa relative humidity (percent)\
'40-22,500hpa_geop.nc' 500 hpa geopotential (m^2/s^2)\
'40-22,500hpa_T.nc' 500 hpa temperature (Kelvin)\

To process data and do analysis, scripts should be run in  the following order:

1: combine_weatherstations.py, which combines all weather stations into a single netcdf file, then daily averages them, using only daily averages per station with only 4 or more measurements, of which the furthest are more than 16 hours apart.
This program includes multithreading, set to use 16 cores.  It saves a netcdf file 'reprocessed_daily_stats_calcWBT.nc' which is a daily mean netcdf file including all relevant data (including processed wet-bulb temperature and moist static energy).
It also makes some intermediate files, which are no longer necessary and may be deleted.

2: reanalysis_processing.py which generates reanalysis surface MSE and wet bulb fields, and free tropospheric saturation deficit and saturation MSE.  This is also multithreaded

3: calculate_monthly_maxes.py, which collects the time and space locations and MSE and wet-bulb temperatures in weather stations on the monthly maximum days.  This is also multithreaded

4: interpolate_reanalysis.py, which interpolates the weather station locations onto reanalysis fields as necessary to calculate stability.

5: generate_plots.py, which does the final data processing for plot generation (binning processes), and generates the plots.

