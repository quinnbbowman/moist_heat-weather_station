# moist_heat-weather_station
Code necessary for processing weather station and reanalysis stability and moist heat

Data processing requires all weather station netcdf files from hadISD in a single directory [].
Reanalysis fields (in base .25x.25 degree resolution netcdf files) should be called:
a: (which includes these fields)
b(which includes these fields) etc


To process data, scripts should be run in  the following order:

1: combine_weatherstations.py, which combines all weather stations into a single netcdf file.

2: ERA5 data preprocessing [], which generates surface MSE and wet bulb fields, and free tropospheric saturation deficit and saturation MSE:

3: calculate_monthly_maxes, which collects the time and space locations and MSE and wet-bulb temperatures in weather stations

4: interpolate_reanalysis, which interpolates the weather station locations onto reanalysis fields as necessary to calculate stability.

5: generate figures, which does the final data processing for plot generation (binning processes), and generates the plots.

