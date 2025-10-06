import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import xarray as xr
import pandas as pd
import scipy.stats as stats
import copy
import math
from scipy import signal
from itertools import cycle, islice
from scipy.interpolate import RegularGridInterpolator as RGI


#getting the indices to interpolate onto from reanalysis
indays_monmax = np.load('monthmsemaxind[mon,stat,[time,lon,lat].npy')
times = indays_monmax[:,:,0]
lons = indays_monmax[:,:,1]
lats = indays_monmax[:,:,2]
#flattening, removing nans for interpolation
times_flat = times.flatten()
lons_flat = lons.flatten()
lats_flat = lats.flatten()

cleantimes = times_flat[~np.isnan(times_flat)]
cleanlons = lons_flat[~np.isnan(lons_flat)]
cleanlats = lats_flat[~np.isnan(lats_flat)]

#making a list of all the locations (time and space) to interpolate
interpinputs = []
for i in range(len(cleantimes)):
	if not np.isnan(cleantimes[i]):
		interpinputs +=[[cleanlons[i],cleanlats[i],cleantimes[i]]]#location of monthly maximums to interpolate

interpinputszmeans = []
for i in range(len(cleantimes)):
	if not np.isnan(cleantimes[i]):
		interpinputszmeans += [[cleanlats[i],cleantimes[i]]] #location of monthly maximums to interpolate for zonal mean satmse

del times, lons,lats,times_flat,lons_flat,lats_flat,cleantimes,cleanlons,cleanlats
timeseriesfullD = np.arange(np.datetime64('1931-01'), np.datetime64('2021-06'), np.timedelta64(1, "D"))


#interpolating zonal mean 500hpa satmse

#get ERA5 gridded data
era5mse500 = xr.open_dataset('satmse_500,40-22.nc',chunks={'time':100})
era5mse500 = era5mse500.reindex({'time':timeseriesfullD}) #reindex to ensure times are consistent
msezmean = era5mse500.mean(dim='lon') #zonal mean
ys = msezmean.lat #define regular grid indices
ts = np.array(msezmean.time,dtype='datetime64[ns]')
ts = np.int64(ts)
e5zmses = msezmean.satmse
del era5mse500
satmseRGIzmean = RGI((ys,ts), e5zmses.T) #makes interopolator object from the ERA5 field
del ys,ts,e5zmses
interpzmeansatmses = []
i = 0
for setin in interpinputszmeans:
	interpzmeansatmses += [float(satmseRGIzmean(setin))] #interpolates month maximum locations onto the ERA5
	print('zmean MSE',i)
	i += 1
np.save('monmaxzmean500satmse.npy',interpzmeansatmses)
del interpzmeansatmses,satmseRGIzmean,interpinputszmeans


#interpreting local 500hpa satmse
era5mse500 = xr.open_dataset('satmse_500,40-22.nc',chunks={'time':100})
era5mse500 = era5mse500.reindex({'time':timeseriesfullD})
xs = era5mse500.lon
ys = era5mse500.lat
ts = np.array(era5mse500.time,dtype='datetime64[ns]')
ts = np.int64(ts)
e5mses = era5mse500.satmse
del era5mse500
satmseRGI = RGI((xs,ys,ts), e5mses.T)
del xs,ys,ts,e5mses
interpsatmses = []
i = 0
for setin in interpinputs:
	interpsatmses += [float(satmseRGI(setin))]
	print('local FT MSE',i)
	i += 1
np.save('monmax500satmse.npy',interpsatmses)
del interpsatmses,satmseRGI


#interpreting local 850hpa satdef
era5satdef = xr.open_dataset('satdef850,40-22.nc',chunks={'time':100})
era5satdef = era5satdef.reindex({'time':timeseriesfullD})
xs = era5satdef.lon
ys = era5satdef.lat
ts = np.array(era5satdef.time,dtype='datetime64[ns]')
ts = np.int64(ts)
e5satdefs = era5satdef.satdef
del era5satdef
satdefRGI = RGI((xs,ys,ts), e5satdefs.T)
del xs,ys,ts,e5satdefs
interpsatdefs = []
i = 0
for setin in interpinputs:
	interpsatdefs += [float(satdefRGI(setin))]
	print('local LFT satdef',i)
	i += 1
np.save('monmax850satdef.npy',interpsatdefs)
del interpsatdefs,satdefRGI


#interpolating surface MSE
era5surfmse = xr.open_dataset('surfmse,40-22.nc',chunks={'time':100})
era5surfmse = era5surfmse.reindex({'time':timeseriesfullD})
xs = era5surfmse.lon
ys = era5surfmse.lat
ts = np.array(era5surfmse.time,dtype='datetime64[ns]')
ts = np.int64(ts)
e5surfmse = era5surfmse.mse
del era5surfmse
surfmseRGI = RGI((xs,ys,ts), e5surfmse.T)
del xs,ys,ts,e5surfmse
interpsurfmse = []
i = 0
for setin in interpinputs:
	interpsurfmse += [float(surfmseRGI(setin))]
	i += 1
np.save('monmaxsurfmse_era5.npy',interpsurfmse)


#interpolating surface WBT
era5surfwbt = xr.open_dataset('surfwbt,40-22.nc',chunks={'time':100})
era5surfwbt = era5surfwbt.reindex({'time':timeseriesfullD})
xs = era5surfwbt.lon
ys = era5surfwbt.lat
ts = np.array(era5surfwbt.time,dtype='datetime64[ns]')
ts = np.int64(ts)
e5surfwbt = era5surfwbt.wbt
del era5surfwbt
surfwbtRGI = RGI((xs,ys,ts), e5surfwbt.T)
del xs,ys,ts,e5surfwbt
interpsurfwbt = []
i = 0
for setin in interpinputs:
	interpsurfwbt += [float(surfwbtRGI(setin))]
	i += 1
np.save('monmaxsurfwbt_era5.npy',interpsurfwbt)