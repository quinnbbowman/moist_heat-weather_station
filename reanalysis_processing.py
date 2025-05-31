#code for processing reanalysis fields
import xarray as xr
import multiprocessing as mp
import numpy as np
import os
import time
import pandas as pd
from metpy.calc import wet_bulb_temperature
from metpy.units import units

lv = 2501000 #j/kg
cp = 1005.7 #j/kg/c
eps = .622 # rd/rv

#first, calculate surface wet bulb temperatures across the tropics

def calculate_daily_wbt(P_ar, T_ar, Td_ar):
    P_ar = P_ar.load()
    T_ar = T_ar.load()
    Td_ar = Td_ar.load()

    wetbulbar = wet_bulb_temperature(P_ar.sp.values * units.Pa, T_ar.t2m.values * units.kelvin,
                                     Td_ar.d2m.values * units.kelvin).magnitude

    P_ar = P_ar.close()
    T_ar = T_ar.close()
    Td_ar = Td_ar.close()

    del P_ar, T_ar, Td_ar
    return wetbulbar

def calculate_daily_mse(P_ar, T_ar, Td_ar,Z_ar):
    P_ar = P_ar.load()
    T_ar = T_ar.load()
    Td_ar = Td_ar.load()
    Z_ar = Z_ar.load()

    gz = Z_ar.z
    p = P_ar.sp

    dtC = Td_ar.d2m.values - 273.15

    e = 611.2 * np.e ** ((17.67 * dtC) / (dtC + 243.5)) # vapor pressure pa - Bolton formula
    q = eps * e / (p.values - (1 - eps) * e) # specific humidity kg/kg

    msear = lv*q + cp*T_ar.t2m.values + gz.values

    P_ar = P_ar.close()
    T_ar = T_ar.close()
    Td_ar = Td_ar.close()
    Z_ar  = Z_ar.close()

    del P_ar, T_ar, Td_ar, Z_ar
    return msear


def multithreaded_era5_wbt_mse_calc(yearset):
    #raw MSE files from ERA5 are daily means in roughly 20 year chunks (due to data download restrictions when initially downloaded)
    #calculates wet bulb temperatures for each set separately
    if yearset == 't1':
        Txr = xr.open_dataset('2mT,40-61.nc')
        T_dxr = xr.open_dataset('2md_t,40-61.nc')
        P_xr = xr.open_dataset('2msurf_p,40-61.nc')
        Z_xr = xr.open_dataset('2m_geop,40-61.nc')
    elif yearset == 't2':
        Txr = xr.open_dataset('2mT,61-81.nc')
        T_dxr = xr.open_dataset('2md_t,61-81.nc')
        P_xr = xr.open_dataset('2msurf_p,61-81.nc')
        Z_xr = xr.open_dataset('2m_geop,61-81.nc')
    elif yearset == 't3':
        Txr = xr.open_dataset('2mT,81-02.nc')
        T_dxr = xr.open_dataset('2md_t,81-02.nc')
        P_xr = xr.open_dataset('2msurf_p,81-02.nc')
        Z_xr = xr.open_dataset('2m_geop,81-02.nc')
    elif yearset == 't4':
        Txr = xr.open_dataset('2mT,02-22.nc')
        T_dxr = xr.open_dataset('2md_t,02-22.nc')
        P_xr = xr.open_dataset('2msurf_p,02-22.nc')
        Z_xr = xr.open_dataset('2m_geop,02-22.nc')

    #regrid from native .25x.25 to 1 degree grids

    Txr = Txr.chunk({'time': 100})  # t2m - kelvin
    T_dxr = T_dxr.chunk({'time': 100})  # d2m - kelvin
    P_xr = P_xr.chunk({'time': 100})  # sp - pascals
    Z_xr = Z_xr.chunk({'time': 100})

    newlatgrid = np.arange(-20, 21, 1)
    newlongrid = np.arange(-180, 180, 1)


    Txr = Txr.interp(method='linear', lat=newlatgrid, lon=newlongrid)
    T_dxr = T_dxr.interp(method='linear', lat=newlatgrid, lon=newlongrid)
    P_xr = P_xr.interp(method='linear', lat=newlatgrid, lon=newlongrid)
    Z_xr = Z_xr.interp(method='linear', lat=newlatgrid, lon=newlongrid)

    #again calculating for a day at a time multithreaded
    multiinputs_wbt = []
    for i in range(len(Txr.time.values)):
        multiinputs_wbt += [[P_xr.isel(time=i), Txr.isel(time=i), T_dxr.isel(time=i)]]

    multiinputs_mse = []
    for i in range(len(Txr.time.values)):
        multiinputs_mse += [[P_xr.isel(time=i), Txr.isel(time=i), T_dxr.isel(time=i), Z_xr.isel(time=i)]]

    #initializing blank array for saving data
    outar_wbt = np.zeros((len(Txr.time.values), len(Txr.lat.values), len(Txr.lon.values)), dtype=np.float64)
    outar_mse = np.zeros((len(Txr.time.values), len(Txr.lat.values), len(Txr.lon.values)), dtype=np.float64)

    del Txr, T_dxr, P_xr, Z_xr,newlatgrid, newlongrid

    pool = mp.Pool(processes=16)
    testresults = pool.starmap(calculate_daily_wbt, multiinputs_wbt)

    for i, ar in enumerate(testresults): #indexes through time to save as 3d array in time-lat-lon coordinates
        outar_wbt[i, :, :] = ar
    #saves for each set of years
    outstr = yearset + 'wbts.npy'
    np.save(outstr, outar_wbt)

    pool = mp.Pool(processes=16)
    testresults = pool.starmap(calculate_daily_mse, multiinputs_mse)

    for i, ar in enumerate(testresults): #indexes through time to save as 3d array in time-lat-lon coordinates
        outar_mse[i, :, :] = ar
    #saves for each set of years
    outstr = yearset + 'mses.npy'
    np.save(outstr, outar_mse)

    del outstr, outar_mse,outar_wbt, testresults, pool, multiinputs_mse, multiinputs_wbt


if __name__ == '__main__':
    multithreaded_era5_wbtcalc('t1')
    multithreaded_era5_wbtcalc('t2')
    multithreaded_era5_wbtcalc('t3')
    multithreaded_era5_wbtcalc('t4')

#saves all the wet bulb temps into a netcdf
wbt1 = np.load('t1wbts.npy')
wbt2 = np.load('t2wbts.npy')
wbt3 = np.load('t3wbts.npy')
wbt4 = np.load('t4wbts.npy')
wbtsfull = np.concatenate((wbt1,wbt2,wbt3,wbt4))

#assign new lat and lon grids to wrap furthest longitudes around for interpolation
newlatgrid = np.arange(-20,21,1)
newlongrid = np.arange(-180,181,1)

wbtds = xr.Dataset()
wbtds = wbtds.assign_coords({'time':eratest.time,'lat':newlatgrid,'lon':newlongrid})
newwbtar = np.zeros((len(wbtds.time),len(wbtds.lat),len(wbtds.lon)))
newwbtar[:,:,:-1] = wbtsfull
newwbtar[:,:,-1] = wbtsfull[:,:,0]
wbtds['wbt'] = (('time','lat','lon'),newwbtar)
wbtds.to_netcdf('surfwbt,40-22.nc') #surface wet-bulb temperature array


#concatting mses
mse1 = np.load('t1mses.npy')
mse2 = np.load('t2mses.npy')
mse3 = np.load('t3mses.npy')
mse4 = np.load('t4mses.npy')
msesfull = np.concatenate((mse1,mse2,mse3,mse4))

#assign new lat and lon grids to wrap furthest longitudes around for interpolation
mseds = xr.Dataset()
mseds = mseds.assign_coords({'time':eratest.time,'lat':newlatgrid,'lon':newlongrid})
newmsear = np.zeros((len(mseds.time),len(mseds.lat),len(mseds.lon)))
newmsear[:,:,:-1] = msesfull
newmsear[:,:,-1] = msesfull[:,:,0]
wbtds['mse'] = (('time','lat','lon'),newmsear)
wbtds.to_netcdf('surfmse,40-22.nc') #surface mse temperature array

#calculate satdef from 850hpa relative and specific humidity

q1 = xr.open_dataset('40-22,850hpa_sphum.nc',chunks={'time':100}) #kg/kg
r1 = xr.open_dataset('40-22,850hpa_rhum.nc',chunks={'time':100}) # in %

satdefar = (q1.q / (r1.r/100) - q1.q)
satdefar = satdefar.values


satdef_850 = xr.Dataset()
satdef_850 = satdef_850.assign_coords({'time':eratest.time,'lat':newlatgrid,'lon':newlongrid})

newsatdefar = np.zeros((len(satdef_850.time),len(satdef_850.lat),len(satdef_850.lon)))
newsatdefar[:,:,:-1] = satdefar
newsatdefar[:,:,-1] = satdefar[:,:,0]
satdef_850['satdef'] = (('time','lat','lon'),newsatdefar)
satdef_850.to_netcdf('satdef850,40-22.nc')

#calculate 500 hpa saturation MSE from relative and specific humidity, temperature, geopotential
q500 = xr.open_dataset('40-22,500hpa_sphum.nc',chunks={'time':100}) #kg/kg
r500 = xr.open_dataset('40-22,500hpa_rhum.nc',chunks={'time':100}) # in %
z500 = xr.open_dataset('40-22,500hpa_geop.nc',chunks={'time':100})
T500 = xr.open_dataset('40-22,500hpa_T.nc',chunks={'time':100})

mse_500 = q500.q/(r500.r/100) * lv + T500.t*cp + z500.z
mse500ar = mse_500.values

satmse_500 = xr.Dataset()
satmse_500 = satmse_500.assign_coords({'time':eratest.time,'lat':newlatgrid,'lon':newlongrid})

newsatmsear = np.zeros((len(satmse_500.time),len(satmse_500.lat),len(satmse_500.lon)))
newsatmsear[:,:,:-1] = mse500ar
newsatmsear[:,:,-1] = mse500ar[:,:,0]
satmse_500['satmse'] = (('time','lat','lon'),newsatmsear)
satmse_500.to_netcdf('satmse_500,40-22.nc')