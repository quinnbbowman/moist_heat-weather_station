def calculate_month_hot_inds(month_dailyds):
    tempar = month_dailyds.where(month_dailyds.mse.count(dim='time') >= 25) #only use months with at least 25 days

    dates = tempar.time.where(tempar.mse == tempar.mse.quantile(1, dim='time')) #select the time of the maximum mse
    lons = tempar.longitude.where(tempar.mse == tempar.mse.quantile(1, dim='time')) #longitude
    lats = tempar.latitude.where(tempar.mse == tempar.mse.quantile(1, dim='time')) #latitude

    dtar = np.squeeze(dates.values)  # days in months x stations
    lonsar = np.squeeze(lons.values.T)
    latsar = np.squeeze(lats.values.T)

    monthlydats = []
    monthlylons = []
    monthlylats = []

    for i in range(len(dtar[0, :])):  # indexing through stations

        mondt = dtar[:, i] #selecting the days in the month for every station
        monlon = lonsar[:, i]
        monlat = latsar[:, i]

        cleandt = mondt[~np.isnan(mondt)]  #removing all of the non-maximums from the month
        cleanlon = monlon[~np.isnan(monlon)]
        cleanlat = monlat[~np.isnan(monlat)]

        if (len(cleandt) == 0) or (len(cleanlon) == 0) or (len(cleanlat) == 0): #if no maximum:
            monthlydats += [np.nan] #the value is just a nan
            monthlylons += [np.nan]
            monthlylats += [np.nan]
        else:
            hotdate = cleandt[0] #or else save the first maximum in a list (for the station-month
            hotlon = cleanlon[0]
            hotlat = cleanlat[0]

            monthlydats += [np.int64(hotdate)] #put times into int form for interpolation
            monthlylons += [hotlon]
            monthlylats += [hotlat]

    return [monthlydats, monthlylons, monthlylats]


def calculate_month_hot_vals(month_dailyds):
    tempar = month_dailyds.where(month_dailyds.mse.count(dim='time') >= 25)
    msemean = tempar.mse.where(tempar.mse == tempar.mse.quantile(1, dim='time'))  # .mean(dim='time') #value of monthly maximum mse, removes all other days
    msear = np.squeeze(msemean.values).T  # stations x days in months

    monmaxmse = []

    for i in range(len(msear[0, :])):  # indexing through stations
        monmse = msear[:, i]  # selects single station month
        cleanmse = monmse[~np.isnan(monmse)]  # dates of the maximum MSE days
        if (len(cleanmse) == 0):  # if there's no "maximum" (no data or less than 25 days)
            monmaxmse += [np.nan]  # then monthly maximum is a nan
        else:
            hotmse = cleanmse[0]  # otherwise the maximum mse is returned and appended to a list
            monmaxmse += [hotmse]

    return monmaxmse  # array of shape 934 of monthly maximum MSE values in j/kg


def calculate_month_hot_vals_wbt(month_dailyds):
    tempar = month_dailyds.month_dailyds(daily.mse.count(dim='time') >= 25)
    wbtmean = tempar.wet_bulb_temperatures.where(tempar.mse == tempar.mse.quantile(1, dim='time'))
    wbtar = np.squeeze(wbtmean.values).T  # stations x days in months

    monmaxwbt = []

    for i in range(len(wbtar[0, :])):  # indexing through stations
        monwbt = wbtar[:, i]  # selects single station month
        cleanwbt = monwbt[~np.isnan(monwbt)]  # removing nans from the flat list of wbts

        if (len(cleanwbt) == 0):  # if there's no "maximum" (no data or less than 25 days)
            monmaxwbt += [np.nan]  # then monthly maximum is a nan

        else:
            hotwbt = cleanwbt[0]  # otherwise the wbt on maximum MSE day is returned and appended to a list
            monmaxwbt += [hotwbt]

    return monmaxwbt  # array of shape 934 of monthly maximum WB temperatiers


def calculate_month_maxes():
    indays = xr.open_dataset('reprocessed_daily_stats_calcWBT.nc')

    dsm = indays.resample(time='M')
    inmonths = []
    for group_name, group_ds in dsm:
        inmonths += [group_ds]  # all stations, 1 month get put into the processing script

    pool = mp.Pool(processes=16)
    outar = np.zeros((len(testresults),934,3))
    for monthind, monthdata  in enumerate(testresults):
        for varind,vardata in enumerate(monthdata): # dates,lats,lons
            outar[monthind,:,varind] = vardata
    np.save('monthmsemaxind[mon,stat,[time,lon,lat].npy',outar)


    testresults = pool.map(calculate_month_hot_vals,inmonths) # this will return per month [highest MSE]
    outar = np.zeros((len(testresults),934))
    for monthind, monthdata in enumerate(testresults):
        outar[monthind,:] = monthdata
    np.save('monthmsemax[mon,stat].npy',outar)


    testresults = pool.map(calculate_month_hot_vals_wbt,inmonths)
    outar = np.zeros((len(testresults), 934))
    for monthind, monthdata in enumerate(testresults):
        outar[monthind, :] = monthdata
    np.save('monthwbtmax[mon,stat].npy', outar)


import time
import xarray as xr
import os
import pandas as pd
import numpy as np
import multiprocessing as mp

if __name__ == '__main__':
    calculate_month_maxes()