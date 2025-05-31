#code for combining all weather stations, hourly
import xarray as xr
import multiprocessing as mp
import numpy as np
import os
import time
import pandas as pd
from metpy.calc import wet_bulb_temperature
from metpy.units import units

#this code combines all weather stations from tropical weather stations in the name format as downloaded from the hadISD portal
# in 2 folders 'WMOHUM', which is the humidity weather station data,
# and 'WMODATA', which is the temperature data.  These will both contain the same weather stations, but the humidity and temperature files separately.

filelist = []
for root, dirs, files in os.walk("WMOHUM", topdown=False):
    for file in files:
        filelist += [root+'/'+file]

timeseriesfull = np.arange(np.datetime64('1931-01'),np.datetime64('2021-06'),np.timedelta64(1,'h')) #timeseries to combine all weather stations by, hourly data

#getting name of weather station files so each index collects a single weather station, both temperature and humidity files
filename = filelist[0]
basestr = 'WMODATA'+filename[6:]
basestr = basestr[:25]+basestr[34:]
basestr = basestr[:-12]+'.nc'
basestr = basestr[:-19] + '5' + basestr[-18:]
filenamep = basestr[:44] + '4' + basestr[45:]

ds = xr.open_dataset(filename).reindex({"time":timeseriesfull})
ds2 = xr.open_dataset(filenamep).reindex({"time":timeseriesfull})


#combining the humidity and temperature datasets for the first station
savedatavarslist = []
for datavar in ds.data_vars:
    savedatavarslist += [datavar]

savedatavarslistp = []
for datavar in ds2.data_vars:
    if datavar not in savedatavarslist:
        savedatavarslistp += [datavar]

for datavar in savedatavarslistp:
    ds[datavar] = ds2[datavar]
#ds is combined filelist now, of the first weather station in the list

sidlist = []
sidlist += [ds.attrs['station_id']] #makes a list of the station id number

combined = ds.copy(deep=True) #makes a copy so it doesn't get modified

combinedlist = [ds] # this is a list of all datasets to be combined, starting with the first index

for filename in filelist[1:]: #for each humidity station filename:
    #making combined file
    basestr = 'WMODATA'+filename[6:]
    basestr = basestr[:25]+basestr[34:]
    basestr = basestr[:-12]+'.nc'
    basestr = basestr[:-19] + '5' + basestr[-18:]
    filenamep = basestr[:44] + '4' + basestr[45:] # gets the corresponding station temperature file

    ds = xr.open_dataset(filename).reindex({"time":timeseriesfull}) #opens both
    ds2 = xr.open_dataset(filenamep).reindex({"time":timeseriesfull})

    for datavar in savedatavarslistp:
        ds[datavar] = ds2[datavar] #combines the data into the humidity file

    sidlist += [ds.attrs['station_id']] #adds the station id to the list

    combinedlist += [ds] #adds the combined dataset to the list

combined = xr.concat(combinedlist,dim='statid') #concats all weather stations into a single file
combined = combined.assign_coords(statid=(sidlist)) #adds station id as a dimension

combined.to_netcdf('combined_stations_all.nc') #saves as a netcdf

#code for resampling into daily


# turns hourly data into daily averages.  uses multithreading for speed.
# makes 4 npy arrays of shape number of stations (hardcoded here as 934) x number of days
# it does it in 4 chunks because of python errors

#code for daily averaging
#only averages a day, if it contains at least 4 measurements,
def dailyaverage(dayno, ds):
    #code for daily averaging, ensuring each day is only used if it contains enough data
    #takes in a day index (because the full time length is split into multiple sections)
    # and an xarray dataset of a single day (in hourly measurements)
    ds.load()

    numstats = 934
    dayTar = np.zeros(numstats)
    dayPar = np.zeros(numstats)
    dayQar = np.zeros(numstats)
    daytdar = np.zeros(numstats)

    for s in range(numstats):  # for every station
        stat_day = ds.isel(statid=s)
        stat_day.load()
        #get all of the hourly indices of humidity measurements (as there are the fewest of these)

        raw_huminds = np.argwhere(stat_day.specific_humidity.notnull().values == True)
        proc_huminds = []
        for ind in raw_huminds:
            proc_huminds += [ind[0].tolist()]

        if len(proc_huminds) >= 4:  # If the day includes at least 4 measurements
            #get all of the hourly measurements _only_ on the indices of the humidity meaurements
            #so they are all coincident in time
            nummeas = len(proc_huminds)


            Tvals = stat_day.temperatures.isel(time=proc_huminds).dropna(dim='time').values
            pvals = stat_day.stnlp.isel(time=proc_huminds).dropna(dim='time').values
            Qvals = stat_day.specific_humidity.isel(time=proc_huminds).dropna(dim='time').values
            tdvals = stat_day.dewpoints.isel(time=proc_huminds).dropna(dim='time').values

            timevals = stat_day.time.isel(time=proc_huminds).dropna(dim='time').values

            del stat_day

            if (len(Tvals) != nummeas) or (len(pvals) != nummeas) or (len(tdvals) != nummeas) or (
                    len(Qvals) != nummeas) or (np.int64(timevals[-1] - timevals[0]) < np.int64(
                    6.12e13)):  # if we're missing any measurements, or if the difference between first and last time in day is less than 17 hours, don't use the day (turn to nan)
                outval = np.nan #if not enough data or too close together, daily mean is just a nan
                dayTar[s] = outval
                dayPar[s] = outval
                dayQar[s] = outval
                daytdar[s] = outval

            else: #succesfully daily averaging
                dayTar[s] = np.mean(Tvals)
                dayPar[s] = np.mean(pvals)
                dayQar[s] = np.mean(Qvals)
                daytdar[s] = np.mean(tdvals)

        else: # if not enough measurements, then nans for the day
            del stat_day
            outval = np.nan

            dayTar[s] = outval
            dayPar[s] = outval
            dayQar[s] = outval
            daytdar[s] = outval

    ds.close()

    return (dayTar, dayPar, dayQar, daytdar) #returns 1d arrays of every variable for the day


def multithreaded_dailyavg(timeseg):
    dailystats = xr.open_dataset('/home/quinn/Downloads/combined_stations_all.nc')
    #drops unnecessary variables (for memory optimization)
    dailystats = dailystats.drop_vars(
        ['relative_humidity', 'saturation_vapor_pressure', 'vapor_pressure', 'slp', 'wet_bulb_temperature'])
    dailystats = dailystats.chunk(chunks={'time': 1440})
    #removes invalid measurements from weather stations

    dailystats['temperatures'] = dailystats['temperatures'].where(dailystats['temperatures'] != -2e+30)
    dailystats['stnlp'] = dailystats['stnlp'].where(dailystats['stnlp'] != -2e+30)
    dailystats['specific_humidity'] = dailystats['specific_humidity'].where(dailystats['specific_humidity'] != -2e+30)
    dailystats['dewpoints'] = dailystats['dewpoints'].where(dailystats['dewpoints'] != -2e+30)

    section = timeseg

    # separating into sections because it was breaking when done as one chunk

    if section == 'p1':
        dmslice = dailystats.isel(time=slice(0, 198143))

    elif section == 'p2':
        dmslice = dailystats.isel(time=slice(198144, 396287))

    elif section == 'p3':
        dmslice = dailystats.isel(time=slice(396288, 594431))

    elif section == 'p4':
        dmslice = dailystats.isel(time=slice(594432, 792575))

    #resamples the time slice into days
    dmslice_d = dmslice.resample(time='D')

    #gets indices and corresponding day dataseets
    inds = []
    for index, (name, ds) in enumerate(dmslice_d):
        inds += [[index, ds]]

    dailystats.close()
    del dailystats, dmslice_d, dmslice
    #multithreaded daily averaging
    pool = mp.Pool(processes=16)
    results = pool.starmap(dailyaverage, inds)  # so this will output every day

    #initializing daily mean numpy arrays
    numstats = 934

    dailymeansT = np.zeros((len(results), numstats))
    dailymeansP = np.zeros((len(results), numstats))
    dailymeansQ = np.zeros((len(results), numstats))
    dailymeanstd = np.zeros((len(results), numstats))

    for i in range(len(results)):
        day = results[i]

        T = day[0]
        P = day[1]
        Q = day[2]
        td = day[3]

        dailymeansT[i, :] = T
        dailymeansP[i, :] = P
        dailymeansQ[i, :] = Q
        dailymeanstd[i, :] = td

    del results
    np.save('reprocessed_daily_T' + section + '.npy', dailymeansT)
    np.save('reprocessed_daily_p' + section + '.npy', dailymeansP)
    np.save('reprocessed_daily_Q' + section + '.npy', dailymeansQ)
    np.save('reprocessed_daily_td' + section + '.npy', dailymeanstd)
# then these need to be combined into reprocessed_daily_stations.nc, and WBT calculated

if __name__ == '__main__':
    start_time = time.time()
    multithreaded_dailyavg('p1')
    multithreaded_dailyavg('p2')
    multithreaded_dailyavg('p3')
    multithreaded_dailyavg('p4')

#code for combining the different numpy arrays back into a netcdf
T1 = np.load('reprocessed_daily_Tp1.npy')
T2 = np.load('reprocessed_daily_Tp2.npy')
T3 = np.load('reprocessed_daily_Tp3.npy')
T4 = np.load('reprocessed_daily_Tp4.npy')

Q1 = np.load('reprocessed_daily_Qp1.npy')
Q2 = np.load('reprocessed_daily_Qp2.npy')
Q3 = np.load('reprocessed_daily_Qp3.npy')
Q4 = np.load('reprocessed_daily_Qp4.npy')

p1 = np.load('reprocessed_daily_pp1.npy')
p2 = np.load('reprocessed_daily_pp2.npy')
p3 = np.load('reprocessed_daily_pp3.npy')
p4 = np.load('reprocessed_daily_pp4.npy')

td1 = np.load('reprocessed_daily_tdp1.npy')
td2 = np.load('reprocessed_daily_tdp2.npy')
td3 = np.load('reprocessed_daily_tdp3.npy')
td4 = np.load('reprocessed_daily_tdp4.npy')

tar = np.concatenate((T1,T2,T3,T4))
qar = np.concatenate((Q1,Q2,Q3,Q4))
par = np.concatenate((p1,p2,p3,p4))
tdar = np.concatenate((td1,td2,td3,td4))

fullstatds = xr.open_dataset('/home/quinn/Downloads/combined_stations_all.nc')
timear = fullstatds.temperatures.resample(time='D').mean(dim='time')
timear = timear.time

reprocessed_daily_stats = xr.Dataset()
reprocessed_daily_stats['time'] = timear
reprocessed_daily_stats['statid'] = fullstatds['statid']

reprocessed_daily_stats['temperatures'] = (('time','statid'),tar)
reprocessed_daily_stats['temperatures'] = reprocessed_daily_stats['temperatures'].assign_attrs(units='celsius')

reprocessed_daily_stats['specific_humidity'] = (('time','statid'),qar)
reprocessed_daily_stats['specific_humidity'] = reprocessed_daily_stats['specific_humidity'].assign_attrs(units='g/kg')

reprocessed_daily_stats['stnlp'] = (('time','statid'),par)
reprocessed_daily_stats['stnlp'] = reprocessed_daily_stats['stnlp'].assign_attrs(units='hpa')

reprocessed_daily_stats['dewpoints'] = (('time','statid'),tdar)
reprocessed_daily_stats['dewpoints'] = reprocessed_daily_stats['dewpoints'].assign_attrs(units='celsius')

reprocessed_daily_stats['elevation'] = (('statid'),np.squeeze(elevals))
reprocessed_daily_stats['elevation'] = reprocessed_daily_stats['elevation'].assign_attrs(units='m')

reprocessed_daily_stats['latitude'] = (('statid'),np.squeeze(latvals))
reprocessed_daily_stats['latitude'] = reprocessed_daily_stats['latitude'].assign_attrs(units='degrees')

reprocessed_daily_stats['longitude'] = (('statid'),np.squeeze(lonvals))
reprocessed_daily_stats['longitude'] = reprocessed_daily_stats['longitude'].assign_attrs(units='degrees')

reprocessed_daily_stats = reprocessed_daily_stats.assign_attrs(description='Daily means calculated from days with 4 or more measurements, with the 2 furthest temporal measurements more than 17 hours apart')

reprocessed_daily_stats.to_netcdf('reprocessed_daily_stats.nc') #daily averages now in one netcdf file


#code for adding wet-bulb temperature and moist static energy to weather stations
def calculate_daily_wbt(ds):
    #takes in a day dataset (size = number of stations) with temperatures, dewpoints, and station level pressure
    ds = ds.load()

    tar = ds.temperatures
    tdar = ds.dewpoints
    par = ds.stnlp

    wetbulbar = wet_bulb_temperature(par.values * units.hPa, tar.values * units.degC,
                                     tdar.values * units.degC).magnitude

    ds = ds.close()

    del ds
    return wetbulbar


def calculate_daily_mse(ds):
    ds = ds.load()

    tar = ds.temperatures.values
    qar = ds.specific_humidity.values
    zar = ds.elevation.values

    msear = 2501000 * qar / 1000 + (tar + 273.15) * 1005.7 + zar * 9.81

    ds = ds.close()

    del ds
    return msear


def multithreaded_mse_wbt_calc():
    indays = xr.open_dataset('reprocessed_daily_stats.nc')

    #removes nans for use in wbt calculations, otherwise it fails
    #assuming pressure temperature and wbt in the tropics all are non 0
    indays = indays.fillna(0)
    indays = indays.resample(time='D') #separates the netcdf file into individual days for calculation

    inputs = []
    for index, ds in indays:
        inputs += [ds] #make a list of every input dataset to index through for multithreading

    outar_wbt = np.zeros(np.shape(instats.temperatures), dtype=np.float64) #initializing empty arrays for the 2 calculated variables
    outar_mse = np.zeros(np.shape(instats.temperatures), dtype=np.float64)

    del indays, instats

    pool = mp.Pool(processes=16)
    #multithreaded wbt calcuations
    testresults = pool.map(calculate_daily_wbt, inputs)

    for i, ar in enumerate(testresults):
        outar_wbt[i, :] = ar
    del pool, testresults, inputs
    #code for saving wet bulb temperatures

    reprocessed_daily_stats = xr.open_dataset('reprocessed_daily_stats.nc')
    reprocessed_daily_stats['wet_bulb_temperature'] = (('time', 'statid'), outar_wbt)
    reprocessed_daily_stats['wet_bulb_temperature'] = reprocessed_daily_stats['wet_bulb_temperature'].assign_attrs(
        units='celsius', description='calculated from metpy wet_bulb_temperature (Normand method)')

    #replaces the calculated 0s back with nans again
    reprocessed_daily_stats['temperatures'] = reprocessed_daily_stats['temperatures'].where(
        reprocessed_daily_stats['temperatures'] != 0)
    reprocessed_daily_stats['specific_humidity'] = reprocessed_daily_stats['specific_humidity'].where(
        reprocessed_daily_stats['specific_humidity'] != 0)
    reprocessed_daily_stats['stnlp'] = reprocessed_daily_stats['stnlp'].where(reprocessed_daily_stats['stnlp'] != 0)
    reprocessed_daily_stats['dewpoints'] = reprocessed_daily_stats['dewpoints'].where(
        reprocessed_daily_stats['dewpoints'] != 0)
    reprocessed_daily_stats['wet_bulb_temperature'] = reprocessed_daily_stats['wet_bulb_temperature'].where(
        reprocessed_daily_stats['wet_bulb_temperature'] != 0)

    #now we do the same for moist static energy
    #don't need to replace nans here, nans will turn the variable into a nan
    reprocessed_daily_stats_days = reprocessed_daily_stats.resample(time='D')

    inputs = []
    for index, ds in reprocessed_daily_stats_days:
        inputs += [ds]

    pool = mp.Pool(processes=16)
    #calculating daily MSE values
    testresults = pool.map(calculate_daily_mse,inputs)

    for i, ar in enumerate(testresults):
        outar_mse[i,:] = ar

    reprocessed_daily_stats['mse'] = (('time','statid'),outar_mse)
    reprocessed_daily_stats['mse'] = reprocessed_daily_stats['mse'].assign_attrs(units='j/kg')

    #saves final file, daily netcdf of weather stations, including wet bulb temp and MSE.

    reprocessed_daily_stats.to_netcdf('reprocessed_daily_stats_calcWBT.nc') #final product

if __name__ == '__main__':
    multithreaded_mse_wbt_calc()