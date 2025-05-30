import numpy as np
import xarray as xr
import time
import os
import pandas as pd
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator as RGI
import scipy.stats as stats
import matplotlib.colors as colors

lv = 2501000 # j/kg
cp = 1005.7 # J / (kg * K)

era5_wbt_path  = ''
era5_surfmse_path = ''
era5_lft_satdef_path = ''
era5_ft_satmse_path = ''

#flat numpy arrays are all station monthly maximum values, calculated in calculate_monthly_maxes.py
#the era5 counterparts are generated in interpolate_reanalysis.py
#multiple dimensional arrays are of shape months x stations, and are converted to flat of same size by flattening, then removing nans

interpolated_era5_lftsatdef_path = '' #flat numpy array - preprocessed in month max generation code
interpolated_era5_zonalmean_satmse_500_path = '' #flat numpy array - preprocessed in month max generation code
interpolated_era5_satmse_500_path = '' #flat numpy array - preprocessed in month max generation code

monmax_locationarray_path = '' #numpy array shape of number of months x station id, array list [time, lat, lon], created in month max generation code

monmax_stat_surfmse_path = '' #flat numpy array - preprocessed in month max generation code
monmax_era5_surfmse_path = '' #flat numpy array
monmax_stat_surf_wbts_path = '' #flat numpy array
monmax_era5_surf_wbts_path = '' #flat numpy array

def linregress_constant_yoffset(xs,ys): # calculating the slope of a best fit linear regression, holding the y intercept to 0
    numerators = []
    denominators = []
    for i in range(len(xs)):
        numerators += [xs[i]*ys[i]]
        denominators += [xs[i]*xs[i]]
    slope = np.nansum(numerators) / np.nansum(denominators)
    return slope

def linregress_constant_slope(xs,ys,slope): # calculating  the y intercept of a best fit line, holding the slope constant
    xs_mean = np.nanmean(xs)
    ys_mean = np.nanmean(ys)
    intercept = ys_mean - slope * xs_mean
    return slope, intercept

def linregress_full(xs,ys): # calculating the best fit linear regression line
    xs_mean = np.nanmean(xs)
    ys_mean = np.nanmean(ys)
    numerators = []
    denominators = []
    for i in range(len(xs)):
        numerators += [(xs[i] - xs_mean) * (ys[i] - ys_mean)]
        denominators += [(xs[i] - xs_mean)*(xs[i] - xs_mean)]
    slope = np.nansum(numerators) / np.nansum(denominators)
    intercept = ys_mean - slope * xs_mean
    return slope, intercept

#tropical mean reanalysis EQE relationship binning
tr_surfmse = xr.open_dataset(era5_surfmse_path).mse.mean(dim='lon').mean(dim='lat').values
tr_lftsatdef = xr.open_dataset(era5_lft_satdef_path).satdef.mean(dim='lon').mean(dim='lat').values
tr_ftmse = xr.open_dataset(era5_ft_satmse_path).satmse.mean(dim='lon').mean(dim='lat').values

inst = tr_surfmse - tr_ftmse #undilute instability

maxrange = int(np.nanpercentile(tr_lftsatdef,q=99)*2501000/1000) #99th percentile tropical mean saturation deficit
minrange = int(np.nanpercentile(tr_lftsatdef,q=1)*2501000/1000) # 1st percentile tropical mean saturation deficit
trmean_surfbins = np.arange(minrange,maxrange+1)

#calculates the saturation deficit (x axis) values for the tropical mean bin values
xbinmeans = np.arange(minrange+.5,maxrange+.5)

#calculates the mean undilute for each tropical mean bin  saturation deficit
trmean_instvals = []
for i in range(len(trmean_surfbins)-1):
    binmin = trmean_surfbins[i]
    binmax = trmean_surfbins[i+1]
    binvals = []
    for j in range(len(tr_lftsatdef)):
        if (lv*tr_lftsatdef[j]/1000 > binmin) and (lv*tr_lftsatdef[j]/1000 < binmax):
            binvals += [inst[j]/1000]
    binmean = np.nanmean(binvals)
    trmean_instvals += [binmean]

#tropical mean reanalysis wet bulb temperatures
tr_wbt = xr.open_dataset(era5_wbt_path).wbt.mean(dim='lon').mean(dim='lat').values - 273.15

#calculates the mean surface wet-bulb temperature for each tropical mean bin  saturation deficit
trmean_wbtvals = []
for i in range(len(trmean_surfbins)-1):
    binmin = trmean_surfbins[i]
    binmax = trmean_surfbins[i+1]
    binvals = []
    for j in range(len(tr_lftsatdef)):
        if (lv*tr_lftsatdef[j]/1000 > binmin) and (lv*tr_lftsatdef[j]/1000 < binmax):
            binvals += [tr_wbt[j]]
    binmean = np.nanmean(binvals)
    trmean_wbtvals += [binmean]

#function for limit extent of colormaaps
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

#binning monthly maximums
localsatdef = np.load(interpolated_era5_lftsatdef_path)
zmeanmse = np.load(interpolated_era5_zonalmean_satmse_500_path)
localftmse = np.load(interpolated_era5_satmse_500_path)


indays_monmax = np.load(monmax_locationarray_path)
times = indays_monmax[:, :, 0]
times_flat = times.flatten()
cleantimes = times_flat[~np.isnan(times_flat)]

surfmses_stat = np.load(monmax_stat_surfmse_path).flatten()
surfmses_stat = surfmses_stat[~np.isnan(surfmses_stat)]

surfmses_era5 = np.load(monmax_era5_surfmse_path)

surf_wbts_stat = np.load(monmax_stat_surf_wbts_path).flatten()
surf_wbts_stat = surf_wbts_stat[~np.isnan(surf_wbts_stat)]


surf_wbts_era5 = np.load(monmax_era5_surf_wbts_path) - 273.15

monthdata = np.zeros((len(surfmses_stat),8))
monthdata[:,0] = surfmses_stat
monthdata[:,1] = surfmses_era5
monthdata[:,2] = surf_wbts_stat
monthdata[:,3] = surf_wbts_era5
monthdata[:,4] = localftmse
monthdata[:,5] = zmeanmse
monthdata[:,6] = localsatdef
monthdata[:,7] = cleantimes
monmaxpd = pd.DataFrame(monthdata,columns=['surfmses_stat','surfmses_era5','surf_wbts_stat','surf_wbts_era5','h5local','zmeanh5','850satdef','timeints'])

monmaxpd = monmaxpd[monmaxpd['surfmses_stat'].notna()] # removing nans from the dataframe
monmaxpd = monmaxpd[monmaxpd['zmeanh5'].notna()] # removing nans from the dataframe

# Classifies relevant data by percentile

# number of y bins
numybins = 5
ybindiv = 1 / numybins

pcsy = np.arange(0, 1 + ybindiv, ybindiv)  # y percentile values
pcsx = np.arange(0, 1.05, .05)  # x bin percentile values

dqpmins = []  # lower bin limit saturation deficit
dqpmaxs = []  # upper bin limit saturation deficit

hstarmins = []  # lower bin limit zonal mean h*500
hstarmaxs = []  # upper bin limit zonal mean h*500

wtgdevmins = []  # lower bin limit deviation from WTG
wtgdevmaxs = []  # upper bin limit deviation from WTG

# calculates bin minimums and maximums
for i in range(len(pcsx) - 1):
    dqpmin = monmaxpd['850satdef'].quantile(q=pcsx[i])
    dqpmax = monmaxpd['850satdef'].quantile(q=pcsx[i + 1])
    dqpmins += [dqpmin]
    dqpmaxs += [dqpmax]

for i in range(len(pcsy) - 1):
    hstarmin = monmaxpd['zmeanh5'].quantile(q=pcsy[i])
    hstarmax = monmaxpd['zmeanh5'].quantile(q=pcsy[i + 1])
    hstarmins += [hstarmin]
    hstarmaxs += [hstarmax]

# calculates bin mean saturation deficit
sdmeans = []
# calculates bin zonal mean saturation mse
hsmeans = []

# all saturation deficit bin limits (for use in plotting)
sdlims = [dqpmins[0]] + dqpmaxs
sdlims = np.array(sdlims)

# indexes through saturation deficit bins to calculate the mean saturation deficit per bin
for i in range(len(pcsx) - 1):
    monmaxpd.loc[(monmaxpd['850satdef'] >= dqpmins[i]) & (monmaxpd['850satdef'] <= dqpmaxs[i]), 'satdefrank'] = i
    pvals = monmaxpd['850satdef'].where(monmaxpd['satdefrank'] == i).dropna().values
    sdmeans += [np.nanmean(pvals)]  # calculates mean saturation deficit for the bin

# indexes through zonal mean free tropospheric temperature bins to calculate the mean per bin
for i in range(len(pcsy) - 1):
    monmaxpd.loc[(monmaxpd['zmeanh5'] >= hstarmins[i]) & (monmaxpd['zmeanh5'] <= hstarmaxs[i]), 'zmeanh5rank'] = i
    pvals = monmaxpd['zmeanh5'].where(monmaxpd['zmeanh5rank'] == i).dropna().values
    hsmeans += [np.nanmean(pvals)]

fullgrid_era5_med = np.zeros((len(pcsy) - 1,
                              len(pcsx) - 1))  # creates empty array of median reanalysis surface MSE for each saturation deficit and zonal mean h*500 bin
fullgrid_ws_med = np.zeros((len(pcsy) - 1, len(pcsx) - 1))  # same, but weather station surface MSE
fullgrid_h5loc_med = np.zeros((len(pcsy) - 1, len(pcsx) - 1))  # same, but local h*500 (not zonal mean)
fullgridwbt_ws_med = np.zeros((len(pcsy) - 1, len(pcsx) - 1))  # weather station wet-bulb temperatures
fullgridwbt_e5_med = np.zeros((len(pcsy) - 1, len(pcsx) - 1))  # reanalysis surface wet-bulb temperatures

# indexes through h*500 and q- bins
for x in range(len(pcsx) - 1):
    for y in range(len(pcsy) - 1):
        bin = monmaxpd['surfmses_era5'].where(monmaxpd['satdefrank'] == x).where(
            monmaxpd['zmeanh5rank'] == y).dropna().values
        fullgrid_era5_med[y, x] = np.nanmedian(bin)  # calculates media era5 surface MSE

        bin = monmaxpd['surfmses_stat'].where(monmaxpd['satdefrank'] == x).where(
            monmaxpd['zmeanh5rank'] == y).dropna().values
        fullgrid_ws_med[y, x] = np.nanmedian(bin)  # calculates median station surface MSE

        bin = monmaxpd['h5local'].where(monmaxpd['satdefrank'] == x).where(monmaxpd['zmeanh5rank'] == y).dropna().values
        fullgrid_h5loc_med[y, x] = np.nanmedian(bin)  # calculates median local free tropospheric temperature

        bin = monmaxpd['surf_wbts_stat'].where(monmaxpd['satdefrank'] == x).where(
            monmaxpd['zmeanh5rank'] == y).dropna().values
        fullgridwbt_ws_med[y, x] = np.nanmedian(
            bin)  # caluclates bin median weather station surface wet-bulb temperatures

        bin = monmaxpd['surf_wbts_era5'].where(monmaxpd['satdefrank'] == x).where(
            monmaxpd['zmeanh5rank'] == y).dropna().values
        fullgridwbt_e5_med[y, x] = np.nanmedian(bin)  # caluclates bin median reanalysis surface wet-bulb temperatures

# puts arrays in dataframe and array form, from arrays and lists

sdmeans = np.array(sdmeans) * lv  # from kg/kg to j/kj
hsmeans = np.array(hsmeans)

e5meddf = pd.DataFrame(fullgrid_era5_med)
wsmeddf = pd.DataFrame(fullgrid_ws_med)

h5df_med = pd.DataFrame(fullgrid_h5loc_med)

era_instdf = e5meddf - h5df_med  # undilute instability
ws_instdf = wsmeddf - h5df_med  # untilute instability

ws_wbtdf = pd.DataFrame(fullgridwbt_ws_med)
era_wbtdf = pd.DataFrame(fullgridwbt_e5_med)

#-----making plots-----

def double_line_plot(era_instdf, ws_instdf, trxbinmeans, trmeanmse, bin_boundaries):
    # takes in pandas dataframes of reanalysis and weather station undilute instability
    # tropical mean arrays of undilute instability, and tropical mean saturation deficit locations
    # also takes in saturation deficit bin boundaries

    # formatting
    fig = plt.figure(figsize=(16, 8))

    subplot_title_fontsize = 20
    label_title_fontsize = 18
    tick_fontsize = 16

    pcsy = np.arange(0, 1 + ybindiv, ybindiv)

    n_lines = len(pcsy) - 1
    c = np.arange(1, n_lines + 1)
    cmap = plt.get_cmap('plasma')
    cmapsubset = truncate_colormap(cmap, 0, .7)
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmapsubset)
    cmap.set_array([])

    gs = fig.add_gridspec(2, 3, bottom=0.1, top=0.9, wspace=0.05, hspace=.04, width_ratios=(10, 10, 1),
                          height_ratios=(10, 0.5))

    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1], sharey=ax_l)
    ax_cbar = fig.add_subplot(gs[0, 2])

    ax_bl = fig.add_subplot(gs[1, 0], sharex=ax_l)
    ax_br = fig.add_subplot(gs[1, 1], sharex=ax_r)

    plt.setp(ax_r.get_yticklabels(), visible=False);

    # for each free tropospheric temperature bin
    for y in range(len(pcsy) - 1):
        e5meanar = np.squeeze(era_instdf.iloc[y,
                              :].values)  # ERA5 undilute instability varying with satdef along a constant free tropospheric temperature percentile
        wsmeanar = np.squeeze(ws_instdf.iloc[y,
                              :].values)  # weather station undilute instability varying with satdef along a constant free tropospheric temperature percentile

        ax_l.plot(sdmeans / 1000, e5meanar / 1000, c=cmap.to_rgba(y + 1),
                  lw=2)  # plotting instability per saturation deficit in jk/kg
        ax_r.plot(sdmeans / 1000, wsmeanar / 1000, c=cmap.to_rgba(y + 1), lw=2)

    cbar = fig.colorbar(cmap, ticks=[1, len(pcsy) - 1], cax=ax_cbar)
    cbar.set_ticklabels(['Colder Free \n Troposphere', 'Warmer Free \nTroposphere'], fontsize=tick_fontsize)

    plt.setp(ax_l.get_yticklabels(), fontsize=tick_fontsize);

    ax_l.set_ylabel(r"Bin Undilute Instability $\enspace kjJ\:kg^{-1}$", fontsize=label_title_fontsize)

    fig.suptitle('Binned Instability Response to Saturation Deficit', fontsize=24);

    # plotting tropical mean lines
    xbinmeans = trxbinmeans
    trmean_msevals = trmeanmse
    ax_l.plot(xbinmeans, trmean_msevals, color='black', label='Tropical Mean', lw=3)
    ax_r.plot(xbinmeans, trmean_msevals, color='black', label='Tropical Mean', lw=3)

    ax_l.set_title('Reanalysis', fontsize=subplot_title_fontsize)
    ax_r.set_title('Weather Station', fontsize=subplot_title_fontsize)

    ax_l.text(x=2, y=13, s='a', size=26)
    ax_r.text(x=2, y=13, s='b', size=26)

    # formatting, plotting bin boundaries
    Z = [[1, 0] * 10]
    ax_bl.pcolormesh(bin_boundaries * 2501000 / 1000, [0, 1], Z, cmap='binary')
    ax_bl.arrow(35, .5, 4, 0, head_width=.7, length_includes_head=True, fc='black')
    ax_bl.set_xlim(0, 40)
    ax_bl.set_xlabel(r"Bin Saturation Deficit   $Lv\:q^+ \enspace kj\:kg^{-1}$", fontsize=label_title_fontsize);

    ax_br.pcolormesh(bin_boundaries * 2501000 / 1000, [0, 1], Z, cmap='binary')
    ax_br.arrow(35, .5, 4, 0, head_width=.7, length_includes_head=True, fc='black')
    ax_br.set_xlim(0, 40)
    ax_br.set_xlabel(r"Bin Saturation Deficit   $Lv\:q^+ \enspace kj\:kg^{-1}$", fontsize=label_title_fontsize);

    plt.setp(ax_l.get_xticklabels(), visible=False);
    plt.setp(ax_r.get_xticklabels(), visible=False);
    plt.setp(ax_bl.get_yticklabels(), visible=False);
    plt.setp(ax_br.get_yticklabels(), visible=False);

    plt.setp(ax_bl.get_xticklabels(), fontsize=tick_fontsize);
    plt.setp(ax_br.get_xticklabels(), fontsize=tick_fontsize);
    #
    ax_l.grid()
    ax_r.grid()

    ax_l.legend(fontsize=12)
    ax_r.legend(fontsize=12)

    return fig, ax_l, ax_r


def double_line_plot_wbt(era_wbtdf, ws_wbtdf, trxbinmeans, trmeanwbts, bin_boundaries):
    # takes in pandas dataframes of reanalysis and weather station wet bulb temps
    # tropical mean arrays of wet bulb temperatures, and tropical mean saturation deficit locations
    # also takes in saturation deficit bin boundaries

    # formatting
    subplot_title_fontsize = 20
    label_title_fontsize = 18
    tick_fontsize = 16
    fig = plt.figure(figsize=(16, 8))

    pcsy = np.arange(0, 1 + ybindiv, ybindiv)

    n_lines = len(pcsy) - 1
    c = np.arange(1, n_lines + 1)
    cmap = plt.get_cmap('plasma')
    cmapsubset = truncate_colormap(cmap, 0, .7)
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmapsubset)
    cmap.set_array([])

    gs = fig.add_gridspec(2, 3, bottom=0.1, top=0.9, wspace=0.05, hspace=.04, width_ratios=(10, 10, 1),
                          height_ratios=(10, 0.5))

    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1], sharey=ax_l)
    ax_cbar = fig.add_subplot(gs[0, 2])

    ax_bl = fig.add_subplot(gs[1, 0], sharex=ax_l)
    ax_br = fig.add_subplot(gs[1, 1], sharex=ax_r)

    plt.setp(ax_r.get_yticklabels(), visible=False);

    # for each free tropospheric temperature bin:
    for y in range(len(pcsy) - 1):
        e5meanar = np.squeeze(era_wbtdf.iloc[y,
                              :].values)  # surface MSE varying with satdef along a constant free tropospheric temperature percentile
        wsmeanar = np.squeeze(ws_wbtdf.iloc[y,
                              :].values)  # surface MSE varying with satdef along a constant free tropospheric temperature percentile

        ax_l.plot(sdmeans / 1000, e5meanar, c=cmap.to_rgba(y + 1),
                  lw=2)  # plotting reanalysis varying with satdef surface MSE
        ax_r.plot(sdmeans / 1000, wsmeanar, c=cmap.to_rgba(y + 1),
                  lw=2)  # plotting weather station varying with satdef surface MSE

    cbar = fig.colorbar(cmap, ticks=[1, len(pcsy) - 1], cax=ax_cbar)
    cbar.set_ticklabels(['Colder Free \n Troposphere', 'Warmer Free \nTroposphere'], fontsize=label_title_fontsize)

    plt.setp(ax_l.get_yticklabels(), fontsize=tick_fontsize);
    ax_l.set_ylabel(r"Bin Wet-Bulb Temperature $\degree C $", fontsize=label_title_fontsize)
    fig.suptitle('Binned Wet-Bulb Temperature Response to Saturation Deficit', fontsize=24);

    # plotting tropical mean lines
    xbinmeans = trxbinmeans
    trmeanwbts = trmeanwbts
    ax_l.plot(xbinmeans, trmeanwbts, color='black', label='Tropical Mean', lw=3)
    ax_r.plot(xbinmeans, trmeanwbts, color='black', label='Tropical Mean', lw=3)

    ax_l.set_title('Reanalysis', fontsize=subplot_title_fontsize)
    ax_r.set_title('Weather Station', fontsize=subplot_title_fontsize)
    ax_l.set_xlim(0, 40)
    ax_r.set_xlim(0, 40)
    #
    ax_l.text(x=2, y=25.5, s='a', size=26)
    ax_r.text(x=2, y=25.5, s='b', size=26)

    # formatting, plotting bin boundaries
    Z = [[1, 0] * 10]
    ax_bl.pcolormesh(bin_boundaries * 2501000 / 1000, [0, 1], Z, cmap='binary')
    ax_bl.arrow(35, .5, 4, 0, head_width=.7, length_includes_head=True, fc='black')
    ax_bl.set_xlim(0, 40)
    ax_bl.set_xlabel(r"Bin Saturation Deficit   $Lv\:q^+ \enspace kJ\:kg^{-1}$", fontsize=label_title_fontsize);

    ax_br.pcolormesh(bin_boundaries * 2501000 / 1000, [0, 1], Z, cmap='binary')
    ax_br.arrow(35, .5, 4, 0, head_width=.7, length_includes_head=True, fc='black')
    ax_br.set_xlim(0, 40)
    ax_br.set_xlabel(r"Bin Saturation Deficit   $Lv\:q^+ \enspace kJ\:kg^{-1}$", fontsize=label_title_fontsize);

    plt.setp(ax_l.get_xticklabels(), visible=False);
    plt.setp(ax_r.get_xticklabels(), visible=False);
    plt.setp(ax_bl.get_yticklabels(), visible=False);
    plt.setp(ax_br.get_yticklabels(), visible=False);

    plt.setp(ax_bl.get_xticklabels(), fontsize=tick_fontsize);
    plt.setp(ax_br.get_xticklabels(), fontsize=tick_fontsize);
    #
    ax_l.grid()
    ax_r.grid()

    ax_l.legend(fontsize=12)
    ax_r.legend(fontsize=12)

    return fig, ax_l, ax_r


def comparative_line_plot(era_instdf, ws_instdf, bin_boundaries):
    # takes in pandas dataframes of reanalysis and weather station undilute instability, saturation deficit bin size x free tropoospheric bin size
    # also takes in saturation deficit bin boundaries

    # formatting
    fig = plt.figure(figsize=(16, 8))

    label_title_fontsize = 18
    tick_fontsize = 16

    n_lines = len(pcsy) - 1
    c = np.arange(1, n_lines + 1)
    cmap = plt.get_cmap('plasma')
    cmapsubset = truncate_colormap(cmap, 0, .7)
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmapsubset)
    cmap.set_array([])

    gs = fig.add_gridspec(2, 2, bottom=0.1, top=0.9, wspace=0.05, hspace=.04, width_ratios=(10, .5),
                          height_ratios=(10, 0.5))
    ax_m = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_m)
    ax_cbar = fig.add_subplot(gs[0, 1])

    # for each free tropospheric temperature bin:
    for y in range(len(pcsy) - 1):
        e5medar = np.squeeze(era_instdf.iloc[y,
                             :].values)  # era5 undilute instability varying with satdef along a constant free tropospheric temperature percentile
        wsmedar = np.squeeze(ws_instdf.iloc[y,
                             :].values)  # weather station undilute instability varying with satdef along a constant free tropospheric temperature percentile
        dif = (wsmedar - e5medar) / 1000  # calculates the difference (in kj/kg)
        ax_m.plot(sdmeans / 1000, dif, c=cmap.to_rgba(y + 1),
                  lw=2)  # plots the difference - how much reanalysis underpredicts a given day

    # formatting
    cbar = fig.colorbar(cmap, ticks=[1, len(pcsy) - 1], cax=ax_cbar)

    plt.setp(ax_m.get_xticklabels(), visible=False);
    plt.setp(ax_bot.get_yticklabels(), visible=False);
    plt.setp(ax_bot.get_xticklabels(), fontsize=tick_fontsize);
    plt.setp(ax_m.get_yticklabels(), fontsize=tick_fontsize);

    ax_m.set_ylabel(r"Reanalysis Instability Underprediction $\enspace kJ\:kg^{-1}$", fontsize=label_title_fontsize)
    cbar.set_ticklabels(['Colder Free \n Troposphere', 'Warmer Free \nTroposphere'], fontsize=label_title_fontsize)

    ax_m.set_xlim(0, 40)
    ax_m.grid()
    # plotting bin boundaries
    Z = [[1, 0] * 10]
    ax_bot.pcolormesh(bin_boundaries * 2501000 / 1000, [0, 1], Z, cmap='binary')
    ax_bot.arrow(35, .5, 4, 0, head_width=.7, length_includes_head=True, fc='black')
    ax_bot.set_xlim(0, 40)
    ax_bot.set_xlabel(r"Saturation Deficit   $Lv\:q^+ \enspace kJ\:kg^{-1}$", fontsize=label_title_fontsize);
    fig.suptitle('Reanalysis Instability Underprediction', fontsize=24, y=.95);
    return fig


def scatter_heat_plot_4panel(surfmsear_ws, surfWBTar_ws, surfmsear_era5, surfWBTar_era5, title):
    # takes in pandas dataframes of reanalysis and weather station surface MSE and wet bulb temperature
    # if size saturation deficit bin size x free tropoospheric bin size
    # also takes in string of plot title

    os.chdir('/home/quinn/Documents/driergethotternenso/reprocessed_daily')
    localsatdef = np.load(
        'era5interp_850q-_reprocessed.npy')  # but these are just arrrays of all of the non-nan days -  might be sat q instead?
    zmeanmse = np.load('era5interp_zmeanh*500_reprocessed.npy')
    localftmse = np.load('era5interph*500_reprocessed.npy')

    surfmses_ws = surfmsear_ws
    xs_ws = surfmses_ws - zmeanmse

    surfmses_era5 = surfmsear_era5
    xs_era5 = surfmses_era5 - zmeanmse

    wtgdev = zmeanmse - localftmse
    ys = localsatdef * lv - wtgdev

    xs_ws_iqr = (np.percentile(xs_ws, q=60) - np.percentile(xs_ws, q=40)) / 1000

    # defining plot structure and lines and colors
    [x1, y1] = [0, 0]
    [x2, y2] = [50, -50]
    meancolor = 'white'
    numbins = 80
    meansize = 100

    xmin = -30
    xmax = 50
    ymin = -70
    ymax = 10

    xmin_inset = 5
    xmax_inset = 20
    ymin_inset = -15
    ymax_inset = 0

    limline_colors = 'black'
    limline_width = 2
    inset_colors = 'darkgrey'
    best_fit_linecolor = 'lime'
    constant_slope_linecolor = 'orchid'
    mode_dot_color = 'xkcd:vermillion'
    mode_dot_size = 9
    inset_grid_color = 'darkgrey'
    fit_lw = 2.25

    scatter_cmap = plt.get_cmap('plasma')
    scatter_cmap = truncate_colormap(scatter_cmap, 0, .8)

    hist_cmap_scheme = plt.get_cmap('cividis')
    hist_cmap_scheme = truncate_colormap(hist_cmap_scheme, .1, .9)

    fig = plt.figure(figsize=(16, 15))
    gs0 = fig.add_gridspec(2, 5, bottom=0.1, top=0.9, wspace=0.05, width_ratios=(9, 1, 1.2, 9, 1))

    # weather station plots

    ax_scatter_ws = fig.add_subplot(gs0[0, 0])
    ax_sccolor_ws = fig.add_subplot(gs0[0, 1])
    ax_heat_ws = fig.add_subplot(gs0[0, 3], sharey=ax_scatter_ws)
    ax_heatcolor_ws = fig.add_subplot(gs0[0, 4])

    gs01 = mpl.gridspec.GridSpecFromSubplotSpec(3, 3, wspace=0.1, hspace=0.1, subplot_spec=gs0[0, 3],
                                                width_ratios=(9, 9, 1.1), height_ratios=(9, 9, 1.1))
    ax_heat_ws_inset = fig.add_subplot(gs01[1, 1])
    ax_heat_ws_inset.hist2d(xs_ws / 1000, ys / 1000 * -1, cmin=1, bins=numbins, vmin=0, vmax=1500,
                            range=[[xmin, xmax], [ymin, ymax]], cmap=hist_cmap_scheme)  # heatmap
    ax_heat_ws_inset.scatter(np.nanmedian(xs_ws / 1000), np.nanmedian(ys / 1000 * -1), marker="*", s=meansize * 2,
                             c=meancolor, zorder=3)  # median day
    ax_heat_ws_inset.scatter(np.nanmedian(xs_ws / 1000), np.nanmedian(ys / 1000 * -1), marker="*", s=meansize * 4,
                             c='black', zorder=2)
    ax_heat_ws_inset.set_xlim(xmin_inset, xmax_inset)
    ax_heat_ws_inset.set_ylim(ymin_inset, ymax_inset)
    ax_heat_ws_inset.axline((x1, y1), (x2, y2), color=limline_colors, linewidth=4, zorder=1)
    ax_heat_ws_inset.grid(linewidth=2, color=inset_grid_color)

    ax_heat_ws_inset.set_yticks(ticks=[0, -5, -10, -15])
    ax_heat_ws_inset.set_xticks(ticks=[5, 10, 15, 20])

    for spine in ax_heat_ws_inset.spines.values():
        spine.set_edgecolor(inset_grid_color)
        spine.set_linewidth(3)

    inset_cover = mpl.patches.Rectangle((xmin_inset, ymin_inset), (xmax_inset - xmin_inset), (ymax_inset - ymin_inset),
                                        zorder=5, linewidth=2, edgecolor=inset_colors, facecolor='none')
    ax_heat_ws.add_patch(inset_cover)

    # scatter plot
    scplot_ws = ax_scatter_ws.scatter(xs_ws / 1000, ys / 1000 * -1, c=surfWBTar_ws, cmap=scatter_cmap, s=1, vmin=25,
                                      vmax=29)  # scatter plot
    ax_scatter_ws.scatter(np.nanmedian(xs_ws / 1000), np.nanmedian(ys / 1000 * -1), marker="*", s=meansize, c=meancolor,
                          zorder=3)  # median day
    ax_scatter_ws.scatter(np.nanmedian(xs_ws / 1000), np.nanmedian(ys / 1000 * -1), marker="*", s=meansize * 2,
                          c='black', zorder=2)
    ax_scatter_ws.set_xlim(xmin, xmax)
    ax_scatter_ws.set_ylim(ymin, ymax)
    ax_scatter_ws.axline((x1, y1), (x2, y2), color=limline_colors, linewidth=limline_width, zorder=1)
    ax_scatter_ws.axvline(x=0, color=limline_colors, linewidth=limline_width, zorder=1)
    sc_cbar_ws = fig.colorbar(scplot_ws, cax=ax_sccolor_ws, ticks=np.arange(25, 30))

    # getting bin densities for "convecting" bins
    (histvals, xedges, yedges, output) = ax_heat_ws.hist2d(xs_ws / 1000, ys / 1000 * -1, cmin=1, bins=numbins, vmin=0,
                                                           vmax=1500, range=[[xmin, xmax], [ymin, ymax]],
                                                           cmap=hist_cmap_scheme)

    histvals2 = np.nan_to_num(histvals)
    histmodesx = np.nanargmax(histvals2, axis=0)  # gets x index of maximum bin per saturation deficit

    flatsumsx = np.nansum(histvals, axis=1)  # sum of bins on constant x
    flatsumsy = np.nansum(histvals, axis=0)  # sum of bins on constant y
    xmaxind = np.argmax(flatsumsx)  # gets mode of histogram on x
    ymaxind = np.argmax(flatsumsy)  # gets mode of histogram on x - lmit of plotted points
    xlim_dots = xedges[xmaxind + 1]
    ylim_dots = yedges[ymaxind - 1]  # modifies the limit to account for bin vs edges - maximum is furthest bin edge

    # calculating bin positions of the "convecting limit" for linear maximum
    xloc = []
    yloc = []
    for i in range(len(histmodesx)):
        xval = (xedges[histmodesx[i]] + xedges[
            histmodesx[i] + 1]) / 2  # translates indices of maximums on x, to plot coordinates
        yval = (yedges[i] + yedges[i + 1]) / 2
        if (xval >= -1 * xs_ws_iqr) and (xval <= xlim_dots + xs_ws_iqr) and (yval <= 2) and (yval >= -12):
            xloc += [xval]
            yloc += [yval]
    # plots "convecting" bin locations
    ax_heat_ws.scatter(xloc, yloc, s=mode_dot_size, color=mode_dot_color, zorder=7)
    ax_heat_ws_inset.scatter(xloc, yloc, s=mode_dot_size, color=mode_dot_color, zorder=7)

    # calculates and plots lines following convecting bins
    m, b = linregress_constant_slope(xloc, yloc, slope=-1)
    m_full, b_full = linregress_full(xloc, yloc)
    slope = linregress_constant_yoffset(xloc, yloc)
    xs_line = np.array([xmin, xmax])
    ys_line_constslope = xs_line * m + b
    ys_line_full = xs_line * m_full + b_full
    ys_line_constint = xs_line * slope

    ax_heat_ws.plot(xs_line, ys_line_constslope, label='Best w = 1 Line', color=constant_slope_linecolor, lw=fit_lw)
    ax_heat_ws.plot(xs_line, ys_line_full, label='Best Fit Line', color=best_fit_linecolor, lw=fit_lw)

    ax_heat_ws_inset.plot(xs_line, ys_line_constslope, color=constant_slope_linecolor, lw=fit_lw)
    ax_heat_ws_inset.plot(xs_line, ys_line_full, color=best_fit_linecolor, lw=fit_lw)

    ax_heat_ws.scatter(np.nanmedian(xs_ws / 1000), np.nanmedian(ys / 1000 * -1), marker="*", s=meansize, c=meancolor,
                       zorder=3)
    ax_heat_ws.scatter(np.nanmedian(xs_ws / 1000), np.nanmedian(ys / 1000 * -1), marker="*", s=meansize * 2, c='black',
                       zorder=2)
    ax_heat_ws.set_xlim(xmin, xmax)
    ax_heat_ws.set_ylim(ymin, ymax)
    ax_heat_ws.axline((x1, y1), (x2, y2), color=limline_colors, zorder=1, linewidth=limline_width)
    ax_heat_ws.axvline(x=0, color=limline_colors, zorder=1, linewidth=limline_width)
    ax_hm_cbar_ws = fig.colorbar(output, cax=ax_heatcolor_ws, ticks=np.arange(0, 1800, 300))

    inset_cover_axes = mpl.patches.Rectangle((-2, ymin + 1), (xmax + 1), (44), zorder=5, linewidth=1.5,
                                             edgecolor='black', facecolor='white')
    ax_heat_ws.add_patch(inset_cover_axes)

    # -----reanalysis plots-----

    ax_scatter_era5 = fig.add_subplot(gs0[1, 0], sharex=ax_scatter_ws)
    ax_sccolor_era5 = fig.add_subplot(gs0[1, 1])
    ax_heat_era5 = fig.add_subplot(gs0[1, 3], sharey=ax_scatter_era5, sharex=ax_heat_ws)
    ax_heatcolor_era5 = fig.add_subplot(gs0[1, 4])

    # heatmap
    gs11 = mpl.gridspec.GridSpecFromSubplotSpec(3, 3, wspace=0.1, hspace=0.1, subplot_spec=gs0[1, 3],
                                                width_ratios=(9, 9, 1.1), height_ratios=(9, 9, 1.1))
    ax_heat_era5_inset = fig.add_subplot(gs11[1, 1])
    ax_heat_era5_inset.hist2d(xs_era5 / 1000, ys / 1000 * -1, cmin=1, bins=numbins, vmin=0, vmax=1500,
                              range=[[xmin, xmax], [ymin, ymax]], cmap=hist_cmap_scheme)  # heatmap
    ax_heat_era5_inset.scatter(np.nanmedian(xs_era5 / 1000), np.nanmedian(ys / 1000 * -1), marker="*", s=meansize * 2,
                               c=meancolor, zorder=3)  # median day
    ax_heat_era5_inset.scatter(np.nanmedian(xs_era5 / 1000), np.nanmedian(ys / 1000 * -1), marker="*", s=meansize * 4,
                               c='black', zorder=2)
    ax_heat_era5_inset.set_xlim(xmin_inset, xmax_inset)
    ax_heat_era5_inset.set_ylim(ymin_inset, ymax_inset)
    ax_heat_era5_inset.axline((x1, y1), (x2, y2), color=limline_colors, linewidth=4, zorder=1)
    ax_heat_era5_inset.grid(linewidth=2, color=inset_grid_color)

    ax_heat_era5_inset.set_yticks(ticks=[0, -5, -10, -15])
    ax_heat_era5_inset.set_xticks(ticks=[5, 10, 15, 20])

    for spine in ax_heat_era5_inset.spines.values():
        spine.set_edgecolor(inset_grid_color)
        spine.set_linewidth(3)

    inset_cover = mpl.patches.Rectangle((xmin_inset, ymin_inset), (xmax_inset - xmin_inset), (ymax_inset - ymin_inset),
                                        zorder=5, linewidth=2, edgecolor=inset_colors, facecolor='none')
    ax_heat_era5.add_patch(inset_cover)

    # scatter plot
    scplot_era5 = ax_scatter_era5.scatter(xs_era5 / 1000, ys / 1000 * -1, c=surfWBTar_era5, cmap=scatter_cmap, s=1,
                                          vmin=25, vmax=29)  # scatter plot
    ax_scatter_era5.scatter(np.nanmedian(xs_era5 / 1000), np.nanmedian(ys / 1000 * -1), marker="*", s=meansize,
                            c=meancolor, zorder=3)  # median day
    ax_scatter_era5.scatter(np.nanmedian(xs_era5 / 1000), np.nanmedian(ys / 1000 * -1), marker="*", s=meansize * 2,
                            c='black', zorder=2)
    ax_scatter_era5.set_xlim(xmin, xmax)
    ax_scatter_era5.set_ylim(ymin, ymax)
    ax_scatter_era5.axline((x1, y1), (x2, y2), color=limline_colors, linewidth=limline_width, zorder=1)
    ax_scatter_era5.axvline(x=0, color=limline_colors, linewidth=limline_width, zorder=1)
    sc_cbar_era5 = fig.colorbar(scplot_era5, cax=ax_sccolor_era5, ticks=np.arange(25, 30))

    # getting bin densities for "convecting" bins
    (histvals, xedges, yedges, output) = ax_heat_era5.hist2d(xs_era5 / 1000, ys / 1000 * -1, cmin=1, bins=numbins,
                                                             vmin=0, vmax=1500, range=[[xmin, xmax], [ymin, ymax]],
                                                             cmap=hist_cmap_scheme)

    histvals2 = np.nan_to_num(histvals)
    histmodesx = np.nanargmax(histvals2, axis=0)  # gets x index of maximum bin per saturation deficit

    flatsumsx = np.nansum(histvals, axis=1)  # sum of bins on constant x
    flatsumsy = np.nansum(histvals, axis=0)  # sum of bins on constant y
    xmaxind = np.argmax(flatsumsx)  # gets mode of histogram on x
    ymaxind = np.argmax(flatsumsy)  # gets mode of histogram on x - lmit of plotted points
    xlim_dots = xedges[xmaxind + 1]

    # calculating bin positions of the "convecting limit" for linear maximum
    xloc = []
    yloc = []
    for i in range(len(histmodesx)):
        xval = (xedges[histmodesx[i]] + xedges[
            histmodesx[i] + 1]) / 2  # translates indices of maximums on x, to plot coordinates
        yval = (yedges[i] + yedges[i + 1]) / 2
        if (xval >= -1 * xs_ws_iqr) and (xval <= xlim_dots + xs_ws_iqr) and (yval <= 2) and (yval >= -12):
            xloc += [xval]
            yloc += [yval]
    # plots "convecting" bin locations
    ax_heat_era5.scatter(xloc, yloc, s=mode_dot_size, color=mode_dot_color, zorder=7)
    ax_heat_era5_inset.scatter(xloc, yloc, s=mode_dot_size, color=mode_dot_color, zorder=7)

    # calculates and plots lines following convecting bins
    m, b = linregress_constant_slope(xloc, yloc, slope=-1)
    m_full, b_full = linregress_full(xloc, yloc)
    slope = linregress_constant_yoffset(xloc, yloc)
    xs_line = np.array([xmin, xmax])
    ys_line_constslope = xs_line * m + b
    ys_line_full = xs_line * m_full + b_full

    ax_heat_era5.plot(xs_line, ys_line_constslope, label='Best w = 1 Line ', color=constant_slope_linecolor, lw=fit_lw)
    ax_heat_era5.plot(xs_line, ys_line_full, label='Best Fit Line', color=best_fit_linecolor, lw=fit_lw)

    ax_heat_era5_inset.plot(xs_line, ys_line_constslope, color=constant_slope_linecolor, lw=fit_lw)
    ax_heat_era5_inset.plot(xs_line, ys_line_full, color=best_fit_linecolor, lw=fit_lw)

    ax_heat_era5.scatter(np.nanmedian(xs_era5 / 1000), np.nanmedian(ys / 1000 * -1), marker="*", s=meansize,
                         c=meancolor, zorder=3)
    ax_heat_era5.scatter(np.nanmedian(xs_era5 / 1000), np.nanmedian(ys / 1000 * -1), marker="*", s=meansize * 2,
                         c='black', zorder=2)
    ax_heat_era5.set_xlim(xmin, xmax)
    ax_heat_era5.set_ylim(ymin, ymax)
    ax_heat_era5.axline((x1, y1), (x2, y2), color=limline_colors, zorder=1, linewidth=limline_width)
    ax_heat_era5.axvline(x=0, color=limline_colors, zorder=1, linewidth=limline_width)
    ax_hm_cbar_era5 = fig.colorbar(output, cax=ax_heatcolor_era5, ticks=np.arange(0, 1800, 300))

    inset_cover_axes = mpl.patches.Rectangle((-2, ymin + 1), (xmax + 1), (44), zorder=5, linewidth=1.5,
                                             edgecolor='black', facecolor='white')
    ax_heat_era5.add_patch(inset_cover_axes)

    # formatting
    subplot_title_fontsize = 20
    label_title_fontsize = 18
    tick_fontsize = 16

    plt.setp(ax_heat_era5_inset.get_yticklabels(), fontsize=tick_fontsize);
    plt.setp(ax_heat_era5_inset.get_xticklabels(), fontsize=tick_fontsize);
    plt.setp(ax_heat_ws_inset.get_yticklabels(), fontsize=tick_fontsize);
    plt.setp(ax_heat_ws_inset.get_xticklabels(), fontsize=tick_fontsize);

    plt.setp(ax_heat_era5.get_yticklabels(), fontsize=tick_fontsize);
    plt.setp(ax_heat_era5.get_xticklabels(), fontsize=tick_fontsize);
    plt.setp(ax_scatter_era5.get_yticklabels(), fontsize=tick_fontsize);
    plt.setp(ax_scatter_era5.get_xticklabels(), fontsize=tick_fontsize);

    plt.setp(ax_heat_ws.get_yticklabels(), fontsize=tick_fontsize);
    plt.setp(ax_heat_ws.get_xticklabels(), fontsize=tick_fontsize);
    plt.setp(ax_scatter_ws.get_yticklabels(), fontsize=tick_fontsize);
    plt.setp(ax_scatter_ws.get_xticklabels(), fontsize=tick_fontsize);

    plt.setp(ax_sccolor_ws.get_yticklabels(), fontsize=tick_fontsize);
    plt.setp(ax_sccolor_era5.get_yticklabels(), fontsize=tick_fontsize);
    plt.setp(ax_heatcolor_ws.get_yticklabels(), fontsize=tick_fontsize);
    plt.setp(ax_heatcolor_era5.get_yticklabels(), fontsize=tick_fontsize);

    plt.setp(ax_heat_ws.get_yticklabels(), visible=False);
    plt.setp(ax_heat_era5.get_yticklabels(), visible=False);

    ax_hm_cbar_era5.set_label("Bin Count", fontsize=label_title_fontsize)
    ax_hm_cbar_ws.set_label("Bin Count", fontsize=label_title_fontsize)

    ax_scatter_ws.set_title('Weather Station Wet-Bulb Temperatures', fontsize=subplot_title_fontsize, y=1.01)
    ax_heat_ws.set_title('Weather Station Buoyancy Budget Heatmap', fontsize=subplot_title_fontsize, y=1.01)

    ax_scatter_era5.set_title('Reanalysis Wet-Bulb Temperatures', fontsize=subplot_title_fontsize, y=1.01)
    ax_heat_era5.set_title('Reanalysis Buoyancy Budget Heatmap', fontsize=subplot_title_fontsize, y=1.01)

    ax_heat_era5.set_xlabel(r'$MSE_{surf} - MSE^*_{500 \:ZM} \enspace kJ\:kg^{-1}$', fontsize=label_title_fontsize)

    ax_scatter_ws.set_ylabel(r"$Lv\:q^+ \:- \:[MSE^*_{500}]\,^{'} \enspace kJ\:kg^{-1}$", fontsize=label_title_fontsize)
    ax_scatter_era5.set_xlabel(r'$MSE_{surf} - MSE^*_{500 \:ZM} \enspace kJ\:kg^{-1}$', fontsize=label_title_fontsize)
    ax_scatter_era5.set_ylabel(r"$Lv\:q^+ \:- \:[MSE^*_{500}]\,^{'} \enspace kJ\:kg^{-1}$",
                               fontsize=label_title_fontsize)

    sc_cbar_ws.set_label(r"Surface Wet-Bulb Temperature $ \degree C $", fontsize=label_title_fontsize)
    sc_cbar_era5.set_label(r"Surface Wet-Bulb Temperature $ \degree C $", fontsize=label_title_fontsize)

    # abcd labels
    ax_scatter_ws.text(x=-25, y=5, s='a', size=26)
    ax_heat_ws.text(x=-25, y=5, s='b', size=26)
    ax_scatter_era5.text(x=-25, y=5, s='c', size=26)
    ax_heat_era5.text(x=-25, y=5, s='d', size=26)

    ax_heat_ws.legend(fontsize=12)
    ax_heat_era5.legend(fontsize=12)

    fig.suptitle(title, fontsize=24, y=.95)
