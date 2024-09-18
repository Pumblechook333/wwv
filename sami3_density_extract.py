#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:52:33 2024

@author: kuldeep
"""

# import datetime as dt
import numpy as np
# import pandas as pd
from netCDF4 import Dataset
# from scipy.interpolate import griddata
# from scipy.signal import savgol_filter
from scipy import interpolate
from geopy.distance import geodesic
import datetime
# import matplotlib.dates as dtformat
# import matplotlib as mpl
from matplotlib import pyplot as plt
# import cartopy
# import cartopy.feature as cf
import cartopy.crs as ccrs

year = 2021
month = 7
day = 1
hour = 0
minute = 0
date = datetime.datetime(year, month, day, hour, minute)

setup = {
    'region_lat': [39, 41],
    'region_lon': [-106, -70],
    'hh': date.hour,
    'mm': date.minute,
    'choose_alt': 200,
    'day': date.timetuple().tm_yday
}


# %% user inputs
def main():
    file_path = 'D:/Sabastian/Perry_Lab/'
    file_name = 'sami3_dene0Gs.nc'
    file = file_path + file_name

    region = False
    contour = False
    point = True

    if region:
        # select data within geographic region
        region_lat = setup['region_lat']
        region_lon = setup['region_lon']

        # import SAMI3 outputs within selected geographic region
        ut, alt, lat, lon, ne = to_ne_region(file, region_lat, region_lon)

        if not contour:
            return ut, alt, lat, lon, ne

    if contour and region:
        # select values for a particular time
        # day = 1  # 0-first day, 1-second day
        hh = setup['hh']
        mm = setup['mm']

        # previous day
        day = setup['day'] - 1
        idx_ut = to_idx_ut(ut, day, hh, mm)
        ut_pre = ut[idx_ut]
        if ut_pre > 24: ut_pre = ut_pre - 24
        temp_ne_pre = ne[idx_ut, :, :, :]

        # present day
        day = setup['day']
        idx_ut = to_idx_ut(ut, day, hh, mm)
        ut_0 = ut[idx_ut]
        if ut_0 > 24: ut_0 = ut_0 - 24
        temp_ne_0 = ne[idx_ut, :, :, :]

        # ne values at a fixed altitude
        choose_alt = setup['choose_alt']
        idx_alt = np.argmin(np.abs(alt - choose_alt))
        ne_alt_pre = temp_ne_pre[:, idx_alt, :]
        ne_alt_0 = temp_ne_0[:, idx_alt, :]

        # plot contours on geographic map
        to_plot_contours_region(lat, lon, ut_pre, ne_alt_pre, ut_0, ne_alt_0)

        if not point:
            return ut, alt, lat, lon, ne

    if point:
        altitude = 200
        latitude = 41.75
        longitude = -89.62

        point_ne = to_ne_point(file, setup['day'], date, altitude, latitude, longitude)

        return point_ne


# %% import ne values within a selected region
def to_ne_region(file, region_lat, region_lon):
    # to import data, keep longitude in range 0-360
    if region_lon[0] < 0: region_lon[0] = region_lon[0] + 360
    if region_lon[-1] < 0: region_lon[-1] = region_lon[-1] + 360

    # import SAMI3 file
    file_id = Dataset(file)

    # e.g. to check dimensions of a varibale
    # print(file_id.variables['lat0'])

    temp_time = file_id.variables['time'][:]
    temp_alt = file_id.variables['alt0'][:]
    temp_lat = file_id.variables['lat0'][:]
    temp_lon = file_id.variables['lon0'][:]

    # select values within given geographic region
    idx_lat_i = np.argmin(np.abs(temp_lat - region_lat[0]))
    idx_lat_f = np.argmin(np.abs(temp_lat - region_lat[-1]))

    idx_lon_i = np.argmin(np.abs(temp_lon - region_lon[0]))
    idx_lon_f = np.argmin(np.abs(temp_lon - region_lon[-1]))

    ut = temp_time
    alt = temp_alt
    lat = temp_lat[idx_lat_i:idx_lat_f]
    lon = temp_lon[idx_lon_i:idx_lon_f]

    # import density -- index order is ut_time, lon, alt, lat 
    ne = file_id.variables['dene0'][:, idx_lon_i:idx_lon_f, :, idx_lat_i:idx_lat_f]

    # del idx_lat_i, idx_lat_f, idx_lon_i, idx_lon_f
    # del temp_time, temp_alt, temp_lat, temp_lon

    # change the longitude range between -180 and 180
    for i in range(0, len(lon)):
        if lon[i] > 180:
            lon[i] = lon[i] - 360

    return ut, alt, lat, lon, ne


# %% altitude profile of ne values interpolated at each 1km
def to_ne_point(file, day_no, ut_temp, p_alt, p_lat, p_lon):
    # hour and minute of selected time
    hh = ut_temp.hour
    mm = ut_temp.minute

    # make sure p_lon is in range 0-360
    if p_lon < 0: p_lon = p_lon + 360

    # import SAMI3 file
    file_id = Dataset(file)

    time = file_id.variables['time'][:]
    alt = file_id.variables['alt0'][:]
    lat = file_id.variables['lat0'][:]
    lon = file_id.variables['lon0'][:]

    # find index of time
    idx_ut = to_idx_ut(time, day_no, hh, mm)

    # find the nearest neighbour point
    temp = np.argwhere((lat >= p_lat - 3) & (lat <= p_lat + 3))
    idx_lat = temp.reshape(-1)
    temp = np.argwhere((lon >= p_lon - 3) & (lon <= p_lon + 3))
    idx_lon = temp.reshape(-1)

    fi_lat = 0
    fi_lon = 0

    temp_dist = 10 ** 6
    for i in range(0, len(idx_lat)):

        for j in range(0, len(idx_lon)): \
                # calcuate distance of each data location from the desired location
            # Reference: https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude

            point0 = (p_lat, p_lon)
            point1 = (lat[idx_lat[i]], lon[idx_lon[j]])  # (latitude, longitude)
            dist = geodesic(point0, point1).kilometers  # Distance due to changes in latitude

            # index of nearest neighbour in lat-lon plane
            if dist < temp_dist:
                temp_dist = dist
                fi_lat = idx_lat[i]
                fi_lon = idx_lon[j]

    f_lat = lat[fi_lat]
    f_lon = lon[fi_lon]

    # import density -- index order is ut_time, lon, alt, lat 
    ne = file_id.variables['dene0'][idx_ut, fi_lon, :, fi_lat]

    # fit altitude profile of ne to get values at each 1km
    fit_ne = interpolate.interp1d(alt, ne, 'cubic')
    point_ne = fit_ne(p_alt)

    # crosschek interpolated values
    fig = plt.figure(figsize=(12, 6))
    plt.subplot()
    plt.semilogx(ne, alt, '-b')
    plt.axhline(p_alt, color='r')
    # plt.semilogx(point_ne, p_alt, 'or', linewidth=0.5)
    plt.ylim([0, 400])
    plt.grid(which='both')
    t = f"Sami3 Electron Density at [{p_lat},{p_lon}]"
    plt.title(t)
    plt.xlabel("Density (electrons / cm^3)")
    plt.ylabel("Elevation (km)")

    return point_ne


# %% get index of selected ut time within SAMI3 output
def to_idx_ut(ut_array, day, hh, mm):
    total_hh = day * 24 + hh + mm / 60
    idx = np.argmin(np.abs(ut_array - total_hh))

    return idx


# %% plot tec on geographic map
def to_plot_ne_region(idx_alt, lat, lon, ut_pre, temp_ne_pre, ut_0, temp_ne_0):
    # contour plot on geographic map
    fig = plt.figure(figsize=(12, 6))
    crs = ccrs.PlateCarree()

    # previous day
    ax = fig.add_subplot(211, projection=crs)
    ax.coastlines(resolution='110m', color='k', zorder=1, alpha=1)

    temp = np.transpose(temp_ne_pre[:, idx_alt, :])
    # temp = np.log10(temp)
    cmp = plt.contourf(lon, lat, temp,
                       zorder=0, transform=ccrs.PlateCarree(), alpha=1)

    plt.colorbar(cmp, orientation="vertical")

    plt.title(str(ut_pre) + ' UT')

    # present day
    ax = fig.add_subplot(212, projection=crs)
    ax.coastlines(resolution='110m', color='k', zorder=1, alpha=1)

    temp = np.transpose(temp_ne_0[:, idx_alt, :])
    # temp = np.log10(temp)
    cmp = plt.contourf(lon, lat, temp, zorder=0, transform=ccrs.PlateCarree(), alpha=1)

    plt.colorbar(cmp, orientation="vertical")

    plt.title(str(ut_0) + ' UT')

    return


# %% plot tec on geographic map
def to_plot_contours_region(lat, lon, ut_pre, tec_pre, ut_0, tec_0):
    fig = plt.figure(figsize=(12, 6))
    crs = ccrs.PlateCarree()

    # previous day
    ax = fig.add_subplot(211, projection=crs)
    ax.coastlines(resolution='110m', color='k', zorder=1, alpha=1)

    cmp = plt.contourf(lon, lat, tec_pre.T,
                       zorder=0, transform=ccrs.PlateCarree(), alpha=1)

    plt.colorbar(cmp, orientation="vertical")
    plt.title(str(ut_pre) + ' UT')

    # present day
    ax = fig.add_subplot(212, projection=crs)
    ax.coastlines(resolution='110m', color='k', zorder=1, alpha=1)

    cmp = plt.contourf(lon, lat, tec_0.T,
                       zorder=0, transform=ccrs.PlateCarree(), alpha=1)

    plt.colorbar(cmp, orientation="vertical")
    plt.title(str(ut_0) + ' UT')

    return


# %% line plot
def to_plot_line(x, y, x_label, y_label):
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot()
    plt.plot(x, y, '-k')
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    return


# %% Run the whole code if this script is run
if __name__ == "__main__":
    # ut, alt, lat, lon, ne = main()
    point = main()

    plt.show()
    plt.close()
