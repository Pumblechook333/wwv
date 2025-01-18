#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    This module contains the definitions for grape and grapeHandler objects, intended to process data from Grape SDR V1
    stations receiving from WWV-10 in Fort Collins, Colorado

    @Author: Sabastian Carlos Fernandes [New Jersey Institute of Technology]
    @Date: 1.25.2023
    @Version: 1.0
    @Credit:    Dr. Gareth Perry [New Jersey Institute of Technology],
                Tiago Trigo [New Jersey Institute of Technology],
                John Gibbons [Case Western Reserve University],
                Ham Radio Science Citizen Investigation (HamSCI)
"""

# Imports ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import matplotlib.pyplot as plt
from matplotlib import colors, cm
import pylab as pl
import numpy as np
from math import floor, ceil, sin, cos, sqrt, atan2, radians as rad, degrees as deg
from csv import reader
import os
import imageio as imageio
from re import sub
from fitter import Fitter
from datetime import datetime, timedelta
import suncalc
import pandas as pd
from pvlib import solarposition

import pickle
from tqdm import tqdm

# Global Variables ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FNAMES = ('d', 'dop', 'doppler', 'doppler shift', 'f', 'freq', 'frequency')
VNAMES = ('v', 'volt', 'voltage')
PNAMES = ('db', 'decibel', 'p', 'pwr', 'power')

FLABEL = 'Doppler Shift (Hz)'
PLABEL = 'Relative Power (dB)'
VLABEL = 'Voltage (V)'

MONTHS = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', '')
MONTHLEN = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 0)
MONTHINDEX = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365)

# WWV Broadcasting tower coordinates based on:
# https://latitude.to/articles-by-country/us/united-states/6788/wwv-radio-station
WWV_LAT = 40.67583063
WWV_LON = -105.038933178

NJ_DATA_PATH = 'C:/Users/sabas/Documents/NJIT/Work/wwv/DATA/NJ_data'
K2MFF_SIG = 'T000000Z_N0000020_G1_FN20vr_FRQ_WWV10'


class Grape:

    # Loading ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, filename: str, convun: bool = True, filt: bool = False, med: bool = False,
                 count: bool = False, n: int = 1):
        """
        Constructor for a Grape object

        :param filename: Name of the .txt file where the data is kept in tab delimited format
        :param filt: Boolean for if you are filtering or not (default = F)
        :param convun: Boolean for if you want a unit range to be auto created (default = T)
        :param n: Subsampling term
        """

        # Metadata containers
        self.date = None
        self.year = None
        self.month = None
        self.day = None

        self.node = None
        self.gridsq = None

        self.lat = None
        self.lon = None
        self.blat = None  # Coordinates of bounce location
        self.blon = None

        self.t_offset = None  # time offset of midpoint from UTC

        self.ele = None
        self.cityState = None
        self.radID = None
        self.beacon = None

        # Raw data containers
        self.time = None
        self.freq = None
        self.Vpk = None
        self.Vdb = None  # Vpk converted to logscale

        self.freq_filt = None
        self.Vpk_filt = None

        # Raw data adjusted to be plotted with correct units
        self.t_range = None
        self.f_range = None
        self.Vdb_range = None

        self.f_range_filt = None
        self.Vdb_range_filt = None

        # Calculated sun position and choice times (correlated to time series)
        self.zentrace = None

        self.TXsuntimes = None
        self.Bsuntimes = None
        self.RXsuntimes = None

        # Counting variables for collections.counter
        self.f_count = None
        self.Vpk_count = None
        self.Vdb_count = None

        self.bestFits = None
        self.dayMed = None
        self.nightMed = None

        self.dayIQR = None
        self.nightIQR = None

        # Flags to keep track of if the load() or units() function have been called, respectively
        self.loaded = False
        self.converted = False
        self.filtered = False

        # Load core Grape properties
        if filename:
            self.load(filename, n=n)

        # Generate useful information to share between all plots
        fontsize = 22
        self.plot_settings = {
            # Title Information
            'date'      : self.date,
            'lat'       : decdeg2dms(self.lat),
            'lon'       : decdeg2dms(self.lon),
            'blat'      : decdeg2dms(self.blat),
            'blon'      : decdeg2dms(self.blon),
            'wwvlat'    : decdeg2dms(WWV_LAT),
            'wwvlon'    : decdeg2dms(WWV_LON),

            # Title Formatting
            'y'         : 1,
            'pad'       : fontsize - 2,
            'fontsize'  : fontsize
        }
        self.axcount = 0

        if self.loaded:
            if filt:
                self.butFilt()
            if convun:
                self.units()
            if count:
                self.count()
            if med:
                self.dnMedian()
        else:
            raise "Grape not loaded! (self.loaded = False)"

    def load(self, filename: str, n: int = 1):
        """
        Script to load grape data from wwv text file into Grape object

        :param filename: Path of the .txt file containing the grape data in the local repo
        :param n: Subsampling term (every nth)
        :return: loaded grape to grape object
        """

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initializing time, freq and voltage arrays
        self.time = []
        self.freq = []
        self.Vpk = []

        dataFile = open(filename)
        dataReader = reader(dataFile)
        lines = list(dataReader)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save the header data separately from the plottable data

        header_data = lines[:18]

        self.date = str(header_data[0][1]).split('T')[0]
        self.node = header_data[0][2]
        self.gridsq = header_data[0][3]
        self.lat = float(header_data[0][4])
        self.lon = float(header_data[0][5])
        self.ele = float(header_data[0][6])
        self.cityState = header_data[0][7]
        self.radID = header_data[0][8]
        self.beacon = header_data[0][9]

        splitdat = self.date.split('-')
        self.year = int(splitdat[0])
        self.month = int(splitdat[1])
        self.day = int(splitdat[2])

        d1 = datetime(self.year, self.month, self.day)  # Datetime obj for select day
        # d2 = datetime(self.year, self.month, self.day + 1)  # Datetime obj for following day

        # Calculate the midpoint coordinate and timezone offset from UTC
        self.blat, self.blon = mpt_coords(WWV_LAT, WWV_LON, self.lat, self.lon)
        self.t_offset = round_down(self.blon / 15)

        # "Sun-time" calculations performed with suncalc
        height = 200e3
        self.RXsuntimes = suncalc.get_times(d1, self.lon, self.lat, height=height)
        self.Bsuntimes = suncalc.get_times(d1, self.blon, self.blat, height=height)
        self.TXsuntimes = suncalc.get_times(d1, WWV_LON, WWV_LAT, height=height)

        # Read each line of file after the header
        for line in lines[19::n]:
            date_time = str(line[0]).split('T')
            utc_time = str(date_time[1]).split(':')

            hour = int(utc_time[0])
            minute = int(utc_time[1])
            second = int(utc_time[2][0:2])

            # # Suncalc Package [Do Not Use for SZA]
            # d = datetime(self.year, self.month, self.day, hour, minute, second)
            # pos = suncalc.get_position(d, self.blon, self.blat)
            # alt = pos['altitude']
            # self.alttrace.append(alt * (180 / np.pi))

            sec = (float(hour) * 3600) + \
                  (float(minute) * 60) + \
                  (float(second))

            self.time.append(sec)  # time list append
            self.freq.append(float(line[1]))  # doppler shift list append
            self.Vpk.append(float(line[2]))  # voltage list append

        # Save final loaded data to numpy arrays
        self.time = np.array(self.time)
        self.freq = np.array(self.freq)
        self.Vpk = np.array(self.Vpk)

        # Solar Zenith angle calculated with pvlib for all times in self.time
        sza_times = []
        for time in self.time:
            newtime = d1 + timedelta(seconds=time)
            sza_times.append(newtime)

        solpos = solarposition.get_solarposition(sza_times, self.blat, self.blon, altitude=height)
        self.zentrace = solpos['zenith'].values

        if len(self.time) != 0:
            if self.beacon == 'WWV10':
                # Raise loaded flag
                self.loaded = True
                # print("Grape " + self.date + " loaded! \n")
            else:
                # print("Grape " + self.date + " not loaded (not WWV10) \n")
                pass
        else:
            # print("Grape " + self.date + " not loaded (no data) \n")
            pass

    # Get Properties ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def getTFV(self):
        """
        Getter for Grape object's time, frequency and peak voltage values

        :return: time, freq and Vpk values
        """
        if self.loaded:
            return np.array([self.time, self.freq, self.Vpk])
        else:
            return np.array([None, None, None])

    def getTFPr(self):
        """
        Getter for Grape object's converted time, frequency and relative power ranges for use in plotting

        :return: time, freq and Vdb ranges
        """
        if self.converted:
            return np.array([self.t_range, self.f_range, self.Vdb_range])
        else:
            return np.array([None, None, None])

    def getFiltTFPr(self):
        """
        Getter for Grape object's converted and filtereed time, frequency and relative power ranges for use in plotting

        :return: time, freq and Vdb ranges
        """
        if self.converted and self.filtered:
            return np.array([self.t_range, self.f_range_filt, self.Vdb_range_filt])
        else:
            return np.array([None, None, None])

    # Data Processing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def butFilt(self, FILTERORDER: int = 3, FILTERBREAK: float = 0.005):
        """
        Filtering the data with order 3 Butterworth low-pass filter
        Butterworth Order: https://rb.gy/l4pfm

        :param FILTERORDER: Order of the Butterworth filter
        :param FILTERBREAK: Cutoff Frequency of the Butterworth Filter
        :return: data filtered by butterworth filter to grape object
        """

        from scipy.signal import filtfilt, butter

        if self.loaded:
            # noinspection PyTupleAssignmentBalance
            b, a = butter(FILTERORDER, FILTERBREAK, analog=False, btype='low')

            self.freq_filt = filtfilt(b, a, self.freq)
            self.Vpk_filt = filtfilt(b, a, self.Vpk)

            self.filtered = True
        else:
            print("Frequency and Vpk not loaded!")

    def units(self, timediv: int = 3600, fdel: int = 10e6):
        """
        Converting raw/filtered Grape data to correct units for plotting

        :param timediv: Dividing factor for seconds data
        :param fdel: Reference value for doppler shift (10GHz for WWV)
        :return: converted data to grape object
        """

        if self.loaded:
            self.t_range = self.time / timediv
            self.f_range = self.freq - fdel
            self.Vdb_range = 10 * np.log10(self.Vpk ** 2)

            if self.filtered:
                self.f_range_filt = self.freq_filt - fdel
                self.Vdb_range_filt = 10 * np.log10(self.Vpk_filt ** 2)

            self.converted = True
        else:
            print('Time, frequency and Vpk not loaded!')

    def dnMedian(self):
        """
        Calculates the medians of the entire day's sunlight and sundown doppler shifts, seperately

        :return: day and night medians to grape object
        """

        Bsr = conv_time(self.Bsuntimes['sunrise'])
        Bss = conv_time(self.Bsuntimes['sunset'])

        srIndex = min(range(len(self.t_range)), key=lambda i: abs(self.t_range[i] - Bsr))
        ssIndex = min(range(len(self.t_range)), key=lambda i: abs(self.t_range[i] - Bss))

        if ssIndex < srIndex:
            sunUp = self.f_range[0:ssIndex]
            for i in self.f_range[srIndex:(len(self.f_range) - 1)]:
                # sunUp.append(i)
                sunUp = np.append(sunUp, i)
            sunDown = self.f_range[ssIndex:srIndex]
        else:
            sunDown = self.f_range[0:srIndex]
            for i in self.f_range[ssIndex:(len(self.f_range) - 1)]:
                # sunDown.append(i)
                sunDown = np.append(sunDown, i)
            sunUp = self.f_range[srIndex:ssIndex]

        qmarks = [0.25, 0.50, 0.75]

        if len(sunUp) > 1:
            qt = np.quantile(sunUp, qmarks)
            self.dayMed = qt[1]
            self.dayIQR = qt[2] - qt[0]
        else:
            print('No sunup detected (dayMed None)\n')
        if len(sunDown) > 1:
            qt = np.quantile(sunDown, qmarks)
            self.nightMed = qt[1]
            self.nightIQR = qt[2] - qt[0]
        else:
            print('No sundown detected (nightMed None)\n')

    def count(self):
        """
        Employs collections.Counter to produce counts of individual values in grape value ranges

        :return: discrete counts to grape object
        """

        from collections import Counter

        if self.converted:
            self.f_count = Counter(self.f_range)
            self.Vpk_count = Counter(self.Vpk)
            self.Vdb_count = Counter(self.Vdb_range)
        else:
            print('Data units not yet converted! \n'
                  'Attempting unit conversion... \n'
                  'Please try again.')
            self.units()

    # Utilities ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def valCh(self, valname: str):
        """
        Determines determines the requested value and label for the grape object
        Args:
            valname (str): The name of the value to check.
        Returns:
            tuple: A tuple containing the value (or None if not found) and the corresponding label.
        """
        label = 'None'

        if valname in FNAMES:
            vals = self.f_range
            label = FLABEL
        elif valname in VNAMES:
            vals = self.Vpk
            label = VLABEL
        elif valname in PNAMES:
            vals = self.Vdb_range
            label = PLABEL
        else:
            vals = None

        return vals, label

    # Plot Elements ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def sunPosOver(self, fSize: int, end_times: bool = False, local: bool = False):
        """
        Plots overlay of the grape's sunrise, sunset, and solar noon features over the currently open matplotlib plot

        :param fSize: Font size of marker legend text
        :param end_times: Setting for displaying vertical markers for TX or RX
        :param local: Setting for offsetting markers to midpoint local time
        :return: vertical lines on current graph
        """

        loc_offset = self.t_offset if local else 0

        if end_times:
            RXsr = conv_time(self.RXsuntimes['sunrise'])
            RXsn = conv_time(self.RXsuntimes['solar_noon'])
            RXss = conv_time(self.RXsuntimes['sunset'])

            RXsrMark = plt.axvline(x=RXsr, color='y', linewidth=3, linestyle='dashed', alpha=0.3)
            RXsnMark = plt.axvline(x=RXsn, color='g', linewidth=3, linestyle='dashed', alpha=0.3)
            RXssMark = plt.axvline(x=RXss, color='b', linewidth=3, linestyle='dashed', alpha=0.3)

            TXsr = conv_time(self.TXsuntimes['sunrise'])
            TXsn = conv_time(self.TXsuntimes['solar_noon'])
            TXss = conv_time(self.TXsuntimes['sunset'])

            TXsrMark = plt.axvline(x=TXsr, color='y', linewidth=3, linestyle='dashed', alpha=0.3)
            TXsnMark = plt.axvline(x=TXsn, color='g', linewidth=3, linestyle='dashed', alpha=0.3)
            TXssMark = plt.axvline(x=TXss, color='b', linewidth=3, linestyle='dashed', alpha=0.3)

        Bsr = conv_time(self.Bsuntimes['sunrise'])
        Bsn = conv_time(self.Bsuntimes['solar_noon'])
        Bss = conv_time(self.Bsuntimes['sunset'])

        BsrMark = plt.axvline(x=Bsr, color='y', linewidth=3, linestyle='dashed')
        BsnMark = plt.axvline(x=Bsn, color='g', linewidth=3, linestyle='dashed')
        BssMark = plt.axvline(x=Bss, color='b', linewidth=3, linestyle='dashed')

        times = np.array([Bsr, Bsn, Bss]) + loc_offset
        for i, time in enumerate(times):
            if time < 0:
                times[i] = time + 24

        utc_string = ' UTC' if not local else ''
        plt.legend([BsrMark, BsnMark, BssMark], ["Sunrise: " + str(round_down(times[0], 2)) + utc_string,
                                                 "Solar Noon: " + str(round_down(times[1], 2)) + utc_string,
                                                 "Sunset: " + str(round_down(times[2], 2)) + utc_string],
                   fontsize=fSize,
                   loc='upper left')

    def powerOver(self, ax1):
        Vdbrange = self.Vdb_range if not self.filtered else self.Vdb_range_filt
        fSize = self.plot_settings['fontsize']
        labelpad = self.plot_settings['pad']

        alt_color = 'r'

        if self.axcount > 0:
            ax2 = ax1.twinx()
        else:
            ax2 = ax1

        ax2.plot(self.t_range, Vdbrange, alt_color, linewidth=2)
        ax2.set_ylabel(PLABEL, color=alt_color, fontsize=fSize)
        ax2.set_ylim(-80, 0)
        
        if self.axcount >= 0:
            ax2.tick_params(axis='y', colors=alt_color, labelsize=fSize - 2, direction='out', pad=labelpad)
            ax2.spines['right'].set_color(alt_color)
        if self.axcount > 1:
            additional_padding = 1 + 0.05 * self.axcount
            ax2.spines['right'].set_position(('axes', additional_padding))
        
        self.axcount += 1

        return ax2

    def szaOver(self, ax1):
        fSize = self.plot_settings['fontsize']
        labelpad = self.plot_settings['pad']

        alt_color = 'c'

        if self.axcount > 0:
            ax2 = ax1.twinx()
        else:
            ax2 = ax1

        ax2.plot(self.t_range, self.zentrace, alt_color, linewidth=2)
        ax2.set_ylabel('Solar Zenith Angle (°)', color=alt_color, fontsize=fSize)
        ax2.set_ylim(0, 180)
        
        if self.axcount >= 0:
            ax2.tick_params(axis='y', colors=alt_color, labelsize=fSize - 2, direction='out', pad=labelpad)
            ax2.spines['right'].set_color(alt_color)
        if self.axcount > 1:
            additional_padding = 1 + 0.05 * self.axcount
            ax2.spines['right'].set_position(('axes', additional_padding))

        self.axcount += 1

        return ax2

    def dopOver(self, ax1, **kwargs):
        frange = self.f_range if not self.filtered else self.f_range_filt
        fSize = self.plot_settings['fontsize']
        labelpad = self.plot_settings['pad']
        ylim = kwargs.get('ylim', [-1, 1])

        alt_color = 'k'

        if self.axcount > 0:
            ax2 = ax1.twinx()
        else:
            ax2 = ax1

        ax2.plot(self.t_range, frange, alt_color, linewidth=2)
        ax2.set_ylabel(FLABEL, color=alt_color, fontsize=fSize)
        ax2.set_ylim(ylim)

        if self.axcount >= 0:
            ax2.tick_params(axis='y', colors=alt_color, labelsize=fSize - 2, direction='out', pad=labelpad)
            ax2.spines['right'].set_color(alt_color)
        if self.axcount > 1:
            additional_padding = 1 + 0.05 * self.axcount
            ax2.spines['right'].set_position(('axes', additional_padding))

        self.axcount += 1

        return ax2
    
    def timeAxis(self, ax1, **kwargs):
        """
        Sets the time axis for the plot

        :param ax1: The axis to set the time axis for
        :return: The axis with the time axis set
        """

        self.axcount = 0

        fSize = self.plot_settings['fontsize']
        labelpad = self.plot_settings['pad']

        ax1.set_xlim(0, 24)

        tmp = np.arange(0, 25, 2)
        if kwargs.get('local', False):
            # If time local to midpoint is requested
            ax1.set_xlabel('Midpoint Local Time (Hours)', fontsize=fSize)

            xrange = tmp + self.t_offset
            for i, x in enumerate(xrange):
                if x < 0:
                    xrange[i] = x + 24
        else:
            # If local time is not requested (default UT)
            ax1.set_xlabel('Time UT (Hours)', fontsize=fSize)

            xrange = tmp

        ax1.set_xticks(tmp, labels=xrange.astype(int))
        ax1.tick_params(axis='x', labelsize=fSize - 2, direction='out', pad=labelpad)
        ax1.grid(axis='x', alpha=1)
        ax1.grid(axis='y', alpha=0.5)

        return ax1

    # Plots ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def dopPowPlot(self, figname: str, ylim: list = None, fSize: int = 22, axis2: str = None, val: str = None,
                   end_times: bool = False, local: bool = False, **kwargs):
        """
        Plot the doppler shift and relative power over time of the signal

        :param figname: Filename for the produced .png plot image
        :param ylim: Provide a python list containing minimum and maximum doppler shift in Hz
         for the data (default = [-1, 1])
        :param fSize: Font size to scale all plot text (default = 22)
        :param axis2: String hint for second axis plot
        :param val: String hint for primary axis plot ('pwr' for power, default to doppler)
        :param end_times: Setting for displaying end time suntime markers
        :param local: Setting for displaying time axis as midpoint local time

        :return: .png plot into local repository
        """
        # Checks if the data has been converted to the proper units
        if not self.converted:
            raise('Data units not yet converted! \n'
                  'Please try again.')
        
        # Initializes figure and axis
        fig = plt.figure(figsize=(19, 10))
        ax1 = fig.add_subplot(111)

        # Sets up time axis
        self.timeAxis(ax1, **kwargs)

        # Toggles sun position overlay
        spo = kwargs.get('SPO', False)
        if spo: self.sunPosOver(fSize, end_times=end_times, local=local)

        # Toggles additional axis overlays
        if kwargs.get('dop', False):
            self.dopOver(ax1, ylim=ylim)
        if kwargs.get('pwr', False):
            self.powerOver(ax1)
        if kwargs.get('sza', False):
            self.szaOver(ax1)

        # Selects title for the plot
        lbl_split = FLABEL
        figtitle(ax1, lbl_split, **self.plot_settings)

        # Saves the plot to the local repository
        if kwargs.get('save', True):
            plt.savefig(str(figname) + '.png', dpi=300, orientation='landscape')
            print(f'Plot saved to {figname}.png \n')

        return ax1

    def distPlot(self, valname, figname):
        """
        Plot the distribution of a value loaded into the grape object

        :param valname: Hint for the system to determine which value to plot the distribution of (doppler shift, power, etc.)
        :param figname: Name for the produced .png plot file
        :return: .png plot into local repository
        """

        if self.converted:
            vals, label = self.valCh(valname)

            if vals is not None:
                binlims = None
                if valname in FNAMES:
                    binlims = np.arange(-2.5, 2.6, 0.1)

                fSize = 22
                fig = plt.figure(figsize=(19, 10))  # inches x, y with 72 dots per inch
                ax1 = fig.add_subplot(111)
                ax1.hist(vals, color='r', edgecolor='k', bins=binlims) if (binlims is not None) else ax1.hist(vals,
                                                                                                              color='r',
                                                                                                              edgecolor='k')
                ax1.set_xlabel(label, fontsize=fSize)
                ax1.set_ylabel('Counts, N', color='r', fontsize=fSize)
                ax1.grid(axis='x', alpha=1)
                ax1.grid(axis='y', alpha=0.5)
                ax1.tick_params(axis='x',
                                labelsize=fSize - 2)  # ax1.set_xlim([-2.5, 2.5])  # 0.1Hz Bins (-2.5Hz to +2.5Hz)
                ax1.tick_params(axis='y', labelsize=fSize - 2)

                if valname in FNAMES:
                    pl.xlim([-1, 1])  # Doppler Shift Range
                    pl.xticks(np.arange(-1, 1.1, 0.1))

                lbl_split = label.split(',')[0]
                figtitle(ax1, lbl_split, **self.plot_settings)
                
                plt.savefig(str(figname) + '.png', dpi=300, orientation='landscape')
                plt.show()
                plt.close()
            else:
                print("Please provide a valid valname!")
        else:
            print('Data units not yet converted! \n'
                  'Attempting unit conversion... \n'
                  'Please try again.')
            self.units()

    def distPlots(self, valname, dirname='dshift_dist_plots', figname='dshift_dist_plot', minBinLen=5, sel=None,
                  fSize=22):
        """
        Plot the distributions of a grape object value separated by specified time bins

        :param fSize: Font size to scale all plot text (default = 22)
        :param valname: string value dictating value selection (eg. 'f', 'v', or 'db')
        :param dirname: string value for the name of the local directory where the plots will be saved
        :param figname: string value for the beginning of each image filename
        :param minBinLen: int value for the length of each time bin in minutes (should be a factor of 60)
        :return: .png plot into local repository
        """

        if self.converted:
            if valname in FNAMES:
                vals = self.f_range
            elif valname in VNAMES:
                vals = self.Vpk
            elif valname in PNAMES:
                vals = self.Vdb_range
            else:
                vals = None

            if vals is not None:

                secrange, minrange = mblHandle(minBinLen)

                # Make subsections and begin plot generation
                subranges = []  # contains equally sized ranges of data

                index = 0
                while not index > len(vals):
                    subranges.append(vals[index:index + secrange])
                    index += secrange

                hours = []  # contains 24 hour chunks of data

                index = 0
                while not index > len(subranges):
                    hours.append(subranges[index:index + minrange])
                    index += minrange

                # initializes directory on local path if it does not exist
                if not os.path.exists(dirname):
                    os.mkdir(dirname)

                if not sel:
                    count = 0
                    indexhr = 0
                    for hour in hours:
                        print('\nResolving hour: ' + str(indexhr) + ' ('
                              + str(floor((indexhr / len(hours)) * 100)) + '% complete) \n'
                              + '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

                        index = 0
                        for srange in hour:
                            print('Resolving subrange: ' + str(index) + ' ('
                                  + str(floor((index / len(hour)) * 100)) + '% complete)')

                            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            # Plot the subsections
                            binlims = np.arange(-2.5, 2.6, 0.1)

                            fig = plt.figure(figsize=(19, 10))  # inches x, y with 72 dots per inch
                            ax1 = fig.add_subplot(111)
                            ax1.hist(srange, color='r', edgecolor='k', bins=binlims)
                            ax1.set_xlabel('Doppler Shift, Hz')
                            ax1.set_ylabel('Counts, N', color='r')
                            ax1.set_xlim([-2.5, 2.5])  # 0.1Hz Bins (-2.5Hz to +2.5Hz)
                            ax1.set_xticks(binlims[::2])

                            plt.title('WWV 10 MHz Doppler Shift Distribution Plot \n'
                                      'Hour: ' + str(indexhr) + ' || 5-min bin: ' + str(index) + ' \n'  # Title (top)
                                                                                                 'Node: N0000020    Gridsquare: FN20vr \n'
                                                                                                 'Lat=40.40.742018  Long=-74.178975 Elev=50M \n'
                                      + self.date + ' UTC',
                                      fontsize='10')

                            plt.savefig(str(dirname) + '/' + str(figname) + str(count) + '.png', dpi=300,
                                        orientation='landscape')
                            count += 1

                            plt.close()

                            index += 1

                        indexhr += 1
                else:
                    hrSel = sel[0]
                    binSel = sel[1]

                    hours = hours[hrSel][binSel]

                    binlims = np.arange(-2.5, 2.6, 0.1)
                    fig = plt.figure(figsize=(19, 10))  # inches x, y with 72 dots per inch

                    fSize = fSize
                    ax1 = fig.add_subplot(111)
                    ax1.hist(hours, color='r', edgecolor='k', bins=binlims)
                    ax1.set_xlabel('Doppler Shift, Hz', fontsize=fSize)
                    ax1.set_ylabel('Counts, N', color='r', fontsize=fSize)
                    ax1.set_xlim([-1, 1])  # Doppler Shift Range
                    ax1.set_xticks(np.arange(-1, 1.1, 0.1))
                    ax1.tick_params(axis='x',
                                    labelsize=fSize - 2)  # ax1.set_xlim([-2.5, 2.5])  # 0.1Hz Bins (-2.5Hz to +2.5Hz)
                    # ax1.set_xticks(binlims[::2])
                    ax1.tick_params(axis='y', labelsize=fSize - 2)

                    plt.title('WWV 10 MHz Doppler Shift Distribution Plot \n'
                              'Hour: ' + str(hrSel) + ' || 5-min bin: ' + str(binSel) + ' \n'  # Title (top)
                              # 'Node: N0000020    Gridsquare: FN20vr \n'
                              # 'Lat=40.40.742018  Long=-74.178975 Elev=50M \n'
                              + self.date + ' UTC',
                              fontsize=fSize)

                    plt.savefig(str(figname) + '.png', dpi=300,
                                orientation='landscape')
                    plt.close()


            else:
                print("Please provide a valid valname!")
        else:
            print('Data units not yet converted! \n'
                  'Attempting unit conversion... \n'
                  'Please try again.')
            self.units()

    def distPlotFit(self, valname, figname):
        """
        Produces a fitted histogram for the entire day's worth of data (for the specified value)

        :param valname: string value dictating value selection (eg. 'f', 'v', or 'db')
        :param figname: string value for the beginning of the image filename
        :return: .png plot into local repository
        """

        if self.converted:
            if valname in FNAMES:
                vals = self.f_range
            elif valname in VNAMES:
                vals = self.Vpk
            elif valname in PNAMES:
                vals = self.Vdb_range
            else:
                vals = None

            if vals is not None:
                binlims = np.arange(-2.5, 2.6, 0.1)
                pl.figure(figsize=(19, 10))  # inches x, y with 72 dots per inch

                f = Fitter(vals, bins=binlims, distributions='common')
                f.fit()
                summary = f.summary()
                print(summary)
                f.hist()

                fSize = 22
                pl.xlabel('Doppler Shift, Hz', fontsize=fSize)
                pl.ylabel('Normalized Counts', fontsize=fSize)
                pl.xlim([-1, 1])  # Doppler Shift Range
                pl.xticks(np.arange(-1, 1.1, 0.1))

                pl.legend(fontsize=fSize)
                pl.grid(axis='x', alpha=1)
                pl.grid(axis='y', alpha=0.5)
                pl.tick_params(axis='x',
                               labelsize=fSize - 2)  # ax1.set_xlim([-2.5, 2.5])  # 0.1Hz Bins (-2.5Hz to +2.5Hz)
                pl.tick_params(axis='y', labelsize=fSize - 2)

                # pl.xlim([-2.5, 2.5])  # Doppler Shift Range
                # pl.xticks(binlims[::2])

                pl.title('Fitted Doppler Shift Distribution \n'
                         # 'Node: N0000020    Gridsquare: FN20vr \n'
                         # 'Lat=40.40.742018  Long=-74.178975 Elev=50M \n'
                         + self.date + ' UTC',
                         fontsize=fSize)
                pl.savefig(str(figname) + '.png', dpi=300,
                           orientation='landscape')
                pl.show()
                pl.close()

            else:
                print("Please provide a valid valname!")
        else:
            print('Data units not yet converted! \n'
                  'Attempting unit conversion... \n'
                  'Please try again.')
            self.units()

    def distPlotsFit(self, valname, dirname='dshift_dist_plots', figname='dshift_dist_plot',
                     minBinLen=5, sel=None, fSize=22):
        """
        Produces a series of fitted histograms at specified minute intervals

        :param sel: Provide an integer list of the form [hour, bin] to plot that singular selection of data
        :param fSize: Font size to scale all plot text (default = 22)
        :param valname: string value dictating value selection (eg. 'f', 'v', or 'db')
        :param dirname: string value for the name of the local directory where the plots will be saved
        :param figname: string value for the beginning of each image filename
        :param minBinLen: int value for the length of each time bin in minutes (should be a factor of 60)
        :return: .png plot into dirname repository
        """

        if self.converted:
            if valname in FNAMES:
                vals = self.f_range
            elif valname in VNAMES:
                vals = self.Vpk
            elif valname in PNAMES:
                vals = self.Vdb_range
            else:
                vals = None

            if vals is not None:

                secrange, minrange = mblHandle(minBinLen)

                # Make subsections and begin plot generation
                subranges = []  # contains equally sized ranges of data

                index = 0
                while not index > len(vals):
                    subranges.append(vals[index:index + secrange])
                    index += secrange

                hours = []  # contains 24 hour chunks of data

                index = 0
                while not index > len(subranges):
                    hours.append(subranges[index:index + minrange])
                    index += minrange

                # initializes directory on local path if it does not exist
                if not os.path.exists(dirname):
                    os.mkdir(dirname)

                self.bestFits = []

                if not sel:
                    count = 0
                    indexhr = 0
                    for hour in hours:
                        print('\nResolving hour: ' + str(indexhr) + ' ('
                              + str(floor((indexhr / len(hours)) * 100)) + '% complete) \n'
                              + '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

                        index = 0
                        for srange in hour:
                            print('Resolving subrange: ' + str(index) + ' ('
                                  + str(floor((index / len(hour)) * 100)) + '% complete)')

                            binlims = np.arange(-2.5, 2.6, 0.1)
                            pl.figure(figsize=(19, 10))  # inches x, y with 72 dots per inch

                            f = Fitter(srange, bins=binlims, timeout=10, distributions='common')
                            f.fit()
                            summary = f.summary()
                            print(summary)
                            self.bestFits.append(f.get_best())

                            f.hist()

                            fSize = fSize
                            pl.xlabel('Doppler Shift, Hz', fontsize=fSize)
                            pl.ylabel('Normalized Counts', fontsize=fSize)
                            pl.xlim([-2.5, 2.5])  # Doppler Shift Range
                            pl.xticks(binlims[::2], fontsize=fSize / 1.4)
                            pl.yticks(fontsize=fSize / 1.4)

                            pl.legend(fontsize=fSize)

                            pl.title('Fitted Doppler Shift Distribution \n'  # Title (top)
                                     'Hour: ' + str(indexhr) +
                                     ' || 5-min bin: ' + str(index) + ' \n'
                                     # 'Node: N0000020    Gridsquare: FN20vr \n'
                                     # 'Lat=40.40.742018  Long=-74.178975 Elev=50M \n'
                                     + self.date + ' UTC',
                                     fontsize=fSize)

                            pl.savefig(str(dirname) + '/' + str(figname) + '_' + str(count) + '.png', dpi=300,
                                       orientation='landscape')

                            pl.close()

                            count += 1
                            index += 1

                        indexhr += 1
                else:
                    hrSel = sel[0]
                    binSel = sel[1]

                    hours = hours[hrSel][binSel]

                    binlims = np.arange(-2.5, 2.6, 0.1)
                    pl.figure(figsize=(19, 10))  # inches x, y with 72 dots per inch

                    f = Fitter(hours, bins=binlims, timeout=10, distributions='common')
                    f.fit()
                    summary = f.summary()
                    print(summary)
                    self.bestFits.append(f.get_best())

                    f.hist()

                    fSize = fSize
                    pl.xlabel('Doppler Shift, Hz', fontsize=fSize)
                    pl.ylabel('Normalized Counts', fontsize=fSize)
                    pl.xlim([-1, 1])  # Doppler Shift Range
                    pl.xticks(np.arange(-1, 1.1, 0.1), fontsize=fSize / 1.4)
                    # pl.xlim([-2.5, 2.5])  # Doppler Shift Range
                    # pl.xticks(binlims[::2], fontsize=fSize/1.4)
                    pl.yticks(fontsize=fSize / 1.4)

                    pl.legend(fontsize=fSize)

                    pl.title('Fitted Doppler Shift Distribution \n'  # Title (top)
                             'Hour: ' + str(hrSel) +
                             ' || 5-min bin: ' + str(binSel) + ' \n'
                             # 'Node: N0000020    Gridsquare: FN20vr \n'
                             # 'Lat=40.40.742018  Long=-74.178975 Elev=50M \n'
                             + self.date + ' UTC',
                             fontsize=fSize)

                    pl.show()

                    pl.savefig(str(figname) + '.png', dpi=300,
                               orientation='landscape')

                    pl.close()

            else:
                print("Please provide a valid valname!")
        else:
            print('Data units not yet converted! \n'
                  'Attempting unit conversion... \n'
                  'Please try again.')
            self.units()
 
    def bestFitsPlot(self, valname, figname, minBinLen=5, ylim=None, fSize=22):
        """
        Over-plots the best fits resolved for each specified time bin with the doppler shift data for the day

        :param fSize: Font size to scale all plot text (default = 22)
        :param valname: string value dictating value selection (eg. 'f', 'v', or 'db')
        :param figname: string value for the beginning of each image filename
        :param minBinLen: int value for the length of each time bin in minutes (should be a factor of 60)
        :param ylim: Provide a python list containing minimum and maximum doppler shift in Hz
         for the data (default = [-1, 1])
        :return: .png plot into local repository
        """

        if self.converted:
            if valname in FNAMES:
                vals = self.f_range
            elif valname in VNAMES:
                vals = self.Vpk
            elif valname in PNAMES:
                vals = self.Vdb_range
            else:
                vals = None

            if vals is not None:

                # Make subsections and begin plot generation
                subranges = []  # contains equally sized ranges of data

                secrange, minrange = mblHandle(minBinLen)

                index = 0
                while not index > len(vals):
                    subranges.append(vals[index:index + secrange])
                    index += secrange

                hours = []  # contains 24 hour chunks of data

                index = 0
                while not index > len(subranges):
                    hours.append(subranges[index:index + minrange])
                    index += minrange

                self.bestFits = []

                indexhr = 0
                # for hour in hours:
                print('\nResolving Hours:\n')
                for hour in tqdm(hours):
                    index = 0
                    for srange in hour:
                        binlims = np.arange(-2.5, 2.6, 0.1)

                        f = Fitter(srange, bins=binlims, timeout=10, distributions='common')
                        f.fit()
                        self.bestFits.append(f.get_best())

                        index += 1

                    indexhr += 1

                frange = self.f_range if not self.filtered else self.f_range_filt
                prange = self.Vdb_range if not self.filtered else self.Vdb_range_filt

                yrange = frange if (valname in FNAMES) else prange

                # Sets y range of plot
                if ylim is None:
                    # truncates maximum and minimum values
                    bottom = round_down(yrange.min(), 1)
                    top = round_up(yrange.max(), 1)

                    # snapping to +- 1, 1.2, 1.5 or 2 for doppler vals
                    if valname in FNAMES:
                        if bottom > -2:
                            bottom = -1 if (bottom > -1) else (
                                -1.2 if (bottom > -1.2) else (-1.5 if (bottom > -1.5) else -2))
                        if top < 2:
                            top = 1 if (top < 1) else (1.2 if (top < 1.2) else (1.5 if (top < 1.5) else 2))

                    ylim = [bottom, top]

                fig = plt.figure(figsize=(19, 10), layout='constrained')  # inches x, y with 72 dots per inch
                ax1 = fig.add_subplot(111)
                ax1.plot(self.t_range, yrange, color='k')  # color k for black
                ax1.set_xlabel('UTC Hour', fontsize=fSize)
                ax1.set_ylabel('Doppler shift, Hz', fontsize=fSize)
                ax1.set_xlim(0, 24)  # UTC day
                ax1.set_xticks(np.arange(0, 25, 2))
                ax1.set_ylim(ylim)  # -1 to 1 Hz for Doppler shift
                ax1.grid(axis='x', alpha=1)
                labelpad = 20
                ax1.tick_params(axis='x', labelsize=fSize - 2, direction='out', pad=labelpad)  # ax1.set_xlim([-2.5, 2.5])  # 0.1Hz Bins (-2.5Hz to +2.5Hz)
                ax1.tick_params(axis='y', labelsize=fSize - 2, direction='out', pad=labelpad)

                rng = np.arange(0, len(self.bestFits))
                fitTimeRange = (rng / len(self.bestFits)) * 24
                self.bestFits = [list(i.keys())[0] for i in self.bestFits]

                alt_color = 'c'
                ax2 = ax1.twinx()
                ax2.plot(self.t_range, self.zentrace, alt_color, linewidth=2)
                ax2.set_ylabel('Solar Zenith Angle (°)', color=alt_color, fontsize=fSize)
                ax2.set_ylim(0, 180)
                ax2.tick_params(axis='y', colors=alt_color, labelsize=fSize - 2, direction='out', pad=labelpad)
                ax2.spines['right'].set_color(alt_color)

                alt_color = 'r'
                ax3 = ax1.twinx()
                ax3.scatter(fitTimeRange, self.bestFits, color=alt_color, linewidth=2)
                ax3.set_ylabel('Best Fit PDF', color=alt_color, fontsize=fSize)
                ax3.grid(axis='y', alpha=0.5)
                ax3.tick_params(axis='y', colors=alt_color, labelsize=fSize - 2, direction='out', pad=labelpad)
                ax3.spines['right'].set_position(('axes', 1.15))
                ax3.spines['right'].set_color(alt_color)

                figtitle(ax1, 'Doppler Shift Distribution PDFs', **self.plot_settings)

                plt.savefig(str(figname) + '.png', dpi=300, orientation='landscape')
                plt.show()
                plt.close()

            else:
                print("Please provide a valid valname!")
        else:
            print('Data units not yet converted! \n'
                  'Attempting unit conversion... \n'
                  'Please try again.')
            self.units()

    def dopRtPlot(self, figname, ylim=None, fSize=22):
        """
        Plot the doppler shift and relative power over time of the signal

        :param fSize: Font size to scale all plot text (default = 22)
        :param ylim: Provide a python list containing minimum and maximum doppler shift in Hz
         for the data (default = [-1, 1])
        :param figname: Filename for the produced .png plot image
        :return: .png plot into local repository
        """

        if ylim is None:
            ylim = [-1, 1]

        if not self.converted:
            raise('Data units not yet converted! \n'
                  'Please try again.')

        frange = self.f_range if not self.filtered else self.f_range_filt

        fig = plt.figure(figsize=(19, 10), layout='constrained')  # inches x, y with 72 dots per inch

        ax1 = fig.add_subplot(111)
        ax1.plot(self.t_range, frange, 'k', linewidth=2)  # color k for black

        label = FLABEL
        ax1.set_xlabel('Time UT (Hours)', fontsize=fSize)
        ax1.set_ylabel(label, fontsize=fSize)
        ax1.set_xlim(0, 24)  # UTC day
        ax1.set_ylim(ylim)  # -1 to 1 Hz for Doppler shift
        tmp = np.arange(0, 25, 2)
        ax1.set_xticks(tmp, labels=tmp.astype(int))
        labelpad = 20
        ax1.tick_params(axis='x', labelsize=fSize - 2, direction='out', pad=labelpad)  # ax1.set_xlim([-2.5, 2.5])  # 0.1Hz Bins (-2.5Hz to +2.5Hz)
        ax1.tick_params(axis='y', labelsize=fSize - 2, direction='out', pad=labelpad)  # ax1.set_xlim([-2.5, 2.5])  # 0.1Hz Bins (-2.5Hz to +2.5Hz)
        ax1.grid(axis='x', alpha=1)
        ax1.grid(axis='y', alpha=0.5)

        alt_color = 'c'
        ax2 = ax1.twinx()
        ax2.plot(self.t_range, self.zentrace, alt_color, linewidth=2)
        ax2.set_ylabel('Solar Zenith Angle (°)', color=alt_color, fontsize=fSize)
        ax2.set_ylim(0, 180)
        ax2.tick_params(axis='y', colors=alt_color, labelsize=fSize - 2, direction='out', pad=labelpad)
        ax2.spines['right'].set_color(alt_color)

        lbl_split = label.split(',')[0]
        figtitle(ax1, lbl_split, **self.plot_settings)

        styles = ['y-', 'g-', 'b-', 'r-', 'y--', 'g--', 'b--', 'r--']

        rt_data_dir = 'C:/Users/sabas/Documents/GitHub/PHARvis/export_data/'
        files = os.listdir(rt_data_dir)

        alt_color = 'm'
        ax3 = ax1.twinx()
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='y', colors=alt_color, labelsize=fSize - 2, direction='out', pad=labelpad)
        ax3.spines['right'].set_position(('axes', 1.15))
        ax3.spines['right'].set_color(alt_color)

        for f in files:
            df = pd.read_csv(rt_data_dir + f, header=None)
            hour_range = np.arange(0, df.shape[0], 1)

            lines = []
            for i in range(0, df.shape[1]):
                l = ax3.plot(hour_range, df[i], styles[i], linewidth=2)
                lines.append(l)

            plt.legend(["1-hop", "2-hop", "3-hop", "4-hop"], fontsize=fSize, loc='upper left')
            file_label = f.split('_percentages')[0]
            R12_number = file_label.split('_')[0]
            mode = file_label.split('_')[1]
            ax3.set_ylabel(f'% of Rays Recieved (R12 = {R12_number} / {mode}-mode)', color=alt_color, fontsize=fSize)

            plt.savefig(str(figname) + "_" + file_label + '.png', dpi=300, orientation='landscape')

            for l in lines:
                l[0].remove()

        plt.close()

    def rtBouncePlot(self, figname, ylim=None, fSize=22):
        """
        Plot the reflection point of the raytraced rays over time

        :param fSize: Font size to scale all plot text (default = 22)
        :param ylim: Provide a python list containing minimum and maximum doppler shift in Hz
         for the data (default = [-1, 1])
        :param figname: Filename for the produced .png plot image
        :return: .png plot into local repository
        """

        if ylim is None:
            ylim = [0, 400]

        if not self.converted:
            raise('Data units not yet converted! \n'
                  'Please try again.')

        fig = plt.figure(figsize=(19, 10), layout='constrained')  # inches x, y with 72 dots per inch

        ax1 = fig.add_subplot(111)

        styles = ['y-', 'g-', 'b-', 'r-', 'y--', 'g--', 'b--', 'r--']

        maxheight_data_dir = 'C:/Users/sabas/Documents/GitHub/PHARvis/export_data_max_heights_57_O/'
        initialelev_data_dir = 'C:/Users/sabas/Documents/GitHub/PHARvis/export_data_initial_elevs_57_O/'

        maxheight_files = os.listdir(maxheight_data_dir)
        initialelev_files = os.listdir(initialelev_data_dir)

        nfiles = len(maxheight_files)

        styles = ['yo', 'go', 'bo', 'ro']
        
        p1 = ax1.plot(0,0,styles[0],markersize=10)
        p2 = ax1.plot(0,0,styles[1],markersize=10)
        p3 = ax1.plot(0,0,styles[2],markersize=10)
        p4 = ax1.plot(0,0,styles[3],markersize=10)
        plt.legend(["1-hop", "2-hop", "3-hop", "4-hop"], fontsize=fSize, loc='upper left')
        p1[0].remove()
        p2[0].remove()
        p3[0].remove()
        p4[0].remove()

        for i, f_i in enumerate(range(0, nfiles)):  # Per hop
            mh_df = pd.read_csv(maxheight_data_dir + maxheight_files[f_i], header=None)
            ie_df = pd.read_csv(initialelev_data_dir + initialelev_files[f_i], header=None)
            
            hour_range = np.arange(0, mh_df.shape[0], 1)
            max_elev = 50
            scale = 50
            for hour in hour_range:
                for k in range(0, mh_df.shape[1]):
                    if mh_df[k][hour] > 0:
                        ie_scale = ((ie_df[k][hour] + 1) / max_elev) * scale
                        ax1.plot(hour, mh_df[k][hour], styles[i], markersize=ie_scale)

        f = maxheight_files[0]
        file_label = f.split('max_heights_')[1]
        R12_number = file_label.split('_')[0]
        mode = file_label.split('_')[1][0]
        ax1.set_ylabel(f'Reflection Altitude (km) (R12 = {R12_number} / {mode}-mode)', fontsize=fSize)

        ax1.set_xlabel('Time UT (Hours)', fontsize=fSize)
        ax1.set_xlim(0, 24)  # UTC day
        ax1.set_ylim(ylim)  # -1 to 1 Hz for Doppler shift
        tmp = np.arange(0, 25, 2)
        ax1.set_xticks(tmp, labels=tmp.astype(int))
        labelpad = 20
        ax1.tick_params(axis='x', labelsize=fSize - 2, direction='out', pad=labelpad)  # ax1.set_xlim([-2.5, 2.5])  # 0.1Hz Bins (-2.5Hz to +2.5Hz)
        ax1.tick_params(axis='y', labelsize=fSize - 2, direction='out', pad=labelpad)  # ax1.set_xlim([-2.5, 2.5])  # 0.1Hz Bins (-2.5Hz to +2.5Hz)
        ax1.grid(axis='x', alpha=1)
        ax1.grid(axis='y', alpha=0.5)

        alt_color = 'c'
        ax2 = ax1.twinx()
        ax2.plot(self.t_range, self.zentrace, alt_color, linewidth=2)
        ax2.set_ylabel('Solar Zenith Angle (°)', color=alt_color, fontsize=fSize)
        ax2.set_ylim(0, 180)
        ax2.tick_params(axis='y', colors=alt_color, labelsize=fSize - 2, direction='out', pad=labelpad)
        ax2.spines['right'].set_color(alt_color)

        figtitle(ax1, 'Reflection Point', **self.plot_settings)
        
        plt.savefig(str(figname) + "_" + file_label.split('.csv')[0] + '.png', dpi=300, orientation='landscape')
        plt.close()

class GrapeHandler:

    # Util ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, dirnames, filt=False, comb=True, med=False, tShift=True, n=1, **kwargs):
        """
        Dynamically creates and manipulates multiple instances of the Grape object using a specified data directory

        :param dirnames: list [] of string values for the local directories in which the intended data files (.csv) are located
        :param filt: boolean value dictating whether or not each grape is filtered upon loading (default False)
        :param comb: boolean indicating if the data should be combined for certain GrapeHandler functionality
        :param tShift: boolean indicating if the data should be time shifted to align sunrises
        :param n: subsampling term which allows every nth datapoint to be loaded into each grape
        """
        self.grapes = []
        self.valscomb = []
        self.timecomb = []
        self.dMeds = []
        self.nMeds = []
        self.month = None
        self.valslength = 0
        self.bestgid = 0
        self.bestFits = None
        self.valid = True
        self.ss_factor = n

        for directory in dirnames:
            if os.path.exists(directory):
                pass
            else:
                self.valid = False
                break

        if self.valid:
            self.load(dirnames, filt, comb, med, tShift, n)
        else:
            print('One or more of the provided directories do not exist on the local path! \n'
                  'Please try again. \n')

    def load(self, dirnames, filt, comb, med, tShift, n):
        """
        Script to load selected grape data into GrapeHandler object

        :param dirnames: directory names for the targeted files to be loaded
        :param filt: boolean indicating if the data will be filtered
        :param comb: boolean indicating if the data should be combined for certain GrapeHandler functionality
        :param med: boolean indicating if the day / night medians should be calculated for each Grape
        :param tShift: boolean indicating if the data should be time shifted to align sunrises
        :param n: subsampling term which allows every nth datapoint to be loaded into each grape
        :return:
        """

        filenames = []
        for directory in dirnames:
            # iterate over files in that directory
            for filename in os.scandir(directory):
                if filename.is_file():
                    filenames.append('./' + directory + '/' + filename.name)

        print('\nLoading Grapes:\n')
        for filename in tqdm(filenames):
            g = Grape(filename, filt=filt, med=med, n=n)
            if g.loaded:
                self.grapes.append(g)

        if len(self.grapes) != 0:
            self.month = self.grapes[0].date[0:7]  # Attributes the date of the first grape

            if comb:
                firstSunrise = conv_time(self.grapes[0].Bsuntimes['sunrise'], 's')

                for gid, grape in enumerate(self.grapes):
                    vals = grape.getFiltTFPr() if (
                            filt is True) else grape.getTFPr()  # get time, freq and power from grape
                    t = np.array(vals[0])
                    f = np.array(vals[1])

                    if tShift:
                        thisSunrise = conv_time(grape.Bsuntimes['sunrise'], 's')
                        sunDiff = int(thisSunrise - firstSunrise)

                        t = np.roll(t, sunDiff)
                        f = np.roll(f, sunDiff)

                        if sunDiff > 0:
                            t[0:sunDiff] = 0
                            f[0:sunDiff] = 0
                        if sunDiff < 0:
                            t[sunDiff::] = 0
                            f[sunDiff::] = 0

                    # An array of arrays
                    self.timecomb.append(t)
                    self.valscomb.append(f)

                    if len(f) > self.valslength:
                        self.valslength = len(f)
                        self.bestgid = gid

                print('GrapeHandler loaded with combvals \n')
            else:
                print('GrapeHandler loaded without combvals (no multGrapeDist, medTrend) \n')

        else:
            self.valid = False
            print('GrapeHandler not loaded (no valid grapes) \n')

    # Plots ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def multGrapeDistPlot(self, figname):
        """
        Plots the combined histogram for all Grapes loaded into the GrapeHandler

        :param figname: string value to act as the name for the produced image file
        :return: .png plot into local repository
        """

        valscombline = []
        for i in self.valscomb:
            valscombline += i

        binlims = np.arange(-2.5, 2.6, 0.1)

        fig = plt.figure(figsize=(19, 10))  # inches x, y with 72 dots per inch
        ax1 = fig.add_subplot(111)
        ax1.hist(valscombline, color='r', edgecolor='k', bins=binlims)
        ax1.set_xlabel('Doppler Shift, Hz')
        ax1.set_ylabel('Counts, N', color='r')
        ax1.set_xlim([-2.5, 2.5])  # 0.1Hz Bins (-2.5Hz to +2.5Hz)
        ax1.set_xticks(binlims[::2])

        plt.title('WWV 10 MHz Doppler Shift Distribution Plot \n'  # Title (top)
                  'Node: N0000020    Gridsquare: FN20vr \n'
                  'Lat=40.40.742018  Long=-74.178975 Elev=50M \n'
                  + self.month + ' UTC',
                  fontsize='10')
        plt.savefig(str(figname) + '.png', dpi=300, orientation='landscape')
        plt.close()

    def multGrapeDistPlots(self, dirname, figname, minBinLen=5):
        """
        Plots a series of distribution plots containing data across the provided time bin
        from each of the Grapes in the GrapeHandler

        :param dirname: string value for the name of the local directory where the plots will be saved
        :param figname: string value for the beginning of each image filename
        :param minBinLen: an integer value for the length of every time bin (minutes)
        :return: .png plot into dirname repository
        """

        secrange, minrange = mblHandle(minBinLen)

        # Make subsections and begin plot generation
        subranges = []  # contains equally sized ranges of data

        index = 0
        while not index > self.valslength:
            secs = []
            for vals in self.valscomb:
                secs += vals[index:index + secrange].tolist()
            subranges.append(secs)
            index += secrange

        hours = []  # contains 24 hour chunks of data

        index = 0
        while not index > len(subranges):
            hours.append(subranges[index:index + minrange])
            index += minrange

        # begin plot generation
        # initializes directory on local path if it does not exist
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        count = 0
        indexhr = 0
        for hour in hours:
            print('\nResolving hour: ' + str(indexhr) + ' ('
                  + str(floor((indexhr / len(hours)) * 100)) + '% complete) \n'
                  + '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

            index = 0
            for srange in hour:
                print('Resolving subrange: ' + str(index) + ' ('
                      + str(floor((index / len(hour)) * 100)) + '% complete)')

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Plot the subsections
                binlims = np.arange(-2.5, 2.6, 0.1)

                fig = plt.figure(figsize=(19, 10))  # inches x, y with 72 dots per inch
                ax1 = fig.add_subplot(111)
                ax1.hist(srange, color='r', edgecolor='k', bins=binlims)
                ax1.set_xlabel('Doppler Shift, Hz')
                ax1.set_ylabel('Counts, N', color='r')
                ax1.set_xlim([-2.5, 2.5])  # 0.1Hz Bins (-2.5Hz to +2.5Hz)
                ax1.set_xticks(binlims[::2])

                plt.title('WWV 10 MHz Doppler Shift Distribution Plot \n'
                          'Hour: ' + str(indexhr) + ' || 5-min bin: ' + str(index) + ' \n'  # Title (top)
                          # 'Node: N0000020    Gridsquare: FN20vr \n'
                          # 'Lat=40.40.742018  Long=-74.178975 Elev=50M \n'
                          + self.month + ' UTC',
                          fontsize='10')

                plt.savefig(str(dirname) + '/' + str(figname) + str(count) + '.png', dpi=300,
                            orientation='landscape')
                count += 1

                plt.close()

                index += 1

            indexhr += 1

    def mgBestFitsPlot(self, valname, dirname, figname, minBinLen=5, ylim=None):
        """
        Plots the best fits for each specified time bin over the doppler shift data for each grape object

        :param valname: string value dictating value selection (eg. 'f', 'v', or 'db')
        :param dirname: string value for the name of the local directory where the plots will be saved
        :param figname: string value for the beginning of each image filename
        :param minBinLen: int value for the length of each time bin in minutes (should be a factor of 60)
        :param ylim: Provide a python list containing minimum and maximum doppler shift in Hz
         for the data (default = [-1, 1])
        :return: .png plot into dirname repository
        """

        if not os.path.exists(dirname):
            os.mkdir(dirname)

        count = 0
        for grape in self.grapes:
            print('\nResolving grape: ' + str(count) + ' ('
                  + str(floor((count / len(self.grapes)) * 100)) + '% complete) \n'
                  + '*************************************')

            grape.bestFitsPlot(valname, dirname + '/' + figname + '_' + str(count), minBinLen=minBinLen, ylim=ylim)
            count += 1

    def tileTrend(self, figname, ylim=None, minBinLen=5, fSize=22):
        """
        Plots the 25th, 40th, 50th, 60th, and 75th quartiles for the monthly doppler shift of specified timebins of
        contained grapes

        :param figname: string value for the beginning of each image filename
        :param minBinLen: an integer value for the length of every time bin (minutes)
        :return: .png plot into local repository
        """

        secrange, minrange = mblHandle(minBinLen)

        qmarks = [0.25, 0.40, 0.50, 0.60, 0.75]
        q25, q40, q50, q60, q75 = [], [], [], [], []
        qts = [q25, q40, q50, q60, q75]
        qstyle = [('r', 0.25), ('b', 0.25), ('k', 1), ('b', 0.25), ('r', 0.25)]

        index = 0
        while not index > self.valslength:
            secs = []
            for vals in self.valscomb:
                secs += vals[index:index + secrange].tolist()
            qt = np.quantile(secs, qmarks)
            for i, q in enumerate(qts):
                q.append(qt[i])

            index += secrange

        if ylim is None:
            ylim = [-1, 1]

        t_range = self.grapes[self.bestgid].t_range
        t_range = t_range[::secrange]

        fig = plt.figure(figsize=(19, 10))  # inches x, y with 72 dots per inch
        ax1 = fig.add_subplot(111)
        for i in range(0, 5):
            ax1.plot(t_range, qts[i], qstyle[i][0], alpha=qstyle[i][1], linewidth=2)

        grape0 = self.grapes[0]
        # self.grapes[0].sunPosOver(fSize)

        labelpad = 20

        ax1.set_xlabel('UTC Hour', fontsize=fSize)
        ax1.set_ylabel('Doppler shift, Hz', fontsize=fSize)
        ax1.set_xlim(0, 24)  # UTC day
        ax1.set_ylim(ylim)  # -1 to 1 Hz for Doppler shift
        ax1.set_xticks(np.arange(0, 25, 2))
        ax1.tick_params(axis='x', labelsize=fSize - 2, direction='out', pad=labelpad)  # ax1.set_xlim([-2.5, 2.5])  # 0.1Hz Bins (-2.5Hz to +2.5Hz)
        ax1.tick_params(axis='y', labelsize=fSize - 2, direction='out', pad=labelpad)  # ax1.set_xlim([-2.5, 2.5])  # 0.1Hz Bins (-2.5Hz to +2.5Hz)
        ax1.grid(axis='x', alpha=1)
        ax1.grid(axis='y', alpha=0.5)

        alt_color = 'c'
        ax2 = ax1.twinx()
        ax2.plot(grape0.t_range, grape0.zentrace, alt_color, linewidth=2)
        ax2.set_ylabel('Solar Zenith Angle (°)', color=alt_color, fontsize=fSize)
        ax2.set_ylim(0, 180)
        ax2.tick_params(axis='y', colors=alt_color, labelsize=fSize - 2, direction='out', pad=labelpad)
        ax2.spines['right'].set_color(alt_color)

        plt.title('WWV 10 MHz Doppler Shift Plot (Q 25, 40, 50, 60, 75) \n'  # Title (top)
                  + '[K2MFF %s %s | Midpoint %s %s | WWV %s %s] \n'
                  % (decdeg2dms(self.grapes[0].lat), decdeg2dms(self.grapes[0].lon),
                     decdeg2dms(self.grapes[0].blat), decdeg2dms(self.grapes[0].blon),
                     decdeg2dms(WWV_LAT), decdeg2dms(WWV_LON)) +
                  '# of Grapes: ' + str(len(self.grapes)) + ' || '
                  + self.month,
                  fontsize=fSize)
        plt.savefig(str(figname) + '.png', dpi=300, orientation='landscape')
        plt.close()

    def spreadTrend(self, figname, minBinLen=5, fSize=22, ylim=(0, 1.2)):
        """
        Plots the daily variance in doppler shift Interquartile range for month-long segments of binned data

        :param figname: string value for the beginning of each image filename
        :param minBinLen: an integer value for the length of every time bin (minutes)
        :return: .png plot into local repository
        """

        secrange, minrange = mblHandle(minBinLen)

        # Gets the months and years for each grape
        monthMark = [0] * len(self.grapes)
        yearMark = [0] * len(self.grapes)
        for i, grape in enumerate(self.grapes):
            monthMark[i] = grape.month
            yearMark[i] = grape.year

        # Seperates data from seperate months into own month index
        months = [[]]
        mID = monthMark[0]
        yID = yearMark[0]
        myLabels = [[str(yID), str(mID)]]
        count = 0
        for i in np.arange(0, len(self.grapes)):
            if monthMark[i] > mID:
                mID = monthMark[i]
                count += 1
                months.append([])
                myLabels.append([str(yID), str(mID)])
            elif yearMark[i] > yID:
                yID = yearMark[i]
                count += 1
                months.append([])
                myLabels.append([str(yID), str(mID)])

            months[count].append(self.valscomb[i])

        qmarks = [0.25, 0.75]

        # calculates the difference between the 25th and 75th percentiles
        # of specified bins across all grapes in each month
        monthDiff = []
        for month in months:
            q25, q75 = [], []
            qts = [q25, q75]
            index = 0
            while not index > self.valslength:
                secs = []
                for vals in month:
                    secs += vals[index:index + secrange].tolist()
                qt = np.quantile(secs, qmarks)
                for i, q in enumerate(qts):
                    q.append(qt[i])
                index += secrange

            monthDiff.append([q75[i] - q25[i] for i in np.arange(0, len(q75))])

        # takes the t_range from the grape with the most data (longest day)
        t_range = self.grapes[self.bestgid].t_range
        t_range = t_range[::secrange]

        count = 0
        for diff in monthDiff:
            fig = plt.figure(figsize=(19, 10))  # inches x, y with 72 dots per inch
            ax1 = fig.add_subplot(111)
            ax1.plot(t_range, diff, linewidth=2)
            # t_range = [t_range[i] + 24 for i in range(0,len(t_range))]

            self.grapes[0].sunPosOver(fSize)

            ax1.set_xlabel('Time UTC', fontsize=fSize)
            ax1.set_ylabel('IQR, Hz', fontsize=fSize)
            ax1.set_ylim(ylim)  # UTC day
            ax1.set_xlim(0, 24)  # UTC day
            ax1.set_xticks(np.arange(0, 25, 2))
            ax1.tick_params(axis='x', labelsize=20)  # ax1.set_xlim([-2.5, 2.5])  # 0.1Hz Bins (-2.5Hz to +2.5Hz)
            ax1.tick_params(axis='y', labelsize=20)  # ax1.set_xlim([-2.5, 2.5])  # 0.1Hz Bins (-2.5Hz to +2.5Hz)
            ax1.grid(axis='x', alpha=1)
            ax1.grid(axis='y', alpha=0.5)

            plt.title('WWV 10 MHz Doppler Shift Interquartile Range \n'  # Title (top)
                      + '# of Grapes: ' + str(len(self.grapes)) + ' || '
                      + myLabels[count][0] + '-' + myLabels[count][1],
                      fontsize=fSize)
            plt.savefig(str(figname) + '.png', dpi=300, orientation='landscape')
            plt.close()
            count += 1

    def dopPowPlots(self, dirname, figname, ylim=None, fSize=22):
        """
        Produced Doppler-Power Plots for all contained grapes INDIVIDUALLY

        :param dirname: directory to contain the plots
        :param figname: string for prefix to each plot filename
        :param ylim: bounds for the ylim of the doppler shift plot (default [-1,1])
        :param fSize: integer scaling factor for the fontize of the plot text labels
        :return:
        """

        # If using subdirectories in dirname, ensure parent folder has already been created
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        count = 0
        for grape in self.grapes:
            print('Resolving grape: ' + str(count) + ' ('
                  + str(floor((count / len(self.grapes)) * 100)) + '% complete) \n'
                  + '*************************************\n')

            grape.dopPowPlot(dirname + '/' + figname + str(count + 1), ylim=ylim, fSize=fSize)
            count += 1

    def yearMedTrend(self, figname):
        """
        Plots both the daytime and nighttime medians of all of GrapeHandler's grapes across the year

        :param figname: Filename of the produced plot image
        :return: .png plot into local repository
        """

        self.dMeds = []
        self.nMeds = []

        startday = self.grapes[0].day
        startmonth = self.grapes[0].month

        skipcount = 0
        if startmonth != 1:
            dayadd = MONTHINDEX[startmonth - 1]
            for i in range(0, dayadd - 1):
                self.dMeds.append(0)
                self.nMeds.append(0)
                skipcount += 1
        if startday != 1:
            for i in range(0, startday - 1):
                self.dMeds.append(0)
                self.nMeds.append(0)
                skipcount += 1

        grapeindex = 0
        for i, ind in enumerate(MONTHINDEX):
            while ((grapeindex + skipcount) < (ind + MONTHLEN[i])) and (grapeindex < len(self.grapes)):
                grape = self.grapes[grapeindex]
                day = grape.day
                month = grape.month

                valid = (grape.beacon == 'WWV10') and ((grape.dayMed is not None) and (grape.nightMed is not None)) \
                        and (abs(grape.dayMed) < 100 and abs(grape.nightMed) < 100)
                rightday = (day == (grapeindex + skipcount - ind + 1)) and (MONTHINDEX[month - 1] == ind)

                skip = True
                if rightday:
                    if valid:
                        self.dMeds.append(grape.dayMed)
                        self.nMeds.append(grape.nightMed)
                    else:
                        self.dMeds.append(0)
                        self.nMeds.append(0)
                    grapeindex += 1
                    skip = False

                if skip:
                    self.dMeds.append(0)
                    self.nMeds.append(0)
                    skipcount += 1

        for i in range(len(self.dMeds), 365):
            self.dMeds.append(0)
            self.nMeds.append(0)

        xrange = np.arange(0, 365)

        plt.figure(figsize=(19, 10))  # inches x, y with 72 dots per inch
        plt.plot(xrange, self.dMeds, linewidth=2, color='r')
        plt.plot(xrange, self.nMeds, linewidth=2, color='b')

        for m in MONTHINDEX:
            plt.axvline(x=m, color='y', linewidth=3, linestyle='dashed', alpha=0.3)

        plt.xlabel('Month', fontsize=22)
        plt.ylabel('Median Doppler Shift, Hz', fontsize=22)
        plt.xticks(MONTHINDEX, MONTHS)
        plt.tick_params(axis='x', labelsize=20)
        plt.tick_params(axis='y', labelsize=20)
        plt.grid(axis='x', alpha=0.3)
        plt.grid(axis='y', alpha=1)
        plt.legend(["Sun Up Medians",
                    "Sun Down Medians"], fontsize=22)

        plt.title('WWV 10 MHz Doppler Shift Median Trend for %i \n' % self.grapes[0].year +
                  'Sun Up / Sun Down Times Relative to Midpoint %s %s'
                  % (decdeg2dms(self.grapes[0].blat), decdeg2dms(self.grapes[0].blon)),  # Title (top)
                  fontsize=22)
        plt.savefig(str(figname) + '.png', dpi=300, orientation='landscape')

        plt.show()
        plt.close()

    def yearSpreadTrend(self, figname):
        """
        Plots the daily interquartile range of all of GrapeHandler's grapes across the year

        :param figname: Filename of the produced plot image
        :return: .png plot into local repository
        """

        for grape in self.grapes:
            if grape.beacon == 'WWV10':
                if abs(grape.dayMed) < 100 and abs(grape.nightMed) < 100:
                    self.dMeds.append(grape.dayMed)
                    self.nMeds.append(grape.nightMed)
                else:
                    self.dMeds.append(0)
                    self.nMeds.append(0)
            else:
                self.dMeds.append(0)
                self.nMeds.append(0)

        xrange = np.arange(0, len(self.grapes))

        plt.figure(figsize=(19, 10))  # inches x, y with 72 dots per inch
        plt.plot(xrange, self.dMeds, linewidth=2, color='r')
        plt.plot(xrange, self.nMeds, linewidth=2, color='b')

        for m in MONTHINDEX:
            plt.axvline(x=m, color='y', linewidth=3, linestyle='dashed', alpha=0.3)

        plt.xlabel('Month', fontsize=22)
        plt.ylabel('Median Doppler Shift, Hz', fontsize=22)
        # plt.ylim([-1.5, 1.5])
        plt.xticks(MONTHINDEX, MONTHS)
        plt.tick_params(axis='x', labelsize=20)
        plt.tick_params(axis='y', labelsize=20)
        plt.grid(axis='x', alpha=0.3)
        plt.grid(axis='y', alpha=1)
        plt.legend(["Sun Up Medians",
                    "Sun Down Medians"], fontsize=22)

        plt.title('WWV 10 MHz Doppler Shift Median Trend \n',  # Title (top)
                  fontsize=22)
        plt.savefig(str(figname) + '.png', dpi=300, orientation='landscape')
        plt.close()

    def yearDopPlot(self, figname, fSize=22):
        """
        Plots vertical gradients representative of doppler shifts per day across the year

        :param figname: Filename of the produced plot image
        :return: .png plot into local repository
        """

        year_data = []
        year_times = []
        null_day = [0] * self.valslength

        startday = self.grapes[0].day
        startmonth = self.grapes[0].month
        startyear = self.grapes[0].year

        skipcount = 0
        if startmonth != 1:
            dayadd = MONTHINDEX[startmonth - 1]
            for i in range(0, dayadd - 1):
                year_data.append(null_day)
                year_times.append(null_day)
                skipcount += 1
        if startday != 1:
            for i in range(0, startday - 1):
                year_data.append(null_day)
                year_times.append(null_day)
                skipcount += 1

        grapeindex = 0
        for i, ind in tqdm(enumerate(MONTHINDEX)):
            while ((grapeindex + skipcount) < (ind + MONTHLEN[i])) and (grapeindex < len(self.grapes)):
                grape = self.grapes[grapeindex]
                day = grape.day
                month = grape.month

                valid = (grape.beacon == 'WWV10')
                rightday = (day == (grapeindex + skipcount - ind + 1)) and (MONTHINDEX[month - 1] == ind)

                skip = True
                if rightday:
                    if valid:
                        year_data.append(self.valscomb[grapeindex])
                        year_times.append(self.timecomb[grapeindex])
                    else:
                        year_data.append(null_day)
                        year_times.append(null_day)
                    grapeindex += 1
                    skip = False

                if skip:
                    year_data.append(null_day)
                    year_times.append(null_day)
                    skipcount += 1

        # Fill in rest of year with zeros
        for i in range(grapeindex + skipcount, 365):
            year_data.append(null_day)
            year_times.append(null_day)

        fig = plt.figure(figsize=(19, 10))  # inches x, y with 72 dots per inch

        offset = self.grapes[0].t_offset
        timeShift = int((offset * 60 * 60) / self.ss_factor)

        # Assembling and rolling flat data and time
        flat_year_data = []  # Data
        for day in year_data:
            for second in day:
                flat_year_data.append(second)
        rolled_year = np.roll(flat_year_data, timeShift)

        # Code to re-partition data and time into daylong arrays
        new_year_data = year_data  # Data
        count = 0
        for day in new_year_data:
            for i, second in enumerate(day):
                day[i] = rolled_year[count]
                count += 1
        year_data = new_year_data

        scatter = None
        print('Begin Plotting \n')
        for i in tqdm(range(0, 365)):
            if np.any(year_data[i]):
                xrange = [i for j in range(0, len(year_data[i]))]
                scatter = plt.scatter(xrange, year_times[i], c=year_data[i], norm=colors.CenteredNorm(),
                                      cmap='seismic_r')

        plt.xlabel('Month', fontsize=fSize)
        plt.ylabel('Time, Hr', fontsize=fSize)
        plt.xticks(MONTHINDEX, MONTHS)
        plt.yticks([i for i in range(0, 25)][::2])
        plt.xlim(0, 365)
        plt.ylim(0, 24)
        plt.tick_params(axis='x', labelsize=fSize - 2)
        plt.tick_params(axis='y', labelsize=fSize - 2)
        plt.grid(axis='x', alpha=0.3)
        plt.grid(axis='y', alpha=1)

        cbar = fig.colorbar(scatter, extend='both')
        cbar.minorticks_on()
        cbar.ax.tick_params(labelsize=fSize - 2)
        cbar.ax.get_yaxis().labelpad = fSize
        cbar.ax.set_ylabel('Doppler Shift (Hz)', fontsize=fSize, rotation=270)

        plt.title('%i WWV 10 MHz Doppler Shift Trend at Midpoint %s, %s' %  # Title (top)
                  (startyear, decdeg2dms(self.grapes[0].blat), decdeg2dms(self.grapes[0].blon)),
                  fontsize=fSize)
        plt.savefig('%s.png' % figname, dpi=300, orientation='landscape')
        plt.close()

    def dopPlotOver(self, figname='dopPlotOver', fSize=22, ylim=None, **kwargs):

        if ylim is None:
            ylim = ydoplims(self.valscomb, 'f')

        labelpad = 20

        fig = plt.figure(figsize=(19, 10), layout='tight')  # inches x, y with 72 dots per inch
        ax1 = fig.add_subplot(111)

        grape0 = self.grapes[0]
        # grape0.sunPosOver(fSize)

        for i in range(len(self.grapes)):
            frange = self.valscomb[i]
            trange = self.timecomb[i]

            if i == len(self.grapes) - 1:
                ax1.plot(trange, frange, 'r', linewidth=2)
            else:
                ax1.plot(trange, frange, 'k', linewidth=2, alpha=i / len(self.grapes))

        ax1.set_xlabel('UTC Hour', fontsize=fSize)
        ax1.set_ylabel(FLABEL, fontsize=fSize)
        ax1.set_xlim(0, 24)  # UTC day
        ax1.set_ylim(ylim)  # -1 to 1 Hz for Doppler shift
        ax1.set_xticks(np.arange(0, 25, 2))
        ax1.tick_params(axis='x', labelsize=20)  # plt.set_xlim([-2.5, 2.5])  # 0.1Hz Bins (-2.5Hz to +2.5Hz)
        ax1.tick_params(axis='y', labelsize=20)  # plt.set_xlim([-2.5, 2.5])  # 0.1Hz Bins (-2.5Hz to +2.5Hz)
        ax1.grid(axis='x', alpha=1)
        ax1.grid(axis='y', alpha=0.5)

        cbar = plt.colorbar(cm.ScalarMappable(norm=colors.CenteredNorm(), cmap='Greys'), pad = 0.08)
        tick_labels = kwargs.get('tl',
                                 ['', 'Oct 7', 'Oct 8', 'Oct 9', 'Oct 10', 'Oct 11', 'Oct 12', 'Oct 13', 'Oct 14'])
        cbar.ax.set_yticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=fSize - 2)
        cbar.ax.set_ylabel('Date of Trace', fontsize=fSize)

        alt_color = 'c'
        ax2 = ax1.twinx()
        ax2.plot(grape0.t_range, grape0.zentrace, alt_color, linewidth=2)
        ax2.set_ylabel('Solar Zenith Angle (°)', color=alt_color, fontsize=fSize)
        ax2.set_ylim(0, 180)
        ax2.tick_params(axis='y', colors=alt_color, labelsize=fSize - 2, direction='out', pad=labelpad)
        ax2.spines['right'].set_color(alt_color)

        plt.suptitle('Doppler Residual Traces (' + grape0.date + ' - ' + self.grapes[-1].date + ')',
                     fontsize=fSize + 10, ha='center', weight='bold', x=0.45)  # Title (top)
        plt.title('[K2MFF %s %s | Midpoint %s %s | WWV %s %s] \n'
                  % (decdeg2dms(grape0.lat), decdeg2dms(grape0.lon),
                     decdeg2dms(grape0.blat), decdeg2dms(grape0.blon),
                     decdeg2dms(WWV_LAT), decdeg2dms(WWV_LON)),
                  fontsize=fSize, ha='center')

        plt.tight_layout()
        plt.savefig(str(figname) + '.png', dpi=300, orientation='landscape')
        plt.close()

    def dopPlotOver_copy(self, figname='dopPlotOver', fSize=22, ylim=None, **kwargs):

        if ylim is None:
            ylim = ydoplims(self.valscomb, 'f')

        plt.figure(figsize=(19, 10))  # inches x, y with 72 dots per inch

        self.grapes[0].sunPosOver(fSize)

        for i in range(len(self.grapes)):
            frange = self.valscomb[i]
            trange = self.timecomb[i]

            if i == len(self.grapes) - 1:
                plt.plot(trange, frange, 'r', linewidth=2)
            else:
                plt.plot(trange, frange, 'k', linewidth=2, alpha=i / len(self.grapes))

        plt.xlabel('UTC Hour', fontsize=fSize)
        plt.ylabel(FLABEL, fontsize=fSize)
        plt.xlim(0, 24)  # UTC day
        plt.ylim(ylim)  # -1 to 1 Hz for Doppler shift
        plt.xticks(np.arange(0, 25, 2))
        plt.tick_params(axis='x', labelsize=20)  # plt.set_xlim([-2.5, 2.5])  # 0.1Hz Bins (-2.5Hz to +2.5Hz)
        plt.tick_params(axis='y', labelsize=20)  # plt.set_xlim([-2.5, 2.5])  # 0.1Hz Bins (-2.5Hz to +2.5Hz)
        plt.grid(axis='x', alpha=1)
        plt.grid(axis='y', alpha=0.5)

        cbar = plt.colorbar(cm.ScalarMappable(norm=colors.CenteredNorm(), cmap='Greys'))
        cbar.minorticks_on()
        tick_labels = kwargs.get('tl',
                                 ['', 'Oct 7', 'Oct 8', 'Oct 9', 'Oct 10', 'Oct 11', 'Oct 12', 'Oct 13', 'Oct 14'])
        cbar.ax.set_yticklabels(tick_labels)
        cbar.ax.tick_params(labelsize=fSize - 2)
        cbar.ax.get_yaxis().labelpad = fSize + 3
        cbar.ax.set_ylabel('Date of Trace', fontsize=fSize, rotation=270)

        # plt.title('WWV 10 MHz Doppler Shift Plot Comparison \n'  # Title (top)
        #           + '[K2MFF %s %s | Midpoint %s %s | WWV %s %s] \n'
        #           % (decdeg2dms(self.grapes[0].lat), decdeg2dms(self.grapes[0].lon),
        #             decdeg2dms(self.grapes[0].blat), decdeg2dms(self.grapes[0].blon),
        #             decdeg2dms(WWV_LAT), decdeg2dms(WWV_LON)) +
        #           self.grapes[0].date + ' through ' + self.grapes[-1].date,
        #           fontsize=fSize)
        plt.suptitle('Doppler Residual Traces (' + self.grapes[0].date + ' - ' + self.grapes[-1].date + ')',
                     fontsize=fSize + 10, ha='center', weight='bold', x=0.45)  # Title (top)
        plt.title('[K2MFF %s %s | Midpoint %s %s | WWV %s %s] \n'
                  % (decdeg2dms(self.grapes[0].lat), decdeg2dms(self.grapes[0].lon),
                     decdeg2dms(self.grapes[0].blat), decdeg2dms(self.grapes[0].blon),
                     decdeg2dms(WWV_LAT), decdeg2dms(WWV_LON)),
                  fontsize=fSize, ha='center')

        plt.tight_layout()
        plt.savefig(str(figname) + '.png', dpi=300, orientation='landscape')
        plt.close()


# Shortcut Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def gen_yearDopPlot(figname='yearDopShift', state='NJ', year='2022', beacon='wwv', n=60 * 5, monthRange=None, p=False):
    """
    Shortcut function to quickly generate yearDopPlot using GrapeHandler

    :param figname: prefix of the figures to be produced
    :param state: US State of the RX
    :param year: Year of data collection
    :param beacon: TX Beacon Name
    :param n: Subsampling term
    :param monthRange:  Range of months to search for (eg. [int, int], or [int, 'rest'] to go from first index to last
                        available month)
    :return:
    """

    months = ('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec')
    data_dir = '%s_data/' % state

    folders = []
    if monthRange:
        if monthRange[1] == 'rest':
            monthClip = months[monthRange[0]::]
        else:
            monthClip = months[monthRange[0]:monthRange[1]]
        for m in monthClip:
            folders.append(('%s%s_%s_%s' % (data_dir, beacon, m, year)) if state == 'NJ' else
                           ('%s_%s_%s' % (data_dir, m, year)))
    else:
        for m in months:
            folders.append(('%s%s_%s_%s' % (data_dir, beacon, m, year)) if state == 'NJ' else
                           ('%s_%s_%s' % (data_dir, m, year)))

    gh = GrapeHandler(folders, filt=False, comb=True, med=False, tShift=False, n=n)

    if p:
        pickle_grape(gh)

    gh.yearDopPlot('%s_%s_%s' % (figname, state, year))

    return gh


def grapeLoad(year: int, month: int, day: int, **kwargs):
    # %%
    # Convert all numbers to correct format
    year = str(year)
    monthname = MONTHS[month - 1].lower()

    if day < 10:
        day = '0' + str(day)
    if month < 10:
        month = '0' + str(month)

    day = str(day)
    month = str(month)

    # %%
    # Get Grape Filename (Defaults to Newark WWV folder)
    filename = kwargs.get('filename', None)
    if not filename:
        wwv_folder = f'wwv_{monthname}_{year}'
        filepath = f'{NJ_DATA_PATH}/{wwv_folder}'
        filepath = kwargs.get('filepath', filepath)
        fname = f'{year}-{month}-{day}{K2MFF_SIG}.csv'
        filename = f'{filepath}/{fname}'

    # %%
    # Get Grape parameters
    convun = kwargs.get('convun', True)
    filt = kwargs.get('filt', True)
    med = kwargs.get('med', False)
    count = kwargs.get('count', False)
    n = kwargs.get('n', 1)

    # %%
    # Load Grape
    g = Grape(filename=filename,
              convun=convun,
              filt=filt,
              med=med,
              count=count,
              n=n)

    return g


# Global Util ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def pickle_grape(gObj, filename=None):
    if not filename:
        if isinstance(gObj, Grape):
            filename = 'g_%s_%s' % (gObj.cityState, gObj.date)
        if isinstance(gObj, GrapeHandler):
            filename = 'gh_%s_%s_%s' % (gObj.grapes[0].cityState[-2::], gObj.grapes[0].date, gObj.grapes[-1].date)

    with open(filename + '.pkl', 'wb') as f:  # open a text file
        gPickle = pickle.dump(gObj, f)  # serialize the grape object
        f.close()

    return gPickle


def unpickle_grape(gPickleFname):
    with open(gPickleFname, 'rb') as f:
        gObj = pickle.load(f)  # deserialize using load()
        f.close()

    if isinstance(gObj, Grape):
        print('Grape %s %s loaded!' % (gObj.cityState, gObj.date))
    if isinstance(gObj, GrapeHandler):
        print('Grape Handler %s %s through %s loaded!' % (
            gObj.grapes[0].cityState[-2::], gObj.grapes[0].date, gObj.grapes[-1].date))

    return gObj


def ydoplims(yrange, valname, cutoff=3):
    if isinstance(yrange[0], np.ndarray):
        # truncates maximum and minimum values
        bottom = round_down(yrange[0].min(), 1)
        top = round_up(yrange[0].max(), 1)

        for range in yrange:
            thismin = round_down(range.min(), 1)
            thismax = round_up(range.max(), 1)

            if thismin < bottom:
                bottom = thismin
            if thismax > top:
                top = thismax

    else:
        # truncates maximum and minimum values
        bottom = round_down(yrange.min(), 1)
        top = round_up(yrange.max(), 1)

    # snapping to +- 1, 1.2, 1.5 or 2 for doppler vals
    if valname in FNAMES:
        if bottom > -2:
            bottom = -1 if (bottom > -1) else (
                -1.2 if (bottom > -1.2) else (-1.5 if (bottom > -1.5) else -2))
        else:
            if bottom < -cutoff:
                bottom = -cutoff
        if top < 2:
            top = 1 if (top < 1) else (1.2 if (top < 1.2) else (1.5 if (top < 1.5) else 2))
        else:
            if top > cutoff:
                top = cutoff

    return [bottom, top]


def movie(dirname, gifname, fps=10):
    """
    Combines the images in the provided directory into a gif of the specified framerate

    :param dirname: string value for the name of the local directory containing
    the images (.png) to be processed into a gif
    :param gifname: string value for the name of the produced gif
    :param fps: integer value for the produced gif's frames per second (default 10)
    :return: a gif of the combined files in the local repository
    """
    gif = None

    if os.path.exists(dirname):
        # assign directory
        directory = dirname

        filenames = []
        # iterate over files in that directory
        for filename in os.scandir(directory):
            if filename.is_file():
                filenames.append('./' + directory + '/' + filename.name)

        filenames.sort(key=lambda f: int(sub('\D', '', f)))

        frames = []
        for t in np.arange(0, len(filenames)):
            image = imageio.v2.imread(filenames[t])
            frames.append(image)

        print('Frames processed')

        gif = imageio.mimsave('./' + gifname + '.gif',  # output gif
                              frames,  # array of input frames
                              fps=fps)  # optional: frames per second

        print('.gif success')

    else:
        print('That directory does not exist on the local path! \n'
              'Please try again.')

    return gif


def mblHandle(minBinLen):
    """
    Converts the requested time frame (in minutes) to appropriate factors used for range calculation

    :param minBinLen: an integer value for the length of every time bin (minutes)
    :return: secrange [range of 'seconds' for every time bin] and minrange [range of 'minutes' for time bin]
    """

    if 60 % minBinLen == 0:
        secrange = int(minBinLen * 60)
        minrange = int(60 / minBinLen)
    else:
        hrDivs = (1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60)
        binFix = 1
        for i in hrDivs:
            binFix = i if minBinLen > i else binFix

        print("Please choose a minute bin length that divides into 60 evenly!")
        print("Rounding down to the closest factor, " + str(binFix))

        secrange = int(binFix * 60)
        minrange = int(60 / binFix)

    return secrange, minrange


def round_up(n, decimals=0):
    """
    Rounds the provided number up to the specified number of decimal places

    :param n: the number to be rounded
    :param decimals: the number of decimals the rounded value will contain
    :return: rounded number
    """

    multiplier = 10 ** decimals
    return ceil(n * multiplier) / multiplier


def round_down(n, decimals=0):
    """
    Rounds the provided number down to the specified number of decimal places

    :param n: the number to be rounded
    :param decimals: the number of decimals the rounded value will contain
    :return: rounded number
    """

    multiplier = 10 ** decimals
    return floor(n * multiplier) / multiplier


def conv_time(timestamp, unit='h'):
    """
    Converts timestamps created for sun position times (hr, min, sec) by sunpy into decimal hours

    :param unit: String hint for what to convert the timestamp to
    :param timestamp: timestamp object containing hr, min, sec
    :return: time in decimal hours
    """

    hr = timestamp.hour
    mi = timestamp.minute
    sec = timestamp.second

    convun = 0
    if unit == 'h':
        convun = hr + mi / 60 + sec / 3600
    if unit == 'm':
        convun = hr * 60 + mi + sec / 60
    if unit == 's':
        convun = hr * 3600 + mi * 60 + sec

    return convun


def decdeg2dms(dd):
    mnt, sec = divmod(abs(dd) * 3600, 60)
    deg, mnt = divmod(mnt, 60)

    mult = -1 if dd < 0 else 1
    deg = round_down(mult * deg, 0)
    mnt = round_down(mult * mnt, 0)
    sec = round_down(mult * sec, 0)

    coordString = ' %i°%i\'%i\"' % (deg, mnt, sec)

    return coordString


def mpt_coords(lat1, lon1, lat2, lon2):
    """
    Calculates the midpoint geographic location between the beacon (WWV) and reciever (Grape) along the great
    circle path. Accepts parameters in degrees, and returns coordinates in degrees.

    :param lat1: Latitude of the Beacon
    :param lon1: Longitude of the Beacon
    :param lat2: Latitude of the Receiver
    :param lon2: Longitude of the Receiver
    :return: Latitude, Longitude of the midpoint in degrees
    """
    lat1, lon1, lat2, lon2 = rad(lat1), rad(lon1), rad(lat2), rad(lon2)

    # 1 = origin (WWV), 2 = receiver (Newark)
    X = cos(lat2) * cos(lon2 - lon1)
    Y = cos(lat2) * sin(lon2 - lon1)

    # midpoint (bounce point) latitude
    mlat = atan2(sin(lat1) + sin(lat2), sqrt((cos(lat1) + X) ** 2 + Y ** 2))
    # midpoint (bounce point) longitude
    mlon = lon1 + atan2(Y, cos(lat1) + X)

    mlat, mlon = deg(mlat), deg(mlon)

    return mlat, mlon


def figname(**kwargs):
    """
    Method to produce Grape figure name based off of user input or current time.
    """

    # %%
    # Get Figure Filename
    figname = kwargs.get('figname', None)
    if not figname:
        time = datetime.now()
        y = time.year
        mo = time.month
        d = time.day
        h = time.hour
        mi = time.minute
        s = time.second
        tstring = f'{y}_{mo}_{d}T{h}_{mi}_{s}'

        figfolder = kwargs.get('figfolder', '')
        figdir = f'FIGURES/{figfolder}'

        figname = f'{figdir}/{tstring}'

    return figname


def figtitle(ax, plottype: str, **kwargs):
    """
    Method to produce uniform titles across Grape Plots

    :param ax: matplotlib axes object to apply title to
    :param plottype: String to describe the contents of the plot
    :param kwargs: plot settings containing title information and formatting
    :return: string to be set as plot title
    """
    unknown = '???'

    date =      kwargs.get('date', unknown)
    lat =       kwargs.get('lat', unknown)
    lon =       kwargs.get('lon', unknown)
    blat =      kwargs.get('blat', unknown)
    blon =      kwargs.get('blon', unknown)
    wwvlat =    kwargs.get('wwvlat', unknown)
    wwvlon =    kwargs.get('wwvlon', unknown)

    y =         kwargs.get('y', unknown)
    pad =       kwargs.get('pad', unknown)
    fontsize =  kwargs.get('fontsize', unknown)

    tstring = f'WWV 10 MHz {plottype} for {date} \n' \
                            + '[K2MFF %s %s | Midpoint %s %s | WWV %s %s]' \
                            % (lat, lon, blat, blon, wwvlat, wwvlon)

    ax.set_title(tstring, y=y, pad=pad, fontsize=fontsize)

    return tstring
