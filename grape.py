import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
from math import floor, ceil
from csv import reader
import os
import imageio as imageio
from re import sub
from fitter import Fitter
from datetime import datetime
import suncalc
from statistics import median

fnames = ('d', 'dop', 'doppler', 'doppler shift', 'f', 'freq', 'frequency')
vnames = ('v', 'volt', 'voltage')
pnames = ('db', 'decibel', 'p', 'pwr', 'power')

flabel = 'Doppler Shift, Hz'
plabel = 'Relative Power, B'
vlabel = 'Voltage, V'

months = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
monthlen = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
monthindex = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)

# WWV Broadcasting tower coordinates based on:
# https://latitude.to/articles-by-country/us/united-states/6788/wwv-radio-station
WWV_LAT = 40.67583063
WWV_LON = -105.038933178


class Grape:

    def __init__(self, filename=None, convun=True, filt=False, med=False, count=False, n=1):
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
        self.sunpos = None

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

        # Flags to keep track of if the load() or units() function have been called, respectively
        self.loaded = False
        self.converted = False
        self.filtered = False

        if filename:
            self.load(filename, n=n)

        if self.loaded:
            if filt:
                self.butFilt()
            if convun:
                self.units()
            if count:
                self.count()
            if med:
                self.dnMedian()

    def load(self, filename, n=1):
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
        # for i in header_data:
        #     print(i)
        # col_title = lines[18].split()               # Titles for each data range

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

        d = datetime(self.year, self.month, self.day, 1)  # datetime object for 1st hour of the day

        self.blat = (self.lat + WWV_LAT) / 2
        self.blon = (self.lon + WWV_LON) / 2

        self.RXsuntimes = suncalc.get_times(d, self.lon, self.lat)
        self.Bsuntimes = suncalc.get_times(d, self.blon, self.blat)
        self.TXsuntimes = suncalc.get_times(d, WWV_LON, WWV_LAT)

        self.sunpos = []

        # Read each line of file after the header
        for line in lines[19::n]:
            date_time = str(line[0]).split('T')
            utc_time = str(date_time[1]).split(':')

            hour = int(utc_time[0])
            minute = int(utc_time[1])
            second = int(utc_time[2][0:2])

            d = datetime(self.year, self.month, self.day, hour, minute, second)
            self.sunpos.append(suncalc.get_position(d, self.lon, self.lat))

            sec = (float(hour) * 3600) + \
                  (float(minute) * 60) + \
                  (float(second))

            self.time.append(sec)  # time list append
            self.freq.append(float(line[1]))  # doppler shift list append
            self.Vpk.append(float(line[2]))  # voltage list append

        self.time = np.array(self.time)
        self.freq = np.array(self.freq)
        self.Vpk = np.array(self.Vpk)

        if len(self.time) != 0:
            if self.beacon == 'WWV10':
                # Raise loaded flag
                self.loaded = True
                print("Grape " + self.date + " loaded! \n")
            else:
                print("Grape " + self.date + " not loaded (not WWV10) \n")
        else:
            print("Grape " + self.date + " not loaded (no data) \n")

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

    def butFilt(self, FILTERORDER=3, FILTERBREAK=0.005):
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

    def units(self, timediv=3600, fdel=10e6):
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

            # self.t_range = [(t / timediv) for t in self.time]  # Time range (Hours)
            # self.f_range = [(f - fdel) for f in self.freq]  # Doppler shifts (del from 10GHz)
            # self.Vdb_range = [10 * np.log10(v ** 2) for v in
            #                   self.Vpk]  # Relative power (Power is proportional to V^2; dB)
            if self.filtered:
                self.f_range_filt = self.freq_filt - fdel
                self.Vdb_range_filt = 10 * np.log10(self.Vpk_filt ** 2)

                # self.f_range_filt = [(f - fdel) for f in self.freq_filt]  # Doppler shifts (del from 10GHz)
                # self.Vdb_range_filt = [10 * np.log10(v ** 2) for v in
                #                        self.Vpk_filt]  # Relative power (Power is proportional to V^2; dB)
            self.converted = True
        else:
            print('Time, frequency and Vpk not loaded!')

    def valCh(self, valname):
        label = 'None'

        if valname in fnames:
            vals = self.f_range
            label = flabel
        elif valname in vnames:
            vals = self.Vpk
            label = vlabel
        elif valname in pnames:
            vals = self.Vdb_range
            label = plabel
        else:
            vals = None

        return vals, label

    def dnMedian(self):
        """
        Calculates the medians of the entire day's sunlight and sundown doppler shifts, seperately

        :return: day and night medians to grape object
        """

        Bsr = to_hr(self.Bsuntimes['sunrise'])
        Bss = to_hr(self.Bsuntimes['sunset'])

        # Note - improve with numpy arrays
        srIndex = min(range(len(self.t_range)), key=lambda i: abs(self.t_range[i] - Bsr))
        ssIndex = min(range(len(self.t_range)), key=lambda i: abs(self.t_range[i] - Bss))

        if ssIndex < srIndex:
            sunUp = self.f_range[0:ssIndex]
            for i in self.f_range[srIndex:(len(self.f_range) - 1)]:
                sunUp.append(i)
            sunDown = self.f_range[ssIndex:srIndex]
        else:
            sunDown = self.f_range[0:srIndex]
            for i in self.f_range[ssIndex:(len(self.f_range) - 1)]:
                sunDown.append(i)
            sunUp = self.f_range[srIndex:ssIndex]

        if len(sunUp) > 1:
            self.dayMed = median(sunUp)
        else:
            print('No sunup detected (dayMed None)\n')
        if len(sunDown) > 1:
            self.nightMed = median(sunDown)
        else:
            print('No sundown detected (nightMed None)\n')

    def sunPosOver(self, fSize):
        """
        Plots overlay of the grape's sunrise, sunset, and solar noon features over the currently open matplotlib plot

        :param fSize: fontsize of legend text
        :return: vertical lines on current graph
        """

        RXsr = to_hr(self.RXsuntimes['sunrise'])
        RXsn = to_hr(self.RXsuntimes['solar_noon'])
        RXss = to_hr(self.RXsuntimes['sunset'])

        Bsr = to_hr(self.Bsuntimes['sunrise'])
        Bsn = to_hr(self.Bsuntimes['solar_noon'])
        Bss = to_hr(self.Bsuntimes['sunset'])

        TXsr = to_hr(self.TXsuntimes['sunrise'])
        TXsn = to_hr(self.TXsuntimes['solar_noon'])
        TXss = to_hr(self.TXsuntimes['sunset'])

        RXsrMark = plt.axvline(x=RXsr, color='y', linewidth=3, linestyle='dashed', alpha=0.3)
        RXsnMark = plt.axvline(x=RXsn, color='g', linewidth=3, linestyle='dashed', alpha=0.3)
        RXssMark = plt.axvline(x=RXss, color='b', linewidth=3, linestyle='dashed', alpha=0.3)

        BsrMark = plt.axvline(x=Bsr, color='y', linewidth=3, linestyle='dashed')
        BsnMark = plt.axvline(x=Bsn, color='g', linewidth=3, linestyle='dashed')
        BssMark = plt.axvline(x=Bss, color='b', linewidth=3, linestyle='dashed')

        TXsrMark = plt.axvline(x=TXsr, color='y', linewidth=3, linestyle='dashed', alpha=0.3)
        TXsnMark = plt.axvline(x=TXsn, color='g', linewidth=3, linestyle='dashed', alpha=0.3)
        TXssMark = plt.axvline(x=TXss, color='b', linewidth=3, linestyle='dashed', alpha=0.3)

        plt.legend([BsrMark, BsnMark, BssMark], ["Sunrise: " + str(round_down(Bsr, 2)) + " UTC",
                                                 "Solar Noon: " + str(round_down(Bsn, 2)) + " UTC",
                                                 "Sunset: " + str(round_down(Bss, 2)) + " UTC"], fontsize=fSize)

    def dopPowPlot(self, figname, ylim=None, fSize=22):
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

        if self.converted:
            frange = self.f_range if not self.filtered else self.f_range_filt
            Vdbrange = self.Vdb_range if not self.filtered else self.Vdb_range_filt

            fSize = fSize

            fig = plt.figure(figsize=(19, 10))  # inches x, y with 72 dots per inch
            ax1 = fig.add_subplot(111)
            ax1.plot(self.t_range, frange, 'k', linewidth=2)  # color k for black

            self.sunPosOver(fSize)

            ax1.set_xlabel('UTC Hour', fontsize=fSize)
            ax1.set_ylabel(flabel, fontsize=fSize)
            ax1.set_xlim(0, 24)  # UTC day
            ax1.set_ylim(ylim)  # -1 to 1 Hz for Doppler shift
            ax1.set_xticks(np.arange(0, 25, 2))
            ax1.tick_params(axis='x', labelsize=20)  # ax1.set_xlim([-2.5, 2.5])  # 0.1Hz Bins (-2.5Hz to +2.5Hz)
            ax1.tick_params(axis='y', labelsize=20)  # ax1.set_xlim([-2.5, 2.5])  # 0.1Hz Bins (-2.5Hz to +2.5Hz)
            ax1.grid(axis='x', alpha=1)
            ax1.grid(axis='y', alpha=0.5)

            ax2 = ax1.twinx()
            ax2.plot(self.t_range, Vdbrange, 'r-', linewidth=2)  # NOTE: Set for filtered version
            ax2.set_ylabel(plabel, color='r', fontsize=fSize)
            ax2.set_ylim(-80, 0)  # Try these as defaults to keep graphs similar.
            ax2.tick_params(axis='y', labelsize=20)  # ax1.set_xlim([-2.5, 2.5])  # 0.1Hz Bins (-2.5Hz to +2.5Hz)

            # following lines set ylim for power readings in file

            for tl in ax2.get_yticklabels():
                tl.set_color('r')

            plt.title('WWV 10 MHz Doppler Shift Plot \n'  # Title (top)
                      # 'Node: N0000020    Gridsquare: FN20vr \n'
                      # 'Lat=40.40.742018  Long=-74.178975 Elev=50M \n'
                      + self.date + ' UTC',
                      fontsize=fSize)
            plt.savefig(str(figname) + '.png', dpi=250, orientation='landscape')
            plt.close()
        else:
            print('Data units not yet converted! \n'
                  'Attempting unit conversion... \n'
                  'Please try again.')
            self.units()

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
                if valname in fnames:
                    binlims = np.arange(-2.5, 2.6, 0.1)

                fSize = 22
                fig = plt.figure(figsize=(19, 10))  # inches x, y with 72 dots per inch
                ax1 = fig.add_subplot(111)
                ax1.hist(vals, color='r', edgecolor='k', bins=binlims) if (binlims is not None) else ax1.hist(vals, color='r',
                                                                                                edgecolor='k')
                ax1.set_xlabel(label, fontsize=fSize)
                ax1.set_ylabel('Counts, N', color='r', fontsize=fSize)
                ax1.grid(axis='x', alpha=1)
                ax1.grid(axis='y', alpha=0.5)
                ax1.tick_params(axis='x',
                                labelsize=fSize - 2)  # ax1.set_xlim([-2.5, 2.5])  # 0.1Hz Bins (-2.5Hz to +2.5Hz)
                ax1.tick_params(axis='y', labelsize=fSize - 2)

                if valname in fnames:
                    pl.xlim([-1, 1])  # Doppler Shift Range
                    pl.xticks(np.arange(-1, 1.1, 0.1))

                plt.title('WWV 10 MHz ' + label.split(',')[0] + ' Distribution Plot \n'  # Title (top)
                          # 'Node: N0000020    Gridsquare: FN20vr \n'
                          # 'Lat=40.40.742018  Long=-74.178975 Elev=50M \n'
                          + self.date + ' UTC',
                          fontsize=fSize)
                plt.savefig(str(figname) + '.png', dpi=250, orientation='landscape')
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
            if valname in fnames:
                vals = self.f_range
            elif valname in vnames:
                vals = self.Vpk
            elif valname in pnames:
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

                            plt.savefig(str(dirname) + '/' + str(figname) + str(count) + '.png', dpi=250,
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

                    plt.savefig(str(figname) + '.png', dpi=250,
                                orientation='landscape')

            else:
                print("Please provide a valid valname!")
        else:
            print('Data units not yet converted! \n'
                  'Attempting unit conversion... \n'
                  'Please try again.')
            self.units()

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

    def distPlotFit(self, valname, figname):
        """
        Produces a fitted histogram for the entire day's worth of data (for the specified value)

        :param valname: string value dictating value selection (eg. 'f', 'v', or 'db')
        :param figname: string value for the beginning of the image filename
        :return: .png plot into local repository
        """

        if self.converted:
            if valname in fnames:
                vals = self.f_range
            elif valname in vnames:
                vals = self.Vpk
            elif valname in pnames:
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
                pl.savefig(str(figname) + '.png', dpi=250,
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
            if valname in fnames:
                vals = self.f_range
            elif valname in vnames:
                vals = self.Vpk
            elif valname in pnames:
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

                            pl.savefig(str(dirname) + '/' + str(figname) + '_' + str(count) + '.png', dpi=250,
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

                    pl.savefig(str(figname) + '.png', dpi=250,
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
            if valname in fnames:
                vals = self.f_range
            elif valname in vnames:
                vals = self.Vpk
            elif valname in pnames:
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
                for hour in hours:
                    print('\nResolving hour: ' + str(indexhr) + ' ('
                          + str(floor((indexhr / len(hours)) * 100)) + '% complete) \n'
                          + '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

                    index = 0
                    for srange in hour:
                        # print('Resolving subrange: ' + str(index) + ' ('
                        #       + str(floor((index / len(hour)) * 100)) + '% complete)')

                        binlims = np.arange(-2.5, 2.6, 0.1)

                        f = Fitter(srange, bins=binlims, timeout=10, distributions='common')
                        f.fit()
                        self.bestFits.append(f.get_best())

                        index += 1

                    indexhr += 1

                frange = self.f_range if not self.filtered else self.f_range_filt
                prange = self.Vdb_range if not self.filtered else self.Vdb_range_filt

                yrange = frange if (valname in fnames) else prange

                # Sets y range of plot
                if ylim is None:
                    # truncates maximum and minimum values
                    bottom = round_down(yrange.min(), 1)
                    top = round_up(yrange.max(), 1)

                    # snapping to +- 1, 1.2, 1.5 or 2 for doppler vals
                    if valname in fnames:
                        if bottom > -2:
                            bottom = -1 if (bottom > -1) else (
                                -1.2 if (bottom > -1.2) else (-1.5 if (bottom > -1.5) else -2))
                        if top < 2:
                            top = 1 if (top < 1) else (1.2 if (top < 1.2) else (1.5 if (top < 1.5) else 2))

                    ylim = [bottom, top]

                fSize = fSize
                fig = plt.figure(figsize=(19, 10))  # inches x, y with 72 dots per inch
                ax1 = fig.add_subplot(111)
                ax1.plot(self.t_range, yrange, color='k')  # color k for black
                ax1.set_xlabel('UTC Hour', fontsize=fSize)
                ax1.set_ylabel('Doppler shift, Hz', fontsize=fSize)
                ax1.set_xlim(0, 24)  # UTC day
                ax1.set_xticks(np.arange(0, 25, 2))
                ax1.set_ylim(ylim)  # -1 to 1 Hz for Doppler shift
                ax1.grid(axis='x', alpha=1)
                ax1.tick_params(axis='x',
                                labelsize=fSize - 2)  # ax1.set_xlim([-2.5, 2.5])  # 0.1Hz Bins (-2.5Hz to +2.5Hz)
                ax1.tick_params(axis='y', labelsize=fSize - 2)

                rng = np.arange(0, len(self.bestFits))
                fitTimeRange = (rng / len(self.bestFits)) * 24
                # fitTimeRange = [(i / len(self.bestFits)) * 24 for i in range(0, len(self.bestFits))]
                self.bestFits = [list(i.keys())[0] for i in self.bestFits]
                # self.bestFits = [list(i.keys())[0] for i in self.bestFits]

                ax3 = ax1.twinx()
                ax3.scatter(fitTimeRange, self.bestFits, color='r')
                ax3.set_ylabel('Best Fit PDF', color='r', fontsize=fSize)
                ax3.grid(axis='y', alpha=0.5)
                ax3.tick_params(axis='y', labelsize=fSize - 2)
                for tl in ax3.get_yticklabels():
                    tl.set_color('r')

                self.sunPosOver(fSize)

                plt.title('WWV 10 MHz Doppler Shift Distribution PDFs \n'  # Title (top)
                          # 'Node: N0000020    Gridsquare: FN20vr \n'
                          # 'Lat=40.40.742018  Long=-74.178975 Elev=50M \n'
                          + self.date + ' UTC',
                          fontsize=fSize)

                plt.savefig(str(figname) + '.png', dpi=250, orientation='landscape')
                plt.close()

            else:
                print("Please provide a valid valname!")
        else:
            print('Data units not yet converted! \n'
                  'Attempting unit conversion... \n'
                  'Please try again.')
            self.units()


class GrapeHandler:
    def __init__(self, dirnames, filt=False, comb=True):
        """
        Dynamically creates and manipulates multiple instances of the Grape object using a specified data directory

        :param dirnames: string value for the local directory in which the intended data files (.csv) are located
        :param filt: boolean value dictating whether or not each grape is filtered upon loading (default False)
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

        for directory in dirnames:
            if os.path.exists(directory):
                pass
            else:
                self.valid = False
                break

        if self.valid:

            filenames = []
            for directory in dirnames:
                # iterate over files in that directory
                for filename in os.scandir(directory):
                    if filename.is_file():
                        filenames.append('./' + directory + '/' + filename.name)

            # Unneeded sorting step (only feed sorted files)
            # filenames.sort(key=lambda f: int(sub('\D', '', f)))

            for filename in filenames:
                g = Grape(filename, filt=filt)
                if g.loaded:
                    self.grapes.append(g)

            if len(self.grapes) != 0:
                self.month = self.grapes[0].date[0:7]  # Attributes the date of the first grape

                if comb:
                    for gid, grape in enumerate(self.grapes):
                        vals = grape.getTFPr()  # get time, freq and power from grape
                        t = vals[0]
                        f = vals[1]
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
        else:
            print('One or more of the provided directories do not exist on the local path! \n'
                  'Please try again. \n')

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
        plt.savefig(str(figname) + '.png', dpi=250, orientation='landscape')
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
                secs += vals[index:index + secrange]
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
                                                                                     'Node: N0000020    Gridsquare: FN20vr \n'
                                                                                     'Lat=40.40.742018  Long=-74.178975 Elev=50M \n'
                          + self.month + ' UTC',
                          fontsize='10')

                plt.savefig(str(dirname) + '/' + str(figname) + str(count) + '.png', dpi=250,
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

    def yearMedTrend(self, figname):
        """
        Plots the daily medians of all of GrapeHandler's grapes across the year

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

        for m in monthindex:
            plt.axvline(x=m, color='y', linewidth=3, linestyle='dashed', alpha=0.3)

        plt.xlabel('Month', fontsize=22)
        plt.ylabel('Median Doppler Shift, Hz', fontsize=22)
        # plt.ylim([-1.5, 1.5])
        plt.xticks(monthindex, months)
        plt.tick_params(axis='x', labelsize=20)
        plt.tick_params(axis='y', labelsize=20)
        plt.grid(axis='x', alpha=0.3)
        plt.grid(axis='y', alpha=1)
        plt.legend(["Sun Up Medians",
                    "Sun Down Medians"], fontsize=22)

        plt.title('WWV 10 MHz Doppler Shift Median Trend \n',  # Title (top)
                  fontsize=22)
        plt.savefig(str(figname) + '.png', dpi=250, orientation='landscape')

        # print(str(max(self.dMeds)))
        # print(str(max(self.nMeds)))
        # print()
        # print(str(min(self.dMeds)))
        # print(str(min(self.nMeds)))

        plt.show()
        plt.close()

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
                secs += vals[index:index + secrange]
            qt = np.quantile(secs, qmarks)
            for i, q in enumerate(qts):
                q.append(qt[i])

            index += secrange

        if ylim is None:
            ylim = [-1, 1]

        fSize = fSize

        t_range = self.grapes[self.bestgid].t_range
        t_range = t_range[::secrange]

        fig = plt.figure(figsize=(19, 10))  # inches x, y with 72 dots per inch
        ax1 = fig.add_subplot(111)
        for i in range(0, 5):
            ax1.plot(t_range, qts[i], qstyle[i][0], alpha=qstyle[i][1], linewidth=2)

        self.grapes[self.bestgid].sunPosOver(fSize)

        ax1.set_xlabel('UTC Hour', fontsize=fSize)
        ax1.set_ylabel('Doppler shift, Hz', fontsize=fSize)
        ax1.set_xlim(0, 24)  # UTC day
        ax1.set_ylim(ylim)  # -1 to 1 Hz for Doppler shift
        ax1.set_xticks(np.arange(0, 25, 2))
        ax1.tick_params(axis='x', labelsize=20)  # ax1.set_xlim([-2.5, 2.5])  # 0.1Hz Bins (-2.5Hz to +2.5Hz)
        ax1.tick_params(axis='y', labelsize=20)  # ax1.set_xlim([-2.5, 2.5])  # 0.1Hz Bins (-2.5Hz to +2.5Hz)
        ax1.grid(axis='x', alpha=1)
        ax1.grid(axis='y', alpha=0.5)
        # ax1.legend(['25th Q', '40th Q', 'Median', '60th Q', '75th Q'], fontsize=fSize)

        plt.title('WWV 10 MHz Doppler Shift Plot (Q 25, 40, 50, 60, 75) \n'  # Title (top)
                  + '# of Grapes: ' + str(len(self.grapes)) + ' || '
                  + self.month,
                  # + '2022',
                  fontsize=fSize)
        plt.savefig(str(figname) + '.png', dpi=250, orientation='landscape')
        plt.close()

    def spreadTrend(self, figname, minBinLen=5, fSize=22):
        """
        Plots the difference between the 25th and 75th percentiles of the data (doppler shift) over time

        :param figname: string value for the beginning of each image filename
        :param minBinLen: an integer value for the length of every time bin (minutes)
        :return: .png plot into local repository
        """

        secrange, minrange = mblHandle(minBinLen)

        # Gets the months and years for each grape
        monthMark = [0]*len(self.grapes)
        yearMark = [0]*len(self.grapes)
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
                    secs += vals[index:index + secrange]
                qt = np.quantile(secs, qmarks)
                for i, q in enumerate(qts):
                    q.append(qt[i])
                index += secrange

            monthDiff.append([q75[i]-q25[i] for i in np.arange(0, len(q75))])

        # takes the t_range from the grape with the most data (longest day)
        t_range = self.grapes[self.bestgid].t_range
        t_range = t_range[::secrange]

        count = 0
        for diff in monthDiff:
            fig = plt.figure(figsize=(19, 10))  # inches x, y with 72 dots per inch
            ax1 = fig.add_subplot(111)
            ax1.plot(t_range, diff, linewidth=2)
            # t_range = [t_range[i] + 24 for i in range(0,len(t_range))]

            ax1.set_xlabel('Time UTC', fontsize=fSize)
            ax1.set_ylabel('IQR, Hz', fontsize=fSize)
            ax1.set_ylim(0, 1.2)  # UTC day
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
            plt.savefig(str(figname) + '.png', dpi=250, orientation='landscape')
            plt.close()
            count += 1


def movie(dirname, gifname, fps=10):
    """
    Combines the images in the provided directory into a gif of the specified framerate

    :param dirname: string value for the name of the local directory containing
    the images (.png) to be processed into a gif
    :param gifname: string value for the name of the produced gif
    :param fps: integer value for the produced gif's frames per second (default 10)
    :return: a gif of the combined files in the local repository
    """

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

        imageio.mimsave('./' + gifname + '.gif',  # output gif
                        frames,  # array of input frames
                        fps=fps)  # optional: frames per second

        print('.gif success')

    else:
        print('That directory does not exist on the local path! \n'
              'Please try again.')


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


def to_hr(timestamp):
    """
    Converts timestamps created for sun position times (hr, min, sec) by sunpy into decimal hours

    :param timestamp: timestamp object containing hr, min, sec
    :return: time in decimal hours
    """

    hr = timestamp.hour
    mi = timestamp.minute
    sec = timestamp.second

    convhr = hr + mi / 60 + sec / 3600

    return convhr
