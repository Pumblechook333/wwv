import matplotlib.pyplot as plt
from scipy.signal import filtfilt, butter
from collections import Counter
import numpy as np
from math import floor
import csv
import os
import imageio
import re

fnames = ['d', 'dop', 'doppler', 'doppler shift', 'f', 'freq', 'frequency']
vnames = ['v', 'volt', 'voltage']
pnames = ['db', 'decibel', 'p', 'pwr', 'power']


class Grape:

    def __init__(self, filename=None, filt=False, convun=True, n=1):
        """
        Constructor for a Grape object

        :param filename: Name of the .txt file where the data is kept in tab delimited format
        :param filt: Boolean for if you are filtering or not (default = F)
        :param convun: Boolean for if you want a unit range to be auto created (default = T)
        :param n: Subsampling term
        """

        # Metadata containers
        self.date = None

        # Raw data containers
        self.time = None
        self.freq = None
        self.Vpk = None
        self.Vdb = None  # Vpk converted to logscale

        # Raw data adjusted to be plotted with correct units
        self.t_range = None
        self.f_range = None
        self.Vdb_range = None

        # Counting variables for collections.counter
        self.f_count = None
        self.Vpk_count = None
        self.Vdb_count = None

        # Flags to keep track of if the load() or units() function have been called, respectively
        self.loaded = False
        self.converted = False

        if filename:
            self.load(filename, n=n)
        if filt:
            self.butFilt()
        if convun:
            self.units()

    def load(self, filename, n=1):
        """
        Script to load grape data from wwv text file into Grape object

        :param filename: Path of the .txt file containing the grape data in the local repo
        :param n: Subsampling term (every nth)
        :return: None
        """

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initializing time, freq and voltage arrays
        self.time = []
        self.freq = []
        self.Vpk = []

        dataFile = open(filename)
        dataReader = csv.reader(dataFile)
        lines = list(dataReader)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Save the header data separately from the plottable data

        header_data = lines[:18]
        # for i in header_data:
        #     print(i)
        # col_title = lines[18].split()               # Titles for each data range

        self.date = str(header_data[0][1]).split('T')[0]

        # Read each line of file after the header
        for line in lines[19::n]:
            date_time = str(line[0]).split('T')
            utc_time = str(date_time[1]).split(':')
            sec = (float(utc_time[0]) * 3600) + \
                  (float(utc_time[1]) * 60) + \
                  (float(utc_time[2][0:2]))

            self.time.append(sec)  # time list append
            self.freq.append(float(line[1]))  # doppler shift list append
            self.Vpk.append(float(line[2]))  # voltage list append

        # Raise loaded flag
        self.loaded = True

    def getTFV(self):
        """
        Getter for Grape object's time, frequency and peak voltage values

        :return: time, freq and Vpk values
        """
        if self.loaded:
            return [self.time, self.freq, self.Vpk]
        else:
            return None, None, None

    def getTFPr(self):
        """
        Getter for Grape object's converted time, frequency and relative power ranges for use in plotting

        :return: time, freq and Vdb ranges
        """
        if self.converted:
            return [self.t_range, self.f_range, self.Vdb_range]
        else:
            return None, None, None

    def butFilt(self, FILTERORDER=3, FILTERBREAK=0.005):
        """
        Filtering the data with order 3 Butterworth low-pass filter
        Butterworth Order: https://rb.gy/l4pfm

        :param FILTERORDER: Order of the Butterworth filter
        :param FILTERBREAK: Cutoff Frequency of the Butterworth Filter
        :return: None
        """

        if self.loaded:
            # noinspection PyTupleAssignmentBalance
            b, a = butter(FILTERORDER, FILTERBREAK, analog=False, btype='low')

            self.freq = filtfilt(b, a, self.freq)
            self.Vpk = filtfilt(b, a, self.Vpk)
        else:
            print("Frequency and Vpk not loaded!")

    def units(self, timediv=3600, fdel=10e6):
        """
        Converting raw/filtered Grape data to correct units for plotting

        :param timediv: Dividing factor for seconds data
        :param fdel: Reference value for doppler shift (10GHz for WWV)
        :return: None
        """

        if self.loaded:
            self.t_range = [(t / timediv) for t in self.time]  # Time range (Hours)
            self.f_range = [(f - fdel) for f in self.freq]  # Doppler shifts (del from 10GHz)
            self.Vdb_range = [10 * np.log10(v ** 2) for v in
                              self.Vpk]  # Relative power (Power is proportional to V^2; dB)

            self.converted = True
        else:
            print('Time, frequency and Vpk not loaded!')

    def dopPowPlot(self, figname):
        """
        Plot the doppler shift and relative power over time of the signal

        :param figname: Filename for the produced .png plot image
        :return: None
        """

        if self.converted:
            fig = plt.figure(figsize=(19, 10))  # inches x, y with 72 dots per inch
            ax1 = fig.add_subplot(111)
            ax1.plot(self.t_range, self.f_range, 'k')  # color k for black
            ax1.set_xlabel('UTC Hour')
            ax1.set_ylabel('Doppler shift, Hz')
            ax1.set_xlim(0, 24)  # UTC day
            ax1.set_ylim([-1, 1])  # -1 to 1 Hz for Doppler shift

            ax2 = ax1.twinx()
            ax2.plot(self.t_range, self.Vdb_range, 'r-')  # NOTE: Set for filtered version
            ax2.set_ylabel('Power in relative dB', color='r')
            ax2.set_ylim(-80, 0)  # Try these as defaults to keep graphs similar.
            # following lines set ylim for power readings in file

            for tl in ax2.get_yticklabels():
                tl.set_color('r')

            plt.title('WWV 10 MHz Doppler Shift Plot \n'  # Title (top)
                      'Node: N0000020    Gridsquare: FN20vr \n'
                      'Lat=40.40.742018  Long=-74.178975 Elev=50M \n'
                      + self.date + ' UTC',
                      fontsize='10')
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
        :return: None
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

            if vals:
                binlims = [i / 10 for i in range(-25, 26, 1)]  # 0.1Hz Bins (-2.5Hz to +2.5Hz)

                fig = plt.figure(figsize=(19, 10))  # inches x, y with 72 dots per inch
                ax1 = fig.add_subplot(111)
                ax1.hist(vals, color='r', edgecolor='k', bins=binlims)
                ax1.set_xlabel('Doppler Shift, Hz')
                ax1.set_ylabel('Counts, N', color='r')
                ax1.set_xlim([-2.5, 2.5])  # 0.1Hz Bins (-2.5Hz to +2.5Hz)
                ax1.set_xticks(binlims[::2])

                plt.title('WWV 10 MHz Doppler Shift Distribution Plot \n'  # Title (top)
                          'Node: N0000020    Gridsquare: FN20vr \n'
                          'Lat=40.40.742018  Long=-74.178975 Elev=50M \n'
                          + self.date + ' UTC',
                          fontsize='10')
                plt.savefig(str(figname) + '.png', dpi=250, orientation='landscape')
                plt.close()
            else:
                print("Please provide a valid valname!")
        else:
            print('Data units not yet converted! \n'
                  'Attempting unit conversion... \n'
                  'Please try again.')
            self.units()

    def distPlots(self, valname, dirname='dshift_dist_plots', figname='dshift_dist_plot', secrange=60 * 5, minrange=12):

        if self.converted:
            if valname in fnames:
                vals = self.f_range
            elif valname in vnames:
                vals = self.Vpk
            elif valname in pnames:
                vals = self.Vdb_range
            else:
                vals = None

            if vals:
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
                        binlims = [i / 10 for i in range(-25, 26, 1)]  # 0.1Hz Bins (-2.5Hz to +2.5Hz)

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
                print("Please provide a valid valname!")
        else:
            print('Data units not yet converted! \n'
                  'Attempting unit conversion... \n'
                  'Please try again.')
            self.units()

    def count(self):
        """
        Employs collections.Counter to produce counts of individual values in grape value ranges

        :return: None
        """

        if self.converted:
            self.f_count = Counter(self.f_range)
            self.Vpk_count = Counter(self.Vpk)
            self.Vdb_count = Counter(self.Vdb_range)
        else:
            print('Data units not yet converted! \n'
                  'Attempting unit conversion... \n'
                  'Please try again.')
            self.units()

    def movie(self, dirname, gifname, fps=10):

        if os.path.exists(dirname):
            # assign directory
            directory = dirname

            filenames = []
            # iterate over files in that directory
            for filename in os.scandir(directory):
                if filename.is_file():
                    filenames.append('./' + directory + '/' + filename.name)

            filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

            frames = []
            for t in range(0, 275):
                image = imageio.v2.imread(filenames[t])
                frames.append(image)

            imageio.mimsave('./' + gifname + '.gif',  # output gif
                            frames,  # array of input frames
                            fps=fps)  # optional: frames per second
        else:
            print('That directory does not exist on the local path! \n'
                  'Please try again.')


class GrapeHandler:
    def __init__(self, dirname):
        self.grapes = []
        self.date = None

        if os.path.exists(dirname):
            # assign directory
            directory = dirname

            filenames = []
            # iterate over files in that directory
            for filename in os.scandir(directory):
                if filename.is_file():
                    filenames.append('./' + directory + '/' + filename.name)

            filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

            for filename in filenames:
                self.grapes.append(Grape(filename))

            self.date = self.grapes[0].date     # Attributes the date of the first grape
        else:
            print('That directory does not exist on the local path! \n'
                  'Please try again.')

    def multGrapeDistPlots(self, dirname, figname, secrange=60 * 5, minrange=12):

        valscomb = []

        for grape in self.grapes:
            vals = grape.getTFPr()  # get time, freq and power from grape
            vals = vals[1]          # select just the freq
            valscomb.append(vals)

        # Make subsections and begin plot generation
        subranges = []  # contains equally sized ranges of data

        index = 0
        while not index > len(vals):
            secs = []
            for vals in valscomb:
                secs += vals[index:index+secrange]
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
                binlims = [i / 10 for i in range(-25, 26, 1)]  # 0.1Hz Bins (-2.5Hz to +2.5Hz)

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