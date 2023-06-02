import matplotlib.pyplot as plt
# from scipy.signal import filtfilt, butter
from math import floor
from grape import *

time, freq, Vpk = Grape('wwv_data.txt').getTFV()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Filtering the data with order 3 Butterworth low-pass filter
# Butterworth Order: https://rb.gy/l4pfm

# FILTERORDER = 3     # Order (Falloff Rate)
# FILTERBREAK = .005  # Critical Freq. (Freq to begin falloff)
# b, a = butter(FILTERORDER, FILTERBREAK, analog=False, btype='low')
#
# freq_filt = filtfilt(b, a, freq)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Converting data to correct units

f_range = [(f-10e6) for f in freq]                      # Doppler shifts (del from 10GHz)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make subsections and begin plot generation
f_subranges = []             # contains equally sized ranges of data
# rangewidth = 60*5          # make each range 5-minute long
rangewidth = 60*60           # make each range 5-minute long

index = 0
while not index > len(f_range):
    f_subranges.append(f_range[index:index+rangewidth])
    index += rangewidth

f_hours = []                # contains 24 hour chunks of data
# rangewidth = 12           # 12 5-minute chunks each
rangewidth = 1              # 1 60-minute chunk each

index = 0
while not index > len(f_subranges):
    f_hours.append(f_subranges[index:index+rangewidth])
    index += rangewidth

indexhr = 0
for f_hour in f_hours:
    print('\nResolving hour: ' + str(indexhr) + ' ('
          + str(floor((indexhr / len(f_hours)) * 100)) + '% complete) \n'
          + '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    indexhr += 1

    index = 0
    for srange in f_hour:

        print('Resolving subrange: ' + str(index) + ' ('
              + str(floor((index / len(f_hour)) * 100)) + '% complete)')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Plot the subsections
        # binlims = [i/10 for i in range(-10, 11, 1)]   # 0.1Hz Bins (-1Hz to +1Hz)
        binlims = [i/10 for i in range(-25, 26, 1)]     # 0.1Hz Bins (-2.5Hz to +2.5Hz)

        fig = plt.figure(figsize=(19, 10))              # inches x, y with 72 dots per inch
        ax1 = fig.add_subplot(111)
        ax1.hist(srange, color='r', edgecolor='k', bins=binlims)
        ax1.set_xlabel('Doppler Shift, Hz')
        ax1.set_ylabel('Counts, N', color='r')
        # ax1.set_xlim([-1, 1])                         # 0.1Hz Bins (-1Hz to +1Hz)
        ax1.set_xlim([-2.5, 2.5])                       # 0.1Hz Bins (-2.5Hz to +2.5Hz)
        ax1.set_xticks(binlims[::2])

        plt.title('WWV 10 MHz Doppler Shift Distribution Plot \n'
                  'Hour: ' + str(indexhr) + ' || 5-min bin: ' + str(index) + ' \n'                   # Title (top)
                  'Node: N0000020    Gridsquare: FN20vr \n'
                  'Lat=40.40.742018  Long=-74.178975 Elev=50M \n'
                  '2021-03-24 UTC',
                  fontsize='10')
        plt.savefig('dshift_5min_dist_plots/dshift_dist_plot' + str(indexhr) + '_' + str(index) + '.png', dpi=250, orientation='landscape')

        plt.close()

        index += 1
