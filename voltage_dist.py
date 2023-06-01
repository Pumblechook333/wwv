import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import filtfilt, butter
from collections import Counter

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read file, Initiate Containers

# Initializing time, freq and voltage arrays
Vpk = []

# Open text file and store all of the lines
f = open("wwv_data.txt")
lines = f.readlines()
f.close()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load data

# Save the header data separately from the plottable data
header_data = lines[:18]
for i in header_data:
    print(i)
col_title = lines[18].split()   # Titles for each data range

# Read each line of file after the header
n = 1                                      # Subsampling term (read every nth term)
for line in lines[19::n]:
    holder = line.split()
    Vpk.append(float(holder[2]))            # voltage list append

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Filtering the data with order 3 Butterworth low-pass filter
# Butterworth Order: https://rb.gy/l4pfm

FILTERORDER = 3     # Order (Falloff Rate)
FILTERBREAK = .005  # Critical Freq. (Freq to begin falloff)
b, a = butter(FILTERORDER, FILTERBREAK, analog=False, btype='low')

Vpk_filt = filtfilt(b, a, Vpk)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Converting data to correct units

Vdb_range = [i for i in Vpk_filt]            # Relative power (Power is proportional to V^2; dB)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Counting individual occurrences

Vdb_counts = Counter(Vdb_range)
Vdb_keys = Vdb_counts.keys()
Vdb_values = Vdb_counts.values()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot the count data

N = len(Vdb_range)

fig = plt.figure(figsize=(19, 10))      # inches x, y with 72 dots per inch
ax1 = fig.add_subplot(111)
ax1.hist(Vdb_range, color='r', bins=int(N/(60*5)))         # color k for black
ax1.set_xlabel('Voltage, V')
ax1.set_ylabel('Counts, N (5 min bin)', color='r')
# ax1.set_xlim([-1, 1])
for tl in ax1.get_yticklabels():
    tl.set_color('r')

ax2 = ax1.twinx()
ax2.scatter(Vdb_keys, Vdb_values, s=5, marker='*', color='k')
ax2.set_ylabel('Counts, N', color='k')
# ax2.set_ylim(0, 3)                   # -1 to 1 Hz for Doppler shift


plt.title('WWV 10 MHz Voltage Distribution Plot \n'                   # Title (top)
          'Node: N0000020    Gridsquare: FN20vr \n'
          'Lat=40.40.742018  Long=-74.178975 Elev=50M \n'
          '2021-03-24 UTC',
          fontsize='10')
plt.savefig('voltage_dist_plot.png', dpi=250, orientation='landscape')

# plt.show()
