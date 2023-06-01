import matplotlib.pyplot as plt
from scipy.signal import filtfilt, butter
from collections import Counter

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read file, Initiate Containers

# Initializing time, freq and voltage arrays
freq = []

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
    freq.append(float(holder[1]))           # doppler shift list append

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Filtering the data with order 3 Butterworth low-pass filter
# Butterworth Order: https://rb.gy/l4pfm

FILTERORDER = 3     # Order (Falloff Rate)
FILTERBREAK = .005  # Critical Freq. (Freq to begin falloff)
b, a = butter(FILTERORDER, FILTERBREAK, analog=False, btype='low')

freq_filt = filtfilt(b, a, freq)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Converting data to correct units

f_range = [(f-10e6) for f in freq_filt]                      # Doppler shifts (del from 10GHz)
N = len(f_range)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Make subsections and begin plot generation
f_subranges = []
N_ranges = 60*5         # make each range 5 mins long
index = 0

while not index > N:
    f_subranges.append(f_range[index:index+N_ranges])
    index += N_ranges

index = 0

for srange in f_subranges:

    print('Resolving subrange: ' + index)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Counting individual occurrences

    f_counts = Counter(srange)
    f_keys = f_counts.keys()
    f_values = f_counts.values()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot the subsections

    fig = plt.figure(figsize=(19, 10))      # inches x, y with 72 dots per inch
    ax1 = fig.add_subplot(111)
    ax1.hist(srange, color='r', bins=int(N_ranges/(5*6)))         # 10s long bins
    ax1.set_xlabel('Doppler Shift, Hz')
    ax1.set_ylabel('Counts, N (10 sec bin)', color='r')
    ax1.set_xlim([-1, 1])
    for tl in ax1.get_yticklabels():
        tl.set_color('r')

    ax2 = ax1.twinx()
    ax2.scatter(f_keys, f_values, s=5, marker='*', color='k')
    ax2.set_ylabel('Counts, N', color='k')
    ax2.set_ylim(0, 3)                   # -1 to 1 Hz for Doppler shift

    plt.title('WWV 10 MHz Doppler Shift Distribution Plot ' + str(index) + ' \n'                   # Title (top)
              'Node: N0000020    Gridsquare: FN20vr \n'
              'Lat=40.40.742018  Long=-74.178975 Elev=50M \n'
              '2021-03-24 UTC',
              fontsize='10')
    plt.savefig('dshift_5min_dist_plots/dshift_dist_plot' + str(index) + '.png', dpi=250, orientation='landscape')

    plt.close()

    index += 1

