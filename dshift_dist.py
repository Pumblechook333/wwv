import matplotlib.pyplot as plt
# from scipy.signal import filtfilt, butter
# from collections import Counter

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

# FILTERORDER = 3     # Order (Falloff Rate)
# FILTERBREAK = .005  # Critical Freq. (Freq to begin falloff)
# b, a = butter(FILTERORDER, FILTERBREAK, analog=False, btype='low')
#
# freq_filt = filtfilt(b, a, freq)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Converting data to correct units

f_range = [(f-10e6) for f in freq]                      # Doppler shifts (del from 10GHz)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Counting individual occurrences

# f_counts = Counter(f_range)
# f_keys = f_counts.keys()
# f_values = f_counts.values()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot the count data

binlims = [i/10 for i in range(-25, 26, 1)]     # 0.1Hz Bins (-2.5Hz to +2.5Hz)

fig = plt.figure(figsize=(19, 10))              # inches x, y with 72 dots per inch
ax1 = fig.add_subplot(111)
ax1.hist(f_range, color='r', edgecolor='k', bins=binlims)
ax1.set_xlabel('Doppler Shift, Hz')
ax1.set_ylabel('Counts, N', color='r')
ax1.set_xlim([-2.5, 2.5])                       # 0.1Hz Bins (-2.5Hz to +2.5Hz)
ax1.set_xticks(binlims[::2])

plt.title('WWV 10 MHz Doppler Shift Distribution Plot \n'                   # Title (top)
          'Node: N0000020    Gridsquare: FN20vr \n'
          'Lat=40.40.742018  Long=-74.178975 Elev=50M \n'
          '2021-03-24 UTC',
          fontsize='10')
plt.savefig('dshift_dist_unfilt_plot.png', dpi=250, orientation='landscape')

# plt.show()

