import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Initializing time, freq and voltage arrays
time = []
freq = []
Vpk = []

# Open text file and store all of the lines
f = open("wwv_data.txt")
lines = f.readlines()
f.close()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Save the header data separately from the plottable data
header_data = lines[:18]
for i in header_data:
    print(i)
col_title = lines[18].split()   # Titles for each data range

# Read each line of file after the header
n = 30                                      # Subsampling term (read every nth term)
for line in lines[19::n]:
    holder = line.split()

    date_time = holder[0].split('T')
    utc_time = date_time[1].split(':')
    sec = (float(utc_time[0]) * 3600) + \
          (float(utc_time[1]) * 60) + \
          (float(utc_time[2][0:2]))

    time.append(sec)                        # time list append
    freq.append(float(holder[1]))           # doppler shift list append
    Vpk.append(float(holder[2]))            # voltage list append

# print(time[:5])
# print("\n")
# print(freq[:5])
# print("\n")
# print(Vpk[:5])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

t_range = [(t/3600) for t in time]                      # Time range
f_range = [(f-10e6) for f in freq]                      # Doppler shifts
# Vpk_range = [v for v in Vpk]                          # Peak voltage readings
Vdb_range = [10*np.log10(v**2) for v in Vpk]            # Relative power (Power is proportional to V^2)

footprint = 10                                          # Neighbors for median
f_med = median_filter(f_range, size=footprint)          # Median doppler shift
# Vpk_med = median_filter(Vdb, size=footprint)          # Median voltage readings
Vdb_med = median_filter(Vdb_range, size=footprint)      # Median relative power

# Plot the data and make plot title and axes labels
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

# frequency plot
ax1.scatter(t_range, f_range, s=1, marker='*', color='grey')
ax1.plot(t_range, f_med, color='black', lw='1')
# voltage plot
# ax2.scatter(t_range, Vpk_range, s=1, marker='*', color='gold')    # Raw voltage
# ax2.plot(t_range, Vpk_med, color='red', lw='1')
ax2.scatter(t_range, Vdb_range, s=1, marker='*', color='gold')      # Relative Power
ax2.plot(t_range, Vdb_med, color='red', lw='1')

# labels
ax1.set_xlabel('UTC (hr)')                                     # Time axis (bottom)
ax1.set_ylabel('Doppler Shift (Hz)', color='black')            # Doppler shift axis (left)
ax2.set_ylabel('Relative Power (dB)', color='red')             # Relative power axis (right)
plt.title('WWV 10 MHz Doppler Shift Plot \n'                   # Title (top)
          'Node: N0000020    Gridsquare: FN20vr \n'
          'Lat=40.40.742018  Long=-74.178975 Elev=50M \n'
          '2021-03-24 UTC',
          fontsize='10')

# extra formatting
plt.xticks(np.arange(0, 25, 2))     # 24 hours of data, increment by 2
plt.tight_layout()
plt.grid()

plt.savefig('dop_shift.png')
# plt.show()
