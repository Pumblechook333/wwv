from pvlib import solarposition
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def plot_sza(height: float = 0.0):
    lat, lon = 42, -90

    trange = np.arange(0, 25, 1)  # Every hour of the day

    d1 = datetime(2021, 7, 1)
    d2 = datetime(2021, 7, 1 + 1)
    times = pd.date_range(d1, d2, freq='H')
    solpos = solarposition.get_solarposition(times, lat, lon)

    trace = solpos['zenith'].values

    plt.figure()
    plt.plot(trange, trace, color='k')
    ticks = np.arange(0, 24, 2)
    plt.xticks(ticks)
    plt.grid()


if __name__ == '__main__':
    plot_sza()

    plt.show()
    plt.close()

