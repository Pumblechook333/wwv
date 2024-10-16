from suncalc import get_position, get_times
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np


def plot_sza(height: float = 0.0):
    lon = (360-90)
    lat = 42

    trange = np.arange(0, 24, 0.1)  # Every hour of the day
    date = datetime(2021, 7, 1, 0, 0, 1)  # July 1, 2021

    delday = timedelta(hours=24)
    times = get_times(date+delday, lon, lat)

    trace = []
    for t in trange:
        d = timedelta(hours=float(t))   # Convert to fraction of an hour
        pos = get_position((date + d), lon, lat)
        trace.append(pos['altitude'] * (180 / np.pi))   # Degrees

    plt.figure()
    plt.plot(trange, trace, color='k')

    ts = [times['sunrise'],
          times['solar_noon'],
          times['sunset']]
    cs = ['y', 'g', 'b']
    for i, t in enumerate(ts):
        if t.day == date.day:
            x = (t.hour + t.minute / 60 + t.second / 3600)
            plt.axvline(x, color=cs[i])

    ticks = np.arange(0, 24, 2)
    plt.xticks(ticks)
    plt.grid()


if __name__ == '__main__':
    plot_sza()

    plt.show()
    plt.close()
