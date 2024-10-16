from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np


def basemap_setup():
    w = 5e6
    h = 3e6
    ywa = -3
    xwa = -3
    m = Basemap(width=w, height=h, projection='lcc',
                resolution='c', lat_1=35. + ywa, lat_2=55. + ywa, lat_0=40. + ywa, lon_0=-95. + xwa)

    m.drawcoastlines()
    m.drawmapboundary(fill_color='aqua')
    m.fillcontinents(color='coral', lake_color='aqua')
    m.drawstates()

    return m


def drawlines(m):
    parallels = np.arange(0., 81, 10.)
    m.drawparallels(parallels, labels=[True, True, False, False])
    meridians = np.arange(10., 351., 20.)
    m.drawmeridians(meridians, labels=[False, False, False, True])


def plotgrapes(m):
    with open('SCRIPTS/nodemap/node_latlons.txt', 'r') as lines:
        for line in lines:
            if line[0:2] == 'N0':
                splitline = line.split(', ')

                lat = splitline[1]
                lon = splitline[2]

                xpt, ypt = m(lon, lat)
                m.plot(xpt, ypt, 'ko')  # plot a blue dot there


def plotpoint(m, la, lo, txt):
    lon, lat = lo, la  # Location of WWV
    xpt, ypt = m(lon, lat)
    # lonpt, latpt = m(xpt, ypt, inverse=True)
    m.plot(xpt, ypt, 'bo')  # plot a blue dot there

    # plt.text(xpt + 100000, ypt + 100000, f'{txt} (%5.1fW,%3.1fN)' % (lonpt, latpt))


def savemap():
    plt.savefig('SCRIPTS/nodemap/map_test.png', dpi=250)
    # plt.show()
    plt.close()


if __name__ == '__main__':
    m = basemap_setup()
    drawlines(m)
    plotgrapes(m)

    plotpoint(m, 40.6776, -105.0461, 'WWV')
    plotpoint(m, 40.742018, -74.178975, 'K2MFF')

    plt.title("Active Grape V1 Stations in North America (2023)")

    savemap()
