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

    watercolor = '#A8B6CC'
    landcolor = "#EEBAB4"
    
    m.drawcoastlines()
    m.drawmapboundary(fill_color=watercolor)
    m.fillcontinents(color=landcolor, lake_color=watercolor)
    m.drawstates()

    return m


def drawlines(bm):
    parallels = np.arange(0., 81, 10.)
    bm.drawparallels(parallels, labels=[True, True, False, False])
    meridians = np.arange(10., 351., 20.)
    bm.drawmeridians(meridians, labels=[False, False, False, True])

    return bm


def drawshape(bm: Basemap, shppth: str, shpn: str = 'Shape', color: str = 'k'):
    bm.readshapefile(shppth, shpn, color=color)

    return bm


def plotgrapes(m, color='k'):
    with open('node_latlons.txt', 'r') as lines:
        for line in lines:
            if line[0:2] == 'N0':
                splitline = line.split(', ')

                lat = splitline[1]
                lon = splitline[2]

                xpt, ypt = m(lon, lat)
                m.plot(xpt, ypt, marker='o', color=color)  # plot a blue dot there


def plotpoint(bm: Basemap, la: float, lo: float, txt: str, marker: str = 'bo', color: str = 'k', sz: int = 5):
    '''
    Method to plot point on basemap (with optional annotation)

    :param bm: Basemap targer
    :param la: Latitude
    :param lo: Longitude
    :param txt: Optional Annotation
    :param marker: 2-character string for color and marker shape
    :param sz: integer for marker size
    :return:
    '''
    lon, lat = lo, la  # Location of WWV
    xpt, ypt = bm(lon, lat)
    # lonpt, latpt = m(xpt, ypt, inverse=True)
    bm.plot(xpt, ypt, marker=marker, color=color, markersize=sz)  # plot a blue dot there

    # plt.text(xpt + 100000, ypt + 100000, f'{txt} (%5.1fW,%3.1fN)' % (lonpt, latpt))


def savemap(save_path: str = 'test_map.jpg', show: bool = False):
    if not save_path.endswith('.jpg'):
        if save_path.endswith('.png'):
            pass
        else:
            save_path = save_path + '.jpg'

    plt.savefig(save_path, dpi=300)

    if show:
        plt.show()

    plt.close()


if __name__ == '__main__':
    plt.figure(figsize=(8, 6), layout='tight')
    m = basemap_setup()
    drawlines(m)

    shape_path = 'C:/Users/sabas/Documents/NJIT/Work/wwv/DATA/eclipse_data/annular/2023eclipse_shapefiles/umbra_lo'
    shape_name = '2023 Eclipse'
    drawshape(m, shape_path, shape_name, '#F05039')

    shape_path = 'C:/Users/sabas/Documents/NJIT/Work/wwv/DATA/eclipse_data/total/2024eclipse_shapefiles/umbra_lo'
    shape_name = '2024 Eclipse'
    drawshape(m, shape_path, shape_name, '#3D65A5')

    markercolor = '#3d65a5'

    plotgrapes(m, 'k')

    plotpoint(m, 40.6776, -105.0461, 'WWV', '*' , markercolor, 20)
    plotpoint(m, 40.742018, -74.178975, 'K2MFF', '^' , markercolor, 15)

    plt.title("Active Grape V1 Stations in North America (2023)")

    save = True
    if save:
        savemap('grapev1_eclipses.jpg', show=True)
    else:
        plt.show()
        plt.close()

