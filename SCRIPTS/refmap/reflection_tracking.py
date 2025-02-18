# Import packages
import os
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from grape import mpt_coords

WWV =   [40.6776, -105.0461]
K2MFF = [40.742018, -74.178975]

def plotrefs(bm: Basemap = None, data: dict = None):
    df0 = data['max_heights'][0][0]
    
    nhours = df0.shape[0]
    nelevs = df0.shape[1]

    nmodes = 2
    nhops = 4
    
    bm = basemap_setup()
    bm = drawlines(bm)

    for mode in range(0,nmodes):
        print(f"\nMode {mode} \n")
        for hop in range(0, nhops):
            print(f"Hop {hop}")
            mhs = data['max_heights'][mode][hop]
            lats = data['lats'][mode][hop]
            lons = data['lons'][mode][hop]

            for hour in range(0, nhours-1):
                ray = 0
                time = mhs.iloc[hour]
                lat = lats.iloc[hour]
                lon = lons.iloc[hour]

                while ray < (nelevs-1):
                    height = float(time.iloc[ray])
                    if height == 0:
                        pass
                    else:
                        if hop > 0:
                            la = lat.iloc[ray].split()
                            lo = lon.iloc[ray].split()

                            m_la, m_lo = ground2skyref(la, lo)

                            no_duplicates = len(lo) == len(set(lo))
                            if no_duplicates:
                                print (lat.iloc[ray], " | " , lon.iloc[ray])
                                for point in range(0, hop+1):
                                    la_coord = m_la[point]
                                    lo_coord = m_lo[point]

                                    plotpoint(bm, la_coord, lo_coord, mode=mode, hop=hop, height=height)

                        else:
                            print (lat.iloc[ray], " | " , lon.iloc[ray])

                            la = lat.iloc[ray]
                            lo = lon.iloc[ray]

                            m_la, m_lo = ground2skyref(la, lo)

                            la_coord = m_la
                            lo_coord = m_lo

                            plotpoint(bm, la_coord, lo_coord, mode=mode, hop=hop, height=height)

                    ray += 1
    
    plotpoint(bm, WWV[0], WWV[1], style='k*', markersize=20)
    plotpoint(bm, K2MFF[0], K2MFF[1], style='k^', markersize=15)


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


def drawlines(bm):
    parallels = np.arange(0., 81, 10.)
    bm.drawparallels(parallels, labels=[True, True, False, False])
    meridians = np.arange(10., 351., 20.)
    bm.drawmeridians(meridians, labels=[False, False, False, True])

    return bm


def plotpoint(bm: Basemap, la: float, lo: float, mode: int = 0, hop = 'r', height: int = 5, **kwargs):
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

    marker = {0: 'o', 1: 'x'}
    colors = {0: 'r', 1: 'g', 2: 'b', 3: 'm'}
    sz = height / 20

    lon, lat = lo, la
    xpt, ypt = bm(lon, lat)

    style = kwargs.get('style', False)

    if style:
        size = kwargs.get('markersize', 10)
        bm.plot(xpt, ypt, style, markersize=size)
    elif mode == 0:
        bm.plot(xpt, ypt, marker[mode], color=colors[hop], markersize=sz)
    else:
        bm.plot(xpt, ypt, marker[mode], color=colors[hop], markersize=sz, markeredgewidth = 3)


def savemap(save_path: str = 'test_map.png', show: bool = False):
    if not save_path.endswith('.png'):
        save_path = save_path + '.png'

    plt.savefig(save_path, dpi=300)

    if show:
        plt.show()

    plt.close()


def getdata(expdir: str = None):

    folders = os.listdir(expdir)
    print(folders)

    modes = {"O": 0, "X": 1}

    data = {}

    # Sort data
    for folder in folders:
        
        xmode = [None, None, None, None]
        omode = [None, None, None, None]
        da = [xmode, omode]

        folder += '/'
        files = os.listdir(expdir + folder)

        for f in files:
            df = pd.read_csv(expdir + folder + f, header=None)

            hops = int(f[-10])
            mode = modes.get(f[-16])

            da[mode][hops-1] = df
        
        data[folder.split('/')[0]] = da
    
    return data


def ground2skyref(la, lo):
    if isinstance(la, list):
        npts = len(la)
        la = [float(y) for y in la]
        lo = [float(x) for x in lo]
    else:
        npts = 1
        la = float(la)
        lo = float(lo)

    m_la = []
    m_lo = []

    if npts == 1:
        lat, lon = mpt_coords(WWV[0], WWV[1], la, lo)
        
        m_la.append(lat)
        m_lo.append(lon)
    else:
        last_la = WWV[0]
        last_lo = WWV[1]

        for pt in range(0, npts):
            la_i = la[pt]
            lo_i = lo[pt]

            lat, lon = mpt_coords(last_la, last_lo, la_i, lo_i)
        
            m_la.append(lat)
            m_lo.append(lon)

            last_la = la_i
            last_lo = lo_i
    
    return m_la, m_lo



if __name__ == "__main__":

    plt.figure(figsize=(8, 6), layout='tight')
    bm = basemap_setup()
    drawlines(bm)

    expdir = 'SCRIPTS/refmap/export/'

    data = getdata(expdir)

    plotrefs(bm, data)

    plt.title("Reflection Sites Across the Day")

    save = True
    if save:
        savemap('refpoint_dist.png', show=True)
    else:
        plt.show()
        plt.close()
