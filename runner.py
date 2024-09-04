import grape

if __name__ == '__main__':
    # g = grape.GrapeHandler(['tot_eclipse_24'], filt=True, tShift=False)
    # grape.pickle_grape(g, filename='tot_ecl_24_test')

    # g = grape.unpickle_grape('tot_ecl_24_test.pkl')
    #
    # tl = ['', 'Apr 1', 'Apr 2', 'Apr 3', 'Apr 4', 'Apr 5', 'Apr 6', 'Apr 7', 'Apr 8']
    # g.dopPlotOver(figname='tot_ecl_24_edit', tl=tl, ylim=[-1.3, 1.5])

    datapath = 'C:/Users/sabas/Documents/NJIT/Work/wwv/DATA/NJ_data/wwv_jul_2021/2021-07-01T000000Z_N0000020_G1_FN20vr_FRQ_WWV10.csv'
    g = grape.Grape(filename=datapath,
                    convun=True, filt=True,
                    med=False, count=False,
                    n=1)

    figname = 'rtplot'
    dirname = f'FIGURES/{figname}'
    g.dopRtPlot(f'{dirname}/{figname}')

    # figname = 'aztrace'
    # dirname = f'FIGURES/{figname}'
    # g.dopPowPlot(f'{dirname}/{figname}', axis2='sza')

