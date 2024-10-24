import grape

NJ_DATA_PATH = 'C:/Users/sabas/Documents/NJIT/Work/wwv/DATA/NJ_data'
K2MFF_SIG = 'T000000Z_N0000020_G1_FN20vr_FRQ_WWV10'


def eclipse_grapehandler():
    g = grape.GrapeHandler(['tot_eclipse_24'], filt=True, tShift=False)
    grape.pickle_grape(g, filename='tot_ecl_24_test')

    g = grape.unpickle_grape('tot_ecl_24_test.pkl')

    tl = ['', 'Apr 1', 'Apr 2', 'Apr 3', 'Apr 4', 'Apr 5', 'Apr 6', 'Apr 7', 'Apr 8']
    g.dopPlotOver(figname='tot_ecl_24_edit', tl=tl, ylim=[-1.3, 1.5])


def doppowplots():
    day = '08'
    month = 'apr'
    year = '2024'
    folder = f'wwv_{month}_{year}'
    fname = f'2021-07-{day}T000000Z_N0000020_G1_FN20vr_FRQ_WWV10.csv'
    # datapath = f'{NJ_DATA_PATH}/{folder}/{fname}'

    # datapath = 'C:/Users/sabas/Documents/NJIT/Work/wwv/DATA/flare_nov_23/2023-11-28T000000Z_N0000020_G1_FN20vr_FRQ_WWV10.csv'
    datapath = f'C:/Users/sabas/Documents/NJIT/Work/wwv/DATA/tot_eclipse_24/2024-04-{day}T000000Z_N0000020_G1_FN20vr_FRQ_WWV10.csv'
    # datapath = f'C:/Users/sabas/Documents/NJIT/Work/wwv/DATA/ann_ecl_23/2023-10-{day}T000000Z_N0000020_G1_FN20vr_FRQ_WWV10.csv'
    # datapath = f'C:/Users/sabas/Documents/NJIT/Work/wwv/DATA/may_2024/2024-05-{day}T000000Z_N0000020_G1_FN20vr_FRQ_WWV10.csv'

    g = grape.Grape(datapath,
                    convun=True, filt=True,
                    med=False, count=False,
                    n=1)

    # figname = f'eclipse_24_{month}{day}_dopshift_fit'
    figname = f'eclipse_24_{month}{day}_pwr'
    dirname = f'FIGURES/data_request/{figname}'
    # g.bestFitsPlot('f', f'{dirname}', minBinLen=5)
    g.dopPowPlot(f'{dirname}', ylim=[-80, 0], val='pwr', times="Midpoint", local=True)

    # figname = f'eclipse_24_{month}{day}_iltime_doppwr'
    # # figname = f'quiet_{year}_{month}{day}_doppwr'
    # dirname = f'FIGURES/data_request/{figname}'
    # g.dopPowPlot(f'{dirname}', ylim=[-1.25, 1.25], val='dop', axis2='pwr')


def best_fits(year: int, month: int, day: int, **kwargs):
    # %%
    # Get grape for specified date
    g = grape.grapeLoad(year, month, day, **kwargs)

    # %%
    # Get plot parameters
    valname = kwargs.get('valname', 'f')
    minBinLen = kwargs.get('mbl', 5)
    ylim = kwargs.get('ylim', None)
    fSize = kwargs.get('fsize', 22)
    figname = grape.figname(**kwargs)

    # %%
    # Create Plot
    g.bestFitsPlot(valname, figname,
                   minBinLen=minBinLen,
                   ylim=ylim,
                   fSize=fSize)


def solarzenith():

    datapath = 'C:/Users/sabas/Documents/NJIT/Work/wwv/DATA/NJ_data/wwv_jul_2021/2021-07-01T000000Z_N0000020_G1_FN20vr_FRQ_WWV10.csv'
    g = grape.Grape(filename=datapath,
                    convun=True, filt=True,
                    med=False, count=False,
                    n=1)

    # figname = 'rtplot'
    # dirname = f'FIGURES/{figname}'
    # g.dopRtPlot(f'{dirname}/{figname}')

    figname = 'szatrace_nomark'
    dirname = f'FIGURES/aztrace'
    g.dopPowPlot(f'{dirname}/{figname}', axis2='sza', SPO=False)


if __name__ == '__main__':

    best_fits(2021, 7, 1, mbl=30)
