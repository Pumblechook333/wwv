import grape
import pandas as pd
import matplotlib.pyplot as plt

NJ_DATA_PATH = 'C:/Users/sabas/Documents/NJIT/Work/wwv/DATA/NJ_data'
K2MFF_SIG = 'T000000Z_N0000020_G1_FN20vr_FRQ_WWV10'


def eclipse_grapehandler(data_in: str='DATA/tot_eclipse_24', pickle: str='DATA/tot_ecl_24_test', 
                         plot_out: str='tot_ecl_24_edit', gen: bool=False, tl: list=None, ylim: list=None,
                         tgt: int=0):
    
    if not tl:
        tl = ['', 'Apr 1', 'Apr 2', 'Apr 3', 'Apr 4', 'Apr 5', 'Apr 6', 'Apr 7', 'Apr 8']
    if not ylim:
        ylim = [-1.5, 2.0]

    if gen:
        g = grape.GrapeHandler([data_in], filt=True, comb=True, med=False, tShift=False, n=1)
        grape.pickle_grape(g, filename=pickle)
    else:
        g = grape.unpickle_grape(pickle + '.pkl')

    ax = g.dopPlotOver(figname=plot_out, tl=tl, ylim=ylim, 
                       tgt=tgt, sza=False,
                       )
    
    ecl24_path = 'DATA/eclipse_data/total/eclipse_obsc_2024-04-08.csv'
    ecl23_path = 'DATA/eclipse_data/annular/eclipse_obsc_2023-10-14.csv'

    if '23' in data_in or '23' in pickle or '23' in plot_out:
        ecl_df = pd.read_csv(ecl23_path)
    else:
        ecl_df = pd.read_csv(ecl24_path)
    
    def get_time(datetime):
        time = datetime.split(' ')[1]
        return time
    ecl_df['ut'] = ecl_df['ut'].apply(get_time)

    def to_hrs(time):
        time_split = time.split(':')
        hour = int(time_split[0])
        minute = int(time_split[1])
        second = int(time_split[2])

        ut = hour + minute / 60 + second / 3600

        return ut
    ecl_df['ut'] = ecl_df['ut'].apply(to_hrs)

    ax2 = ax.twinx()
    ax2.plot(ecl_df['ut'], ecl_df['obsc'], 'c', linewidth=2)
    ax2.set_xlim(0,24)
    ax2.set_ylim(0,1)
    ax2.tick_params(axis='x', labelsize=20)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.set_ylabel('Eclipse Obscuration', color='c', fontsize=22)
    ax2.tick_params(axis='y', colors='c', labelsize=20, direction='out', pad=20)
    ax2.spines['right'].set_color('c')

    plt.tight_layout()
    plt.savefig('testing_ecl.png')


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
    """
    Generate best fits overplot for a single grape

    eg. best_fits(2021, 7, 1, mbl=30)
    """

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
                   mbl=minBinLen,
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

    year = 23
    start = 7 if year == 23 else 1
    end = 14 if year == 23 else 8
    keyword = 'annular' if year == 23 else 'total'
    month = 'Oct' if year == 23 else 'Apr'

    # Set ticklabels
    tl = [(f'{month} '+ str(i)) for i in range (start,end+1)]
    # Set y limit
    ylim  = [-1.5, 2.0]
    # Point to correct event directory
    event = f'{keyword}/ecl{year}_{start}_{end}'
    # Declare paths
    data_in = f'DATA/eclipse_data/{event}/'
    pickle = f'PICKLES/{event}'
    plot_out = event
    gen = False     # Boolean for regenerating pickle file
    tgt = 7 if year == 23 else 7

    # Plot eclipse data
    eclipse_grapehandler(data_in, pickle, plot_out,
                         gen, tl=tl, ylim=ylim, tgt=tgt)
    
    # directory = 'DATA/eclipse_data/total/ecl24_5_15/'
    # valname = 'f'
    # minBinLen = 5
    # ylim = None
    # fSize = 22

    # df = pd.DataFrame()
    # for i in range(5,16):

    #     figname = directory + 'elc24' + str(i)

    #     # Create Plot
    #     g = grape.Grape(directory + str(i) + '.csv', filt=True)

    #     g.bestFitsPlot(valname, figname,
    #                 mbl=minBinLen,
    #                 ylim=ylim,
    #                 fSize=fSize)
        
    #     df[str(i) + '_t'] = g.fitTimeRange
    #     df[str(i) + '_f'] = g.bestFits

    # df.to_csv('best_fits_ecl24.csv', index=False)