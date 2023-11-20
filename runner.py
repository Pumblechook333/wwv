import grape
import matplotlib.pyplot as plt

# months = ('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec')
# mID = (str(i) for i in range(1, 13))
# monthDict = dict(zip(months, mID))
# state = 'NJ'
# data_dir = '%s_data/' % state
# beacon = 'wwv'
# year = '2022'
# dest_dir = '%s_spreads/spreads_%s' % (state, year)
# test = ['NJ_data/testing']

# gh = grape.gen_yearDopPlot(state='NJ', year='2023', n=10*60, monthRange=[0, 6], p=True)

year = '2023'

gh = grape.unpickle_grape('NJ_%s_data_ss10min.pkl' % year)
gh.yearDopPlot('%s_%s_%s' % ('yearDopShift', 'NJ', year))

# gh = grape.GrapeHandler(['ann_ecl_23'], filt=True, n=1)
# gh = grape.unpickle_grape('2023_ecl_gh.pkl')
# gh.dopPlotOver('test', ylim=[-1.5, 2])

# yr = '22'
# if not os.path.exists('cleveland_doppow_unf'+yr):
#     os.makedirs('cleveland_doppow_unf'+yr)
