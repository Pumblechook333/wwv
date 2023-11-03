import grape

months = ('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec')
mID = (str(i) for i in range(1, 13))
monthDict = dict(zip(months, mID))
state = 'NJ'
data_dir = '%s_data/' % state
beacon = 'wwv'
year = '2022'
dest_dir = '%s_spreads/spreads_%s' % (state, year)
test = ['NJ_data/testing']

# grape.gen_yearDopPlot(state='NJ', year='2022', n=60*5, monthRange=None)

gh = grape.GrapeHandler(['ann_ecl_23'], filt=True, n=1)
gh.dopPlotOver('test', ylim=[-1.5, 2])


# for m in months:
#     # gh = grape.GrapeHandler([data_dir + beacon + '_' + m + '_' + year])
#     gh = grape.GrapeHandler([data_dir + m + '_' + year])
#     if gh.valid:
#         gh.tileTrend(dest_dir + '/' + monthDict[m] + '_tileTrend_' + year, ylim=ylim)

# for m in months:
#     gh = grape.GrapeHandler([data_dir + beacon + '_' + m + '_' + year])
#     # gh = grape.GrapeHandler([data_dir + m + '_' + year])
#     if gh.valid:
#         gh.spreadTrend(dest_dir + '/' + monthDict[m] + '_IQR_' + year, ylim=ylim)

# yr = '22'
# if not os.path.exists('cleveland_doppow_unf'+yr):
#     os.makedirs('cleveland_doppow_unf'+yr)
#
# for m in months:
#     gh = grape.GrapeHandler(['CO_data/' + m+'_20'+yr], filt=False, comb=False, tShift=False)
#     gh.dopPowPlots('cleveland_doppow_unf'+yr+'/' + m+yr, 'doppow', ylim=[-0.1, 0.1])

# g = grape.Grape(data_dir + 'apr_2021/2021-04-01T000000Z_N0000013_S1_DN70ln_FRQ_WWV10.csv', filt=True)
# g.dopPowPlot('test', ylim=ylim)

# grape.movie('monthlyMedTrends2022', 'medTrend2022', 5)
