import grape
import os

months = ('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec')
mID = (str(i) for i in range(1, 13))
monthDict = dict(zip(months, mID))
data_dir = 'newark_data/'
beacon = 'wwv'
year = '2023'
dest_dir = 'NJ_spreads/spreads_' + year

# ylim = [0, 0.035]
# ylim = [-0.020, 0.020]
ylim=None

# gh = grape.GrapeHandler(['CO_data/may_2021'], tShift=True)
# gh.tileTrend('test', ylim=ylim)

# for m in months:
#     # gh = grape.GrapeHandler([data_dir + beacon + '_' + m + '_' + year])
#     gh = grape.GrapeHandler([data_dir + m + '_' + year])
#     if gh.valid:
#         gh.tileTrend(dest_dir + '/' + monthDict[m] + '_tileTrend_' + year, ylim=ylim)

for m in months:
    gh = grape.GrapeHandler([data_dir + beacon + '_' + m + '_' + year])
    # gh = grape.GrapeHandler([data_dir + m + '_' + year])
    if gh.valid:
        gh.spreadTrend(dest_dir + '/' + monthDict[m] + '_IQR_' + year, ylim=ylim)

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
