import grape

months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
mID = [str(i) for i in range(1, 13)]
monthDict = dict(zip(months, mID))
data_dir = 'wwv_data/'
dest_dir = 'testSpread'
beacon = 'wwv'
year = '2023'

# for m in months:
#     gh = grape.GrapeHandler([data_dir + beacon + '_' + m + '_' + year])
#     if gh.valid:
#         gh.tileTrend(dest_dir + '_' + year + '/' + monthDict[m] + 'MedTrend' + year)

# for m in months:
#     gh = grape.GrapeHandler([data_dir + beacon + '_' + m + '_' + year])
#     if gh.valid:
#         gh.spreadTrend(dest_dir + '/' + monthDict[m] + 'IQR' + year)

g = grape.Grape(data_dir + 'wwv_jul_2021/2021-07-01T000000Z_N0000020_G1_FN20vr_FRQ_WWV10.csv', filt=True)
# g.distPlot('f', 'test')
g.bestFitsPlot('f', 'test')


# grape.movie('monthlyMedTrends2022', 'medTrend2022', 5)
