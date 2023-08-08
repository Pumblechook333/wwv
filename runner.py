import grape

months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
data_dir = 'wwv_data/'

# gh = grape.GrapeHandler([(data_dir + 'wwv_' + month + '_2022') for month in months])

# for m in months:
#     gh = grape.GrapeHandler([data_dir + 'wwv_' + m + '_2022'])
#     gh.medTrend(m + 'MedTrend' + '2022')


g = grape.Grape(data_dir + 'wwv_jul_2021/2021-07-01T000000Z_N0000020_G1_FN20vr_FRQ_WWV10.csv', filt=True)
g.distPlot('f', 'test')


# grape.movie('monthlyMedTrends2022', 'medTrend2022', 5)
