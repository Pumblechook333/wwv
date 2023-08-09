import grape

months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
mIDs = {'jan': '1', 'feb': '2', 'mar': '3', 'apr': '4', 'may': '5', 'jun': '6',
        'jul': '7', 'aug': '8', 'sep': '9', 'oct': '10', 'nov': '11', 'dec': '12'}
data_dir = 'wwv_data/'
dest_dir = 'med_trends/monthly_med_trends_2021/'
beacon = 'wwv'
year = '2021'

# gh = grape.GrapeHandler([(data_dir + 'wwv_' + month + '_2022') for month in months])

for m in months[9::]:
    gh = grape.GrapeHandler([data_dir + beacon + '_' + m + '_' + year])
    if gh.valid:
        gh.medTrend(dest_dir + mIDs[m] + 'MedTrend' + year)

# g = grape.Grape(data_dir + 'wwv_jul_2021/2021-07-01T000000Z_N0000020_G1_FN20vr_FRQ_WWV10.csv', filt=True)
# g.distPlot('f', 'test')


# grape.movie('monthlyMedTrends2022', 'medTrend2022', 5)
