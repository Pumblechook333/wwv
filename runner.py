import grape

months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
data_dir = 'wwv_data/'

gh = grape.GrapeHandler([(data_dir + 'wwv_' + month + '_2022') for month in months])
gh.medTrend('medTrend2022')

# for m in months:
#     gh = grape.GrapeHandler([data_dir + 'wwv_' + m + '_2022'])
#     gh.medTrend(m + 'MedTrend' + '2022')

# gh = grape.GrapeHandler([(data_dir + 'wwv_' + month + '_2022') for month in months], comb=False)
# gh = grape.GrapeHandler([data_dir + 'wwv_' + 'mar' + '_2022'], comb=False)

# gh.mgBestFitsPlot('f', 'october_2022_daily_bestfit_overplot', 'october_2022_fit_plot')
# gh.yearMedTrend('medTrendPlot_year')

# g = grape.Grape(data_dir + 'wwv_jul_2021/2021-07-01T000000Z_N0000020_G1_FN20vr_FRQ_WWV10.csv', filt=True)
# g.distPlotFit('f', 'distplotFitTest')
# g.distPlotsFit('f')
# g.distPlot('f', 'distplotTest')
# g.distPlots('f', figname='hr18bin0_jul1_2021', sel=[18, 0])
# g.bestFitsPlot('f', 'testDopFitPlot_new')
# g.dopPowPlot('testDopPlot', ylim=[-1, 1])
# g.dopPowPlot('testDopPlot')

# grape.movie('monthlyMedTrends2022', 'medTrend2022', 5)
