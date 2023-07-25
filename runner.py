import grape

months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
data_dir = 'wwv_data/'

# gh = grape.GrapeHandler([data_dir + 'wwv_april_2022', data_dir + 'wwv_july_2022',
#                          data_dir + 'wwv_october_2022', data_dir + 'wwv_december_2022'])

# gh = grape.GrapeHandler([(data_dir + 'wwv_' + month + '_2022') for month in months], comb=False)
# gh = grape.GrapeHandler([data_dir + 'wwv_' + 'mar' + '_2022'], comb=False)

# gh.mgBestFitsPlot('f', 'october_2022_daily_bestfit_overplot', 'october_2022_fit_plot')
# gh.medTrend('medTrendPlot_year')

g = grape.Grape(data_dir + 'wwv_jul_2021/2021-07-01T000000Z_N0000020_G1_FN20vr_FRQ_WWV10.csv', filt=True)
# g.distPlotFit('f', 'testTotDistNewFit')
# g.distPlotsFit('f')
# g.distPlots('f', figname='hr18bin0_jul1_2021', sel=[18, 0])
g.bestFitsPlot('f', 'testDopFitPlot_new')
# g.dopPowPlot('testDopPlot', ylim=[-1, 1])
# g.dopPowPlot('testDopPlot')

# grape.movie('dshift_dist_plots', 'test_dist_movie', 10)
