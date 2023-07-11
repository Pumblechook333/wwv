import grape

data_dir = 'wwv_data/'

gh = grape.GrapeHandler(data_dir + 'wwv_october_2022', filt=True)
gh.mgBestFitsPlot('f', 'october_2022_daily_bestfit_overplot', 'october_2022_fit_plot')

# g = grape.Grape(data_dir + 'wwv_july_data/2021-07-01T000000Z_N0000020_G1_FN20vr_FRQ_WWV10.csv', filt=True)
# g.distPlotFit('f', 'testTotDistNewFit')
# g.distPlotsFit('f')
# g.distPlots('f', figname='hr18bin0_jul1_2021', sel=[18, 0])
# g.bestFitsPlot('f', 'testDopFitPlot')
# g.dopPowPlot('testDopPlot', ylim=[-1, 1])
# g.dopPowPlot('testDopPlot')

# grape.movie('dshift_dist_plots', 'test_dist_movie', 10)
