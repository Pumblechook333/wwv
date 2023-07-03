import grape

# gh = grape.GrapeHandler('wwv_2022_july_data', filt=True)
# gh = grape.GrapeHandler('wwv_2022_december_data', filt=True)
# gh.mgBestFitsPlot('f', 'december_2022_daily_bestfit_overplot', 'december_fit_plot', ylim=[- 1.5, 2])
# gh.mgBestFitsPlot('f', 'july_2022_daily_bestfit_overplot', 'july_fit_plot')

# g = grape.Grape('wwv_2022_july_data/2022-07-01T000000Z_N0000020_G1_FN20vr_FRQ_WWV10.csv', filt=True)
g = grape.Grape('wwv_july_data/2021-07-01T000000Z_N0000020_G1_FN20vr_FRQ_WWV10.csv', filt=True)
# g.distPlotFit('f', 'testTotDistNewFit')
# g.distPlotsFit('f', sel=[18, 0])
g.distPlots('f', sel=[18, 0])
# g.bestFitsPlot('f', 'testDopFitPlot', ylim=[-1.5, 2])
# g.dopPowPlot('testDopPlot', ylim=[-1.5, 2])
# g.dopPowPlot('testDopPlot')

# grape.movie('july_1_dshift_distplots', 'jul1_dist_movie', 10)
