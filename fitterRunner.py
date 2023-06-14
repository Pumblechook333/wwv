import grape

gh = grape.GrapeHandler('wwv_july_data', filt=True)

gh.mgBestFitsPlot('f', 'july_fit_plots', 'july_fit_plot')
