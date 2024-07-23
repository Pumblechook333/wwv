import grape

# g = grape.GrapeHandler(['tot_eclipse_24'], filt=True, tShift=False)
# grape.pickle_grape(g, filename='tot_ecl_24_test')

g = grape.unpickle_grape('tot_ecl_24_test.pkl')

tl = ['', 'Apr 1', 'Apr 2', 'Apr 3', 'Apr 4', 'Apr 5', 'Apr 6', 'Apr 7', 'Apr 8']
g.dopPlotOver(figname='tot_ecl_24_edit', tl=tl, ylim=[-1.3, 1.5])
