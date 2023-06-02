from grape import *

gr = Grape('wwv_data.txt', filt=True)
gr.dopPowPlot('butter_filt')
