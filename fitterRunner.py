import grape
from fitter import Fitter
import pylab as pl

g = grape.Grape('wwv_july_data/2021-07-01T000000Z_N0000020_G1_FN20vr_FRQ_WWV10.csv')
vals = g.getTFPr()
dop = vals[1]

# 5 best distributions for Jul1 dataset based on initial .fit() call
# (reduce time for subsequent running)
best5 = ['dweibull', 'dgamma', 'laplace', 'cauchy', 'foldcauchy']
best3 = ['dweibull', 'dgamma', 'laplace']

# f = Fitter(dop, distributions=bestDist)
f = Fitter(dop, distributions=best3)
f.fit()
summary = f.summary()

print(summary)

f.hist()
# f.plot_pdf(names=bestDist)

pl.xlabel('Doppler Shift, Hz')
pl.ylabel('Normalized Counts')
pl.xlim(-1, 1)  # UTC day

pl.title('Fitted Distribution from ' + g.date + ' Doppler Shift Readings')
pl.savefig('jul1_fitted.png', dpi=250, orientation='landscape')
pl.close()
