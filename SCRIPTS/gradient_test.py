import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

x = np.linspace(-1, 1, 1000)
y = np.exp(x)
c = np.tan(x)

plt.scatter(x, y, marker='.', c=c, cmap='seismic')
plt.show()
