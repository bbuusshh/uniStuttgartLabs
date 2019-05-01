import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

def sin(t, b, decay, a0, g0):
    return a0*(np.sin(t*b))*np.exp(-decay*(t-t[0])) + g0

data = pd.read_csv("rabi_fall2.csv")
plt.annotate(r'$I(t) = I_0sin(\Omega t)exp\left({-\frac{t-t_0}{\tau}}\right)$',
             xy=(0.7, 0.9), size = 7,
             xycoords='figure fraction', bbox=dict(boxstyle='round', fc='white', alpha=1))
plt.annotate(r'$I_0 = $' + str(round(popt[2],2)) + "V\n" + r'$\Omega = $' + str(round(popt[0])) + 'Hz\n' + r'$\tau = $' + str(round(popt[1])) + 's',
             xy=(0.7, 0.7), size = 7,
             xycoords='figure fraction', bbox=dict(boxstyle='round', fc='white', alpha=1))

a = 650
b = 1300
x = np.array(list(data[data.columns[9]][a:b]))
y = np.array(list(data[data.columns[10]][a:b]))
plt.plot(x,y)
popt, pcov = curve_fit(sin, x, y, [1.64793404e+04, 1.07116904e+03 -130, .5, 0.385 + 0.054])
x1 = np.linspace(0.85,0.8610,100)
plt.plot(x, sin(x, popt[0], popt[1], popt[2], popt[3]), 'g--', label='fit-with-bounds')
plt.tight_layout()
