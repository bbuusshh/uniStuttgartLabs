import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import log
from operator import itemgetter

## Structure [[pressure (mBar), breakdown(V), Error]] ##

R = 22E3
C = 16E-6
tau = R*C
omega = 1/(2+3.14*tau)




data = np.array([[1E-1, 1300, 50],[1.0, 1050, 20],[1.8,1380, 20],[1.3, 1210, 10], [1.4, 1180, 15], [1.6, 1310, 10], [4.5E-1, 745, 10], [7E-1, 871, 10],[1.3E-1, 1100, 10], [1.7E-1, 730, 20], [1.5E-1, 800, 20], [1.7E-1, 705, 20], [1.8E-1, 680, 20], [2.0E-1, 693, 20],[2.3E-1, 690, 20], [2.7E-1, 703, 10], [3.8E-1, 730, 20], [4.8E-1, 780, 10], [5.9E-1, 830, 10]])

data = sorted(data, key=itemgetter(0))


#Split data into arrays
xpltData = []
ypltData = []
error = []

for i in range(len(data)):
    xpltData = np.append(xpltData, 0.2*data[i][0]*10000)
    ypltData = np.append(ypltData, data[i][1])
    error = np.append(error, data[i][2])



## Curve Fitting ##
def func(x, A, B, g):
    return ((B*x)/(np.log((A*x)/(np.log(1) + (1/g)))))


popt, pcov = curve_fit(func, xpltData[2:11], ypltData[2:11], method = 'trf')

dp = np.sqrt(np.diag(pcov))
minimum = min(ypltData)
print(popt)
print(pcov)

for i in range(len(data)):
    if ypltData[i] == minimum:
        BreakVolt = xpltData[i]


## Plotting Results
#plt.scatter(xpltData, ypltData)
plt.plot(xpltData, func(xpltData, *popt), label = 'fit: $C_{1}$ = %5.3f, $C_{2}$ = %5.3f,$\gamma$ = %5.3f' % tuple(popt) + '\n' + '$\delta$P = ' + str(dp[1]))
plt.errorbar(xpltData, ypltData, yerr = error, fmt = 'o', label = 'Experimental Results' + '\n' + 'Minimum = ' + '(' + str(BreakVolt) + ',' +  str(minimum) + ')')
plt.xlabel(" Pressure (Pa*cm)")
plt.ylabel("Breakdown Voltage (V)")
plt.legend()
plt.show()














