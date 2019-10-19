import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from scipy.optimize import curve_fit

## Structure [[pressure (mBar), breakdown(V), Error]]
data = np.array([[6.2E-1, 632, 10], [1.0, 711, 10], [3.0E-1, 508, 10], [2.2E-1, 498, 10], [6.3E-1, 597, 10], [8.0E-1, 665, 10], [9.2E-2, 587, 10], [7.6E-2, 718, 10], [3.7E-1,502, 10], [1.8E-1, 481, 10], [2.7E-1, 490, 10], [3.9E-1, 533, 10], [5E-1, 545, 10], [1.4E-1, 486, 10], [1.6E-1, 475, 10], [1.1E-1, 525, 10], [1.2E-1, 494, 10]])

## Unused Data: [1.2E-1, 535, 10], [5.0E-1, 555, 10], [1E-1, 586, 10]

data = sorted(data, key=itemgetter(0))

print(data)


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

popt, pcov = curve_fit(func, xpltData[2:15], ypltData[2:15], method = 'trf')


dp = np.sqrt(np.diag(pcov))
minimum = min(ypltData)

print(popt)
print(pcov)

for i in range(len(data)):
    if ypltData[i] == minimum:
        BreakVolt = xpltData[i]

## Plotting Results
#plt.scatter(xpltData, ypltData)
plt.plot(xpltData, func(xpltData, *popt), label='fit: $C_{1}$=%5.3f, $C_{2}$=%5.3f, $\gamma$=%5.3f' % tuple(popt) + '\n' + '$ \delta P$ = ' + str(dp[1]))
plt.errorbar(xpltData, ypltData, yerr = error, fmt = 'o', label = 'Experimental Data' + '\n' + 'Minimum = ' +  '(' + str(BreakVolt) + ',' +  str(minimum) + ')')
plt.xlabel(" Pressure (Pa*cm)")
plt.ylabel("Breakdown Voltage (V)")
plt.legend()
plt.show()














