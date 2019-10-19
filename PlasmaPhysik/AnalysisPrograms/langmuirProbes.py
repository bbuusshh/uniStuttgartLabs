# Program to analyze experiments with double and single langmuir probes
# Written by Samuel Tovey at the University of Stuttgart
# Please address any questions to tovey.samuel@gmail.com
# run as python langmuirprobes with hardcoded filepaths, or, include cmd user interface


import numpy as np
import matplotlib.pyplot as plt
import math as m
import csv
import glob
from scipy.optimize import curve_fit
from operator import itemgetter
from scipy import stats
from uncertainties import unumpy as unp
from uncertainties import ufloat
from uncertainties import umath



singleProbeResults = glob.glob('/tikhome/stovey/Documents/Labs/PlasmaPhysik/A3/single/*.txt')
doubleProbeResults = glob.glob('/tikhome/stovey/Documents/Labs/PlasmaPhysik/A3/double/*txt')

#singleProbeResults = glob.glob('/home/dirac/Documents/Labs/PlasmaPhysik/A3/single/*.txt')
#doubleProbeResults = glob.glob('/home/dirac/Documents/Labs/PlasmaPhysik/A3/double/*txt')


pressuresSingle = np.array([1.0, 0.1, 0.1, 0.16, 2.1, 0.25, 0.25, 0.36, 0.5, 0.75])*1000
pressuresDouble = np.array([1.0, 0.1, 0.16, 2.1, 0.25, 0.37, 0.5, 0.75])*1000
pressuresSingleUnc = []
pressuresDoubleUnc = []
for i in range(len(pressuresSingle)):
    pressuresSingleUnc = np.append(pressuresSingleUnc, ufloat(pressuresSingle[i], 20))
for i in range(len(pressuresDouble)):
    pressuresDoubleUnc = np.append(pressuresDoubleUnc, ufloat(pressuresDouble[i], 20))
#### Define physical constants ####
e = 1.6E-19
e0 = 8.85E-12
me = 9.11E-31
k = 1.38E-5 
S = 1.3E-6
Tg = 0.025

        ###########################################################
        ###########################################################
        #### Calculations to be performed upond data analysis  ####
        ###########################################################
        ###########################################################



def calculations(Te, ne, Pg):

    alpha = -(ne*Tg*e)/Pg
    debyeLength = umath.sqrt((e0*Te*e*k)/((e**2)*ne))
    plasmaFrequency = umath.sqrt(((e**2)*ne)/(e0*me))

    return alpha, debyeLength, plasmaFrequency



        ###########################################################
        ###########################################################
        #### Function to analyse single langmuir probe results ####
        ###########################################################
        ###########################################################



def singleProbe(index, Pg):

    ###############################################################
    #### Load data from the input file and seperate into arrays####
    ###############################################################

    dataArray = np.loadtxt(singleProbeResults[index], unpack=False)
    data = sorted(dataArray, key=itemgetter(0))

    voltage = np.array([])
    current = np.array([])
    time = []
    voltageUnc = np.array([])

    for i in range(len(data)):
        voltage = np.append(voltage, data[i][1])
        voltageUnc = np.append(voltageUnc, ufloat(data[i][1], 2))
        current = np.append(current, data[i][2])
        steps = np.append(time, data[i][0])

    ###############################
    ### Adjust data for fitting ###
    ##############################
    currentUnc = []
    Isat = max(abs(current))
    currentpre = np.log(current + Isat)
    for i in range(len(current)):
        currentUnc = np.append(currentUnc, ufloat(currentpre[i], abs((1.45E-6)/(current[i]))))
    current = currentpre
    #######################################################################################################
    #### Extract the data in order to perform the fit in the desired region, namely the linear portion ####
    #######################################################################################################

    fitCurrent = []
    fitVoltage = []
    for i in range(len(current)):
        if current[i] > -7.1 and voltage[i] > -60:
            fitCurrent = np.append(fitCurrent, current[i])
            fitVoltage = np.append(fitVoltage, voltage[i])
    #######################################################
    #### Search for outliers and remove as contingency ####
    #######################################################

    z = np.abs(stats.zscore(fitCurrent))
    for i in range(len(z)):
        if z[i] > 3:
            del fitCurrent[i]
            del fitVoltage[i]

    #######################
    #### Curve Fitting ####
    #######################
    def func(x, m, c):
        return m*x + c

    popt, pcov = curve_fit(func, fitVoltage, fitCurrent)

    ##########################################
    #### Perform Calculations on data set ####
    ##########################################

    Te = ufloat(1/popt[0], np.sqrt(np.diag(pcov))[1])
    phi = Te*(ufloat(Isat,1.45E-6)  - ufloat(popt[1], 0))
    ne = (ufloat(Isat, 1.45E-6)/(e*S*0.61))*umath.sqrt((6.26E-26)/((Te*e)/k))
    alpha, debye, plasma = calculations(Te, ne, Pg)

    #Plotting the function to observe individual results with titles
    plt.grid()
    #plt.ylim(-0.0001, 0.001)
    plt.plot(voltage, func(voltage, *popt))
    plt.errorbar(voltage, current, yerr=unp.std_devs(currentUnc), xerr=unp.std_devs(voltageUnc),fmt = 'rx')
    #plt.title('Data Set ' + str(index))
    plt.xlabel("Voltage (V)")
    plt.ylabel("ln(Current (mA))")
    plt. show()


    return Te, phi, ne, alpha, debye, plasma



        ###############################################################
        ###############################################################
        #### Function to Analyse the Double Langmuir Probe results ####
        ###############################################################
        ###############################################################



def doubleProbe(index, Pg):

    ###############################################################
    #### Load data from the input file and seperate into arrays####
    ###############################################################

    dataArray = np.loadtxt(doubleProbeResults[index], unpack=False)
    data = sorted(dataArray, key=itemgetter(0))

    voltage = []
    current = []
    time = []
    currentUnc = []
    voltageUnc = []
    for i in range(len(data)):
        voltage = np.append(voltage, data[i][1])
        current = np.append(current, data[i][2])
        steps = np.append(time, data[i][0])
        voltageUnc = np.append(voltageUnc, ufloat(data[i][1], 2))
        currentUnc = np.append(currentUnc, ufloat(data[i][2], 1.45E-6))
    current1 = []
    voltage1 = []
    current2 = []
    voltage2 = []
    current3 = []
    voltage3 = []

    ###################################################
    #### Seperate data into three distinct regions ####
    ###################################################

    for i in range(len(current)):
        if current[i] < -0.00002:
            current1 = np.append(current1, current[i])
            voltage1 = np.append(voltage1, voltage[i])
        elif current[i] > -0.00001  and current[i] < 0.000015 and voltage[i] > -30 and voltage[i] < 30:
            current2 = np.append(current2, current[i])
            voltage2 = np.append(voltage2, voltage[i])
        elif current[i] > 0.00002 and voltage[i] > 0:
            current3 = np.append(current3, current[i])
            voltage3 = np.append(voltage3, voltage[i])

    #######################
    #### Curve Fitting ####
    #######################
    def func(x, m, c):
        return m*x + c

    popt1, pcov1 = curve_fit(func, voltage1, current1)
    popt2, pcov2 = curve_fit(func, voltage2, current2)
    popt3, pcov3 = curve_fit(func, voltage3, current3)
    
    I1sat = ufloat(abs(popt1[1]), np.sqrt(np.diag(pcov1))[1])
    I2sat = ufloat(popt3[1], np.sqrt(np.diag(pcov2))[1])
    IvU = ufloat(popt2[0], np.sqrt(np.diag(pcov3))[1])
    Te = (I1sat*I2sat)/(IvU*(I1sat + I2sat))
    ne = (I1sat/(e*S*0.61))*umath.sqrt((6.26E-26)/((Te*e)/k))

    alpha, debye, plasma = calculations(abs(Te), abs(ne), Pg)





    #######################################################################
    ####Plotting the function to observe individual results with titles####
    #######################################################################

    plt.grid()
    plt.ylim(-0.00006, 0.00006)
    plt.errorbar(voltage, current, yerr=unp.std_devs(currentUnc), xerr=unp.std_devs(voltageUnc), fmt='rx')
    plt.plot(voltage, func(voltage, *popt1))
    plt.plot(voltage, func(voltage, *popt2))
    plt.plot(voltage, func(voltage, *popt3))
    #plt.title('Data Set ' + str(index))
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (mA")
    plt. show()


    return Te, ne, alpha, debye, plasma






                                                                    ############################################################
                                                                    #### '''''''''''''''''''''''''''''''''''''''''''''''''' ####
#################################################################** #### Run programs for all files and plot for comparison #### **################################################################################
                                                                    #### '''''''''''''''''''''''''''''''''''''''''''''''''' ####
                                                                    ############################################################



singleProbeTe = np.array([])
singleProbePhi = np.array([])
alphaMat = np.array([])
debyeMat = np.array([])
plasmaMat = np.array([])
neMat = np.array([])
for i in range(len(singleProbeResults)):
    Te, phi, ne, alpha, debye, plasma = singleProbe(i, pressuresSingle[i])
    singleProbeTe = np.append(singleProbeTe, Te)
    singleProbePhi = np.append(singleProbePhi, phi)
    alphaMat = np.append(alphaMat, alpha)
    debyeMat = np.append(debyeMat, debye)
    plasmaMat = np.append(plasmaMat, plasma)
    neMat = np.append(neMat, ne)

#print(singleProbeTe)
#print(alphaMat)
#print(debyeMat)
#print(plasmaMat)
#print(neMat)

doubleProbeTe = np.array([])
alphaMat2 = np.array([])
debyeMat2 = np.array([])
plasmaMat2 = np.array([])
neMat2 = np.array([])
for i in range(len(doubleProbeResults)):
    Te, ne, alpha, debye, plasma = doubleProbe(i, pressuresDouble[i])
    doubleProbeTe = np.append(doubleProbeTe, Te)
    alphaMat2 = np.append(alphaMat2, alpha)
    debyeMat2 = np.append(debyeMat2, debye)
    plasmaMat2 = np.append(plasmaMat2, plasma)
    neMat2 = np.append(neMat2, ne)





plt.subplot(1, 2, 1)
plt.grid()
plt.errorbar(pressuresSingle, unp.nominal_values(singleProbeTe),yerr=unp.std_devs(singleProbeTe*100), xerr=unp.std_devs(pressuresSingleUnc), fmt='rx')
plt.xlabel('Pressure[Pa]')
plt.ylabel("$T_{e}$[eV]")
plt.subplot(1, 2, 2)
plt.errorbar(pressuresDouble, unp.nominal_values(doubleProbeTe), yerr=unp.std_devs(doubleProbeTe), xerr=unp.std_devs(pressuresDoubleUnc), fmt='rx')
plt.grid()
plt.xlabel('Pressure[Pa]')
plt.ylabel("$T_{e}$[eV]")
#plt.show()

plt.subplot(1, 2, 1)
plt.errorbar(pressuresSingle, unp.nominal_values(neMat), yerr=unp.std_devs(neMat*10), xerr=unp.std_devs(pressuresSingleUnc), fmt='rx')
plt.grid()
plt.xlabel('Pressure[Pa]')
plt.ylabel("$n_{e}[m^{-3}]$")
plt.subplot(1, 2, 2)
plt.errorbar(pressuresDouble, unp.nominal_values(neMat2), yerr=unp.std_devs(neMat2), xerr=unp.std_devs(pressuresDoubleUnc), fmt='rx')
plt.grid()
plt.xlabel('Pressure[Pa]')
plt.ylabel("$n_{e}[m^{-3}]$")
#plt.show()

plt.subplot(1, 2, 1)
plt.grid()
plt.ylim(-1.0E-7, 0.25E-7)
plt.errorbar(unp.nominal_values(singleProbeTe), unp.nominal_values(alphaMat), yerr=unp.std_devs(alphaMat*100), xerr=unp.std_devs(singleProbeTe*100), fmt='rx')
plt.xlabel('Temperature[eV]')
plt.ylabel('$\\alpha$')
plt.subplot(1, 2, 2)
plt.grid()
plt.ylim(-1.E-8, 0.25E-8)
plt.errorbar(unp.nominal_values(doubleProbeTe), unp.nominal_values(alphaMat2), yerr=unp.std_devs(alphaMat2*10), xerr=unp.std_devs(doubleProbeTe), fmt='rx')
plt.ylabel('$\\alpha$')
plt.xlabel('Temperature[eV]')
#plt.show()

plt.subplot(1, 2, 1)
plt.grid()
plt.ylim(0, 1E-5)
plt.errorbar(pressuresSingle, unp.nominal_values(debyeMat), yerr=unp.std_devs(debyeMat*100), xerr=unp.std_devs(pressuresSingleUnc), fmt='rx')
plt.xlabel('Pressure[Pa]')
plt.ylabel("$\lambda_{D}[m]$")
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0.00001))
plt.subplot(1, 2, 2)
plt.ylim(0, 1E-5)
plt.errorbar(pressuresDouble, unp.nominal_values(debyeMat2), yerr=unp.std_devs(debyeMat2), xerr=unp.std_devs(pressuresDoubleUnc), fmt='rx')
plt.grid()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0.00001))
plt.xlabel('Pressure[Pa]')
plt.ylabel("$\lambda_{D}[m]$")
#plt.show()

plt.subplot(1, 2, 1)
plt.grid()
plt.errorbar(pressuresSingle, unp.nominal_values(plasmaMat), yerr=unp.std_devs(plasmaMat*10), xerr=unp.std_devs(pressuresSingleUnc), fmt='rx')
plt.xlabel('Pressure[Pa]')
plt.ylabel("$\omega_{p}[s^{-1}]$")
plt.subplot(1, 2, 2)
plt.errorbar(pressuresDouble, unp.nominal_values(plasmaMat2), yerr=unp.std_devs(plasmaMat2), xerr=unp.std_devs(pressuresDoubleUnc), fmt='rx')
plt.grid()
plt.xlabel('Pressure[Pa]')
plt.ylabel("$\omega_{p}[s^{-1}]$")
#plt.show()














