import numpy as np
import matplotlib.pyplot as plt
import math as m


coords = [[0.0, 53.6], [14.5, 48.0],[23.3, 37.0],[37.4, 26.1],[6.7, 28.6],[30.3, 8.1],[46.2, 17.3]]

L = 15
a = 5.64

#Data Preperation
fullcoords = [[0 for i in range(3)] for j in range(len(coords))]
angles = [0 for i in range(len(coords))]
for i in range(len(coords)):
        fullcoords[i][0] = (coords[i][0])/10
        fullcoords[i][1] = (coords[i][1])/10
        fullcoords[i][2] = (np.sqrt(coords[i][0]**2 + coords[i][1]**2)*m.tan(0.5*m.atan((np.sqrt(coords[i][0]**2 + coords[i][1]))/L)))/10
        angles[i] = 0.5*m.atan((np.sqrt(coords[i][0]**2 + coords[i][1]))/L)

#Calculate the Miller indices, input a list of lists
millerIndicespre = [[0 for i in range(3)] for j in range(len(coords))]
for i in range(len(coords)):
    millerIndicespre[i][0] = round(fullcoords[i][0])
    millerIndicespre[i][1] = round(fullcoords[i][1])
    millerIndicespre[i][2] = round(fullcoords[i][2])

#print(millerIndicespre)

millerIndices = [[0 for i in range(3)] for j in range(len(coords))]
for i in range(len(coords)):
    n = 1
    cond = False
    while cond == False:
        for j in range(3):
            if millerIndicespre[i][j] == 0.00:
                millerIndicespre[i][j] = 1

        if n*(1/millerIndicespre[i][0]) - int(n*(1/millerIndicespre[i][0])) == 0  and n*(1/millerIndicespre[i][1]) - int(n*(1/millerIndicespre[i][1])) == 0 and n*(1/millerIndicespre[i][2]) - int(n*(1/millerIndicespre[i][2])) == 0:
            cond = True
        else:
            cond = False
            n += 1

    for j in range(3):
        if type(millerIndicespre[i][j]) == int:
            millerIndices[i][j] = 0
        else:
            millerIndices[i][j] = n*(1/millerIndicespre[i][j])


def latticeSpacing(millerIndex, atomicSpacing):
    d =  atomicSpacing*np.sqrt(millerIndex[0]**2 + millerIndex[1]**2 + millerIndex[2]**2)
    return d


def wavelength(spacing, angle):
    l = 2*spacing*m.sin(angle)
    return l


wavelengths = [ 0 for i in range(len(coords))]
spacings = [0 for i in range(len(coords))]
for i in range(len(coords)):
    spacings[i] = latticeSpacing(millerIndices[i], a)
    wavelengths[i] = wavelength(spacings[i], angles[i])*100

print(millerIndices)
print(spacings)
print(wavelengths)
