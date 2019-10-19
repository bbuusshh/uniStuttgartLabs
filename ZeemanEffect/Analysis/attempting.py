import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
import scipy.ndimage
from scipy import interpolate
from scipy import signal



#imagesFolder = raw_input("Input the filepath to the images: ")
imagesFolder = '/tikhome/stovey/Documents/ZeemanLab/Part1'
images = [imagesFolder+'/'+str(i) for i in os.listdir(imagesFolder)]
imgArray = [cv2.imread(images[i], 0) for i in range(len(images))]



def main(img):
    xL = np.arange(img.shape[1])
    yL = np.arange(img.shape[0])
    x0, y0 = 1670, 930 
    x1, y1 = 1300, 2015

    num_points = 1300

    x = np.linspace(x0,x1, num_points)
    y = np.linspace(y0,y1, num_points)
    zi_slice = img[y.astype(np.int), x.astype(np.int)]

#    fig, axes = plt.subplots(nrows=2, figsize=(15,15))
#    axes[0].imshow(img)
#    axes[0].plot((x0, x1), (y0, y1), 'r.-')
#    axes[0].axis('image')
    data = signal.savgol_filter(zi_slice, 13, 1)
    dists = np.sqrt(x**2 + y**2)
#    axes[1].plot(x, data)
#    plt.show()


###### Analysis of Image ######


    testing = []
    k = 0
    i = 0
    while i < (len(data)-1):
        if data[i] > 200:
            testing.append([])
            j = i
            while data[j] > 200 and j < 1299:
                testing[k].append(j)
                j += 1
            k += 1
            i = j
        i += 1

    radii = [0 for i in range(len(testing))]
    for i in range(len(testing)):
        radii[i] = np.mean(testing[i])
    return radii


###### Running and plotting ######


testing = []
for i in range(len(images)):
    testing.append(main(imgArray[i]))

numRings = 5
peakArray = [[0 for j in range(len(testing))]for i in range(numRings)]



for i in range(numRings):
    for j in range(len(testing)):
        peakArray[i][j] = testing[j][i]


plotArray = [peakArray[i][10:] for i in range(len(peakArray))]
plotArray = [[(2*j*(7.81))**2 for j in i] for i in plotArray]




x = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


fitArray = [np.poly1d(np.polyfit(x, plotArray[i], 1))(x) for i in range(len(plotArray))]
gradientArray = [np.polyfit(x, plotArray[i], 1)[0] for i in range(len(plotArray))]


def BohrCalc(D1, D2):
    L = 9.462E-3
    f = 300E-3
    h = 6.626E-34
    c = 2.98E8
    
    mu = (4*h*c*(f**2))/(L*(D2 - D1))
    return mu
    
bohrArray = [BohrCalc(gradientArray[i], gradientArray[i+1]) for i in range(len(gradientArray)-1)]
#bohrArray = [[100*j for j in i] for i in bohrArray]
print(bohrArray)

for i in range(len(plotArray)):
    plt.scatter(x, plotArray[i])
    plt.plot(x, fitArray[i])
    plt.title('Diameter Squared vs Magnetic Field')
    plt.xlabel('Magnetic Field (T)')
    plt.ylabel('Diameter Squared (m^2)')
plt.show()


















