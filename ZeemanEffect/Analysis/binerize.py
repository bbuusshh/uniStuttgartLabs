"""Binarize (make it black and white) an image with Python."""

from PIL import Image
from scipy.misc import imsave
import numpy


def binarize_image(img_path, target_path, threshold):
    """Binarize an image."""
    image_file = Image.open(img_path)
    image = image_file.convert('L')  # convert image to monochrome
    image = numpy.array(image)
    image = binarize_array(image, threshold)
    imsave(target_path, image)
    return image

def binarize_array(numpy_array, threshold=200):
    """Binarize a numpy array."""
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > threshold:
                numpy_array[i][j] = 255
            else:
                numpy_array[i][j] = 0
    return numpy_array
i = 40
j = 9
while i <= 45:
    binarize_image('/tikhome/stovey/Documents/ZeemanLab/Part1/CIMG36'+str(i)+'.JPG', '/tikhome/stovey/Documents/ZeemanLab/Part1/'+str(j)+'BW.JPG', 40)
    j += 1
    i += 1
