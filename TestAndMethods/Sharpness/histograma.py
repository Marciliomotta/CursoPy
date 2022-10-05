import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from PIL import Image
import math

image = data.coffee()
plt.imshow(image, cmap='gray')

def convert_to_gray(image, luma=False):
    if luma:
        params = [0.299, 0.589, 0.114]
    else:
        params = [0.2125, 0.7154, 0.0721]
    gray_image = np.ceil(np.dot(image[..., :3], params))

    # Saturando os valores em 255
    gray_image[gray_image > 255] = 255

    return gray_image

def instantiate_histogram():
    hist_array = []

    for i in range(0, 256):
        hist_array.append(str(i))
        hist_array.append(0)

    hist_dct = {hist_array[i]: hist_array[i + 1] for i in range(0, len(hist_array), 2)}

    return hist_dct

histogram = instantiate_histogram()

def count_intensity_values(hist, img):
    for row in img:
        for column in row:
            hist[str(int(column))] = hist[str(int(column))] + 1

    return hist

histogram = count_intensity_values(histogram, image)

def plot_hist(hist, hist2=''):
    if hist2 != '':
        figure, axarr = plt.subplots(1,2, figsize=(20, 10))
        axarr[0].bar(hist.keys(), hist.values())
        axarr[1].bar(hist2.keys(), hist2.values())
    else:
        plt.bar(hist.keys(), hist.values())
        plt.xlabel("Níveis intensidade")
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        plt.grid(True)
        plt.show()