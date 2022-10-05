import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('041R_1.png',0)
sobel = cv.Sobel(img, -1, 0, 1)

fig, ax = plt.subplots(ncols=2,figsize=(15,5))
ax[0].imshow(img, cmap = 'gray')
ax[0].set_title('Original')
ax[0].axis('off')
ax[1].imshow(sobel, cmap = 'gray')
ax[1].set_title('Sobel Aplicado')
ax[1].axis('off')
plt.show()