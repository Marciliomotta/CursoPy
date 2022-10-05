import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('027R_3.png',0)

sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)

sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

laplacian = cv.Laplacian(img, cv.CV_64F)

cv.imshow('sobelx', sobelx)
cv.imshow('sobely', sobely)
#cv.imshow('laplacian', laplacian)

cv.waitKey(0)