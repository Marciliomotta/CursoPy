import numpy as np
import cv2

imagem = cv2.imread("027R_3.png")

cinzaclaro= np.array([100,67,0], dtype="uint8")
cinzaescuro= np.array([255,128,50], dtype="uint8")

imagemgray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

cv2.imshow("imagemgray", imagemgray)

iris = cv2.inRange(imagemgray, cinzaescuro, cinzaclaro)

cv2.imshow("iris", iris)
cv2.waitKey(0)