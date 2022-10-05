import cv2
import matplotlib.pyplot as plt
img = cv2.imread('desfocada.jpeg')
cimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
lista = []
for i in range(0, cimg.shape[0]):
    for j in range(0, cimg.shape[1]):
        lista.append(cimg[i][j])


lista = sorted(set(lista))

for i in range(0, len(lista)):
    somatoria = 0
    for j in range(0, len(lista)):
        lista=0