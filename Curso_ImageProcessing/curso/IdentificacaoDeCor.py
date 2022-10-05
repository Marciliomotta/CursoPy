import numpy as np
import cv2

'''Pessoal as duas variaveis estao alteradas para valores em uma escala de cinza proximo ao branco e nao para o azul para o azul 
alterar com os valores que contem no vídeo '''

azulEscuro = np.array([100, 100, 100], dtype = "uint8")
azulClaro = np.array([255, 255, 255], dtype = "uint8")

img = cv2.imread('044R_3.png',0)

obj = cv2.inRange(img, azulEscuro, azulClaro)

(cnts, _) = cv2.findContours(obj.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
if len(cnts) > 0:
    cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
    rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
    cv2.drawContours(img, [rect], -1, (0, 255, 255),2)
cv2.imshow("Tracking", img)
cv2.imshow("Binary", obj)

cv2.waitKey(0)