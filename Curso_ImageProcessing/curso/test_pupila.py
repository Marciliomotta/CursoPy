"""
@file: Pupila.py
@author: Marina Pinho Garcia
@disciplina: Introdução ao Processamento de Imagens
@Matrícula: 170110702
@Professor: Bruno Luiggi Macchiavello Espinoza

Arquivo que contém a implementação do Projeto Final
"""

# bibliotecas

import numpy as np
import cv2


# -----------------------------------------------------------------------------------------------------

# Função que dada uma imagem do olho (em níveis de cinza) retorna uma
# imagem com a pupila contonada por uma elipse e seu centro
def DeteccaoPupila(img):
    # ----------------------------- PRE PROCESSAMENTO DA IMAGEM --------------------------

    # faz a binarização da imagem
    ret, binaria = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

    # transforma o para inteiros de 8 bits
    binaria = np.array(binaria, dtype=np.uint8)

    # cria um kernel preenchido de 1 de dimensão 2x2
    kernel = np.ones([2, 2])
    # realiza o fechamento na imagem binarizada utilizando o kernel 2x2
    fechamento = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)

    # transforma o para inteiros de 8 bits
    fechamento = np.array(fechamento, dtype=np.uint8)

    # pega as bordas da imagem
    edges = cv2.Canny(fechamento, 100, 100)

    # ---------------------------- AJUSTE DA ELIPSE-------------------------------

    # separa os contornos da imagem
    _, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # encontra o contorno com maior comprimento (a pupila)
    areas = [cv2.arcLength(c, True) for c in contours]
    max_index = np.argmax(areas)
    # maior dos contornos
    cnt = contours[max_index]

    # transforma a imagem original para RGB
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Encaixa uma elipse no contorno encontrado
    ellipse = cv2.fitEllipse(cnt)
    # Desenha a elipse na imagem
    eli = cv2.ellipse(img.copy(), ellipse, (255, 255, 0), 1)
    # desenha o centro da elipse na imagem
    cv2.circle(eli, (int(ellipse[0][0]), int(ellipse[0][1])), 2, (0, 255, 0), 1)

    return eli


# -----------------------------------------------------------------------------------------------------

# le 5 imagens
img1 = cv2.imread('027R_3.png', 0)
img2 = cv2.imread('028L_1.png', 0)
img3 = cv2.imread('041R_1.png', 0)
img4 = cv2.imread('044R_3.png', 0)


# le 3 imagens - resultado insatisfatório
"""
img1 = cv2.imread('MMU-Iris-Database/36/left/tanwnl2.bmp', 0) 
img2 = cv2.imread('MMU-Iris-Database/6/right/christiner3.bmp', 0)
img3 = cv2.imread('MMU-Iris-Database/19/left/maranl4.bmp', 0)
"""

# -----------------------------------------------------------------------------------------------------

# Realiza a detecção da pupila das 5 imagens
final1 = DeteccaoPupila(img1)
final2 = DeteccaoPupila(img2.copy())
final3 = DeteccaoPupila(img3.copy())
final4 = DeteccaoPupila(img4.copy())

# -----------------------------------------------------------------------------------------------------


# mostra a imagem original 1
cv2.imshow('original', img1)
cv2.waitKey(2000)
cv2.destroyAllWindows()

# mostra a imagem final 1
cv2.imshow('elipse', final1)
cv2.waitKey(2000)
cv2.destroyAllWindows()

# mostra a imagem original 2
cv2.imshow('original', img2)
cv2.waitKey(2000)
cv2.destroyAllWindows()

# mostra a imagem final 2
cv2.imshow('elipse', final2)
cv2.waitKey(2000)
cv2.destroyAllWindows()

# mostra a imagem original 3
cv2.imshow('original', img3)
cv2.waitKey(2000)
cv2.destroyAllWindows()

# mostra a imagem final 3
cv2.imshow('elipse', final3)
cv2.waitKey(2000)
cv2.destroyAllWindows()

# mostra a imagem original 4
cv2.imshow('original', img4)
cv2.waitKey(2000)
cv2.destroyAllWindows()

# mostra a imagem final 4
cv2.imshow('elipse', final4)
cv2.waitKey(2000)
cv2.destroyAllWindows()


# -----------------------------------------------------------------------------------------------------