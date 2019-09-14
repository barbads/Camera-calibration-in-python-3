import sys
import numpy as np
import cv2
from camera_calibration import calibration
from camera_calibration import correct_distortion
from math import sqrt
import pandas as pd

coord1 = (0,0)
coord2 = (0,0)
firstclick = 0

def Draw_line(event,x,y,flags,param):
    global contador
    global nova_imagem
    global pixel_inicial
    global pixel_final
    global aux_x
    global aux_y
    if event == cv2.EVENT_LBUTTONDOWN:
        if contador == 0:
            aux_x = x
            aux_y = y
            contador = 1
        elif contador == 1:
            nova_imagem = 1;
            pixel_inicial = (aux_x, aux_y)
            pixel_final = (x,y)
            contador = 0
            dist = sqrt((x-aux_x)**2 + (y-aux_y)**2)
            print(dist)



if(str(sys.argv[1])) == '-r1':
    contador = 0
    nova_imagem = 0
    video = cv2.VideoCapture(0)
    flag, frame = video.read()

    if (not (flag)):
        exit("Error while reading the video, try again.")

    cv2.namedWindow('Webcam')
    cv2.setMouseCallback('Webcam', Draw_line)

    while(True):
        flag, frame = video.read()

        if (not (flag)):
            exit("Error while reading the video, try again.")

        cv2.imshow('Webcam', frame)

        if(nova_imagem):
            resultado = cv2.line(frame, pixel_inicial, pixel_final, (0,255,0), 2, 8)
            cv2.namedWindow('Resultado')
            cv2.imshow('Resultado', resultado)

        if cv2.waitKey(25) == 27:
            cv2.destroyWindow('Webcam')
            cv2.destroyWindow('Resultado')
            break
    video.release()

if (str(sys.argv[1]) == '-r2'):
    WebCam = cv2.VideoCapture(0)
    PI_11 = np.empty(0)
    PI_13 = np.empty(0)
    PI_22 = np.empty(0)
    PI_23 = np.empty(0)
    PD_1 = np.empty(0)
    PD_2 = np.empty(0)
    PD_3 = np.empty(0)
    PD_4 = np.empty(0)
    PD_5 = np.empty(0)

    # contador para o numero de imagens onde o xadrez foi detectado
    detected_images = 0

    #numero de imagens que queremos detectar o tabuleiro de xadrez para
    #calcular os parametros intrinsecos da camera
    max_images = 5

    #Numero de bordas (com 4 quadrados) na vertical e na horizontal do tabuleiro
    board_w = 8
    board_h = 6

    #tamanho em mm do quadrado
    tam_quad = 29

    #determina o tempo (s) de espera para mudar o tabuleiro de posicao apos uma deteccao
    time_step = 2

    for i in range(5):
        mtx, dist = calibration(WebCam, tam_quad, board_h, board_w, time_step, max_images)
        PI_11 = np.append(PI_11, mtx[0][0])
        PI_13 = np.append(PI_13, mtx[0][2])
        PI_22 = np.append(PI_22, mtx[1][1])
        PI_23 = np.append(PI_23, mtx[1][2])
        PD_1 = np.append(PD_1, dist[0][0])
        PD_2 = np.append(PD_2, dist[0][1])
        PD_3 = np.append(PD_3, dist[0][2])
        PD_4 = np.append(PD_4, dist[0][3])
        PD_5 = np.append(PD_5, dist[0][4])

print('Media :')
print('%.2f'%PI_11.mean())
print('%.2f'%PI_13.mean())
print('%.2f'%PI_22.mean())
print('%.2f'%PI_23.mean())
print('%.2f'%PD_1.mean())
print('%.2f'%PD_2.mean())
print('%.2f'%PD_3.mean())
print('%.2f'%PD_4.mean())
print('%.2f'%PD_5.mean())
print()

print('Desvio Padrao :')
print('%.2f'%PI_11.std())
print('%.2f'%PI_13.std())
print('%.2f'%PI_22.std())
print('%.2f'%PI_23.std())
print('%.2f'%PD_1.std())
print('%.2f'%PD_2.std())
print('%.2f'%PD_3.std())
print('%.2f'%PD_4.std())
print('%.2f'%PD_5.std())

cv2.destroyAllWindows()
