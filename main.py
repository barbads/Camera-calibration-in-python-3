import sys
import numpy as np
import cv2
from camera_calibration import calibration
from camera_calibration import correct_distortion
from math import sqrt

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
    cam = cv2.VideoCapture(0)
    mtx, dist = calibration(cam, 28, 8, 6, 5, 5)
    correct_distortion(cam, mtx, dist)
