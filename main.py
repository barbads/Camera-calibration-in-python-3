import sys
import numpy as np
import cv2
from camera_calibration import calibration
from camera_calibration import correct_distortion
from math import sqrt
import pandas as pd

# Inicializacao de variaveis globais para o primeiro requisito
pixel_inicial = 0
pixel_final = 0
aux_x = 0
aux_y = 0
contador = 0
contador = 0
nova_imagem = 0
# Variaveis globais para os demais requisitos
IntParam = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype = float)
DistParam = np.array([0, 0, 0, 0, 0], dtype = float)

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


def main():

    if(str(sys.argv[1])) == '-r1':
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

            if cv2.waitKey(0) == 27:
                cv2.destroyAllWindows()
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

        IntParam[0][0] = PI_11.mean()
        IntParam[0][2] = PI_13.mean()
        IntParam[1][1] = PI_22.mean()
        IntParam[1][2] = PI_23.mean()

        DistParam[0] = PD_1.mean()
        DistParam[1] = PD_2.mean()
        DistParam[2] = PD_3.mean()
        DistParam[3] = PD_4.mean()
        DistParam[4] = PD_5.mean()

        strvalue = ""
        for i in range (3):
            for j in range (3):
                strvalue += (str(IntParam[i][j])) 
                strvalue += ("\n")

        strvalue += ("\n")

        for i in range(5):
            strvalue += str(DistParam[i])
            strvalue += ("\n")


        file = open("parametros.txt", "w")
        file.writelines(strvalue)

        correct_distortion(WebCam, IntParam, DistParam)
        
        print('Media :')
        print(IntParam)
        print(DistParam)

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

    if (str(sys.argv[1]) == '-r3'):
        print("eae")


if __name__ == '__main__':
    main()