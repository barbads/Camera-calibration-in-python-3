import sys
import numpy as np
import cv2
import time
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
nova_imagem = 0
# Variaveis globais para os demais requisitos
IntParam = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype = float)
DistParam = np.array([0, 0, 0, 0, 0], dtype = float)
ExtParam = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype = float)
TranParam = np.array([0, 0, 0], dtype = float)

DesvPadInt = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype = float)
DesvPadDist = np.array([0, 0, 0, 0, 0], dtype = float)
DesvPadExt = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype = float)


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
    global IntParam, ExtParam, TranParam, DistParam

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
            mtx, dist, R, T = calibration(WebCam, tam_quad, board_h, board_w, time_step, max_images)
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

        correct_distortion(WebCam, IntParam, DistParam, ExtParam, 2)

        print('Media :')
        print(IntParam)
        print(DistParam)

        print('Desvio Padrao :')
        print('%.2f'%PI_11.std(ddof=1))
        print('%.2f'%PI_13.std(ddof=1))
        print('%.2f'%PI_22.std(ddof=1))
        print('%.2f'%PI_23.std(ddof=1))
        print('%.2f'%PD_1.std(ddof=1))
        print('%.2f'%PD_2.std(ddof=1))
        print('%.2f'%PD_3.std(ddof=1))
        print('%.2f'%PD_4.std(ddof=1))
        print('%.2f'%PD_5.std(ddof=1))

        cv2.destroyAllWindows()

    if (str(sys.argv[1]) == '-r3'):
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
        R11 = np.empty(0)
        R12 = np.empty(0)
        R13 = np.empty(0)
        R21 = np.empty(0)
        R22 = np.empty(0)
        R23 = np.empty(0)
        R31 = np.empty(0)
        R32 = np.empty(0)
        R33 = np.empty(0)
        T1 = np.empty(0)
        T2 = np.empty(0)
        T3 = np.empty(0)

        #numero de imagens que queremos detectar o tabuleiro de xadrez para
        #calcular os parametros intrinsecos da camera
        max_images = 3

        #Numero de bordas (com 4 quadrados) na vertical e na horizontal do tabuleiro
        board_w = 8
        board_h = 6

        #tamanho em mm do quadrado
        tam_quad = 29

        #determina o tempo (s) de espera para mudar o tabuleiro de posicao apos uma deteccao
        time_step = 2
        iteracoes = 3

        for i in range(iteracoes):
            mtx, dist, R, T = calibration(WebCam, tam_quad, board_h, board_w, time_step, max_images)

            print("Iteracao ", i+1, " feita", end='')
            if(i != (iteracoes - 1)):
                print(", preparando proxima iteracao\n")

            start_time = time.time()
            rotation_matrix = np.zeros(shape=(3,3))
            cv2.Rodrigues(R[0], rotation_matrix)
            R11 = np.append(R11, rotation_matrix[0][0])
            R12 = np.append(R12, rotation_matrix[0][1])
            R13 = np.append(R13, rotation_matrix[0][2])
            R21 = np.append(R21, rotation_matrix[1][0])
            R22 = np.append(R22, rotation_matrix[1][1])
            R23 = np.append(R23, rotation_matrix[1][2])
            R31 = np.append(R31, rotation_matrix[2][0])
            R32 = np.append(R32, rotation_matrix[2][1])
            R33 = np.append(R33, rotation_matrix[2][2])

            PI_11 = np.append(PI_11, mtx[0][0])
            PI_13 = np.append(PI_13, mtx[0][2])
            PI_22 = np.append(PI_22, mtx[1][1])
            PI_23 = np.append(PI_23, mtx[1][2])

            PD_1 = np.append(PD_1, dist[0][0])
            PD_2 = np.append(PD_2, dist[0][1])
            PD_3 = np.append(PD_3, dist[0][2])
            PD_4 = np.append(PD_4, dist[0][3])
            PD_5 = np.append(PD_5, dist[0][4])

            T1 = np.append(T1, T[0][0])
            T2 = np.append(T2, T[0][1])
            T3 = np.append(T3, T[0][2])

            if(i != (iteracoes - 1)):
                while(time.time() - start_time < 2):
                    pass

        TranParam = T1.mean(), T2.mean(), T3.mean()

        ExtParam[0][0] = R11.mean()
        ExtParam[0][1] = R12.mean()
        ExtParam[0][2] = R13.mean()
        ExtParam[0][3] = TranParam[0]
        ExtParam[1][0] = R21.mean()
        ExtParam[1][1] = R22.mean()
        ExtParam[1][2] = R23.mean()
        ExtParam[1][3] = TranParam[1]
        ExtParam[2][0] = R31.mean()
        ExtParam[2][1] = R32.mean()
        ExtParam[2][2] = R33.mean()
        ExtParam[2][3] = TranParam[2]

        IntParam[0][0] = PI_11.mean()
        IntParam[0][2] = PI_13.mean()
        IntParam[1][1] = PI_22.mean()
        IntParam[1][2] = PI_23.mean()

        DistParam = PD_1.mean(), PD_2.mean(), PD_3.mean(), PD_4.mean(), PD_5.mean()

        DesvPadExt[0][0] = R11.std(ddof=1)
        DesvPadExt[0][1] = R12.std(ddof=1)
        DesvPadExt[0][2] = R13.std(ddof=1)
        DesvPadExt[0][3] = T1.std(ddof=1)
        DesvPadExt[1][0] = R21.std(ddof=1)
        DesvPadExt[1][1] = R22.std(ddof=1)
        DesvPadExt[1][2] = R23.std(ddof=1)
        DesvPadExt[1][3] = T2.std(ddof=1)
        DesvPadExt[2][0] = R31.std(ddof=1)
        DesvPadExt[2][1] = R32.std(ddof=1)
        DesvPadExt[2][2] = R33.std(ddof=1)
        DesvPadExt[2][3] = T3.std(ddof=1)

        DesvPadInt[0][0] = PI_11.std(ddof=1)
        DesvPadInt[0][2] = PI_13.std(ddof=1)
        DesvPadInt[1][1] = PI_22.std(ddof=1)
        DesvPadInt[1][2] = PI_23.std(ddof=1)

        DesvPadDist = PD_1.std(ddof=1), PD_2.std(ddof=1), PD_3.std(ddof=1), PD_4.std(ddof=1), PD_5.std(ddof=1)


        print("Matriz Intrinsecos: ")
        print("Media")
        print(IntParam)
        print("Desvio Padrao")
        print(DesvPadInt)
        print()

        print("Matriz Extrinsecos: ")
        print("Media")
        print(ExtParam)
        print("Desvio Padrao")
        print(DesvPadExt)
        print()

        print("Distorcao: ")
        print("Media")
        print(DistParam)
        print("Desvio Padrao")
        print(DesvPadDist)

        PI_11 = np.empty(0)
        PI_13 = np.empty(0)
        PI_22 = np.empty(0)
        PI_23 = np.empty(0)
        PD_1 = np.empty(0)
        PD_2 = np.empty(0)
        PD_3 = np.empty(0)
        PD_4 = np.empty(0)
        PD_5 = np.empty(0)
        R11 = np.empty(0)
        R12 = np.empty(0)
        R13 = np.empty(0)
        R21 = np.empty(0)
        R22 = np.empty(0)
        R23 = np.empty(0)
        R31 = np.empty(0)
        R32 = np.empty(0)
        R33 = np.empty(0)
        T1 = np.empty(0)
        T2 = np.empty(0)
        T3 = np.empty(0)

        cv2.destroyAllWindows()

    if (str(sys.argv[1]) == '-r4'):
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
        R11 = np.empty(0)
        R12 = np.empty(0)
        R13 = np.empty(0)
        R21 = np.empty(0)
        R22 = np.empty(0)
        R23 = np.empty(0)
        R31 = np.empty(0)
        R32 = np.empty(0)
        R33 = np.empty(0)
        T1 = np.empty(0)
        T2 = np.empty(0)
        T3 = np.empty(0)

        #numero de imagens que queremos detectar o tabuleiro de xadrez para
        #calcular os parametros intrinsecos da camera
        max_images = 5
        iteracoes = 5

        #Numero de bordas (com 4 quadrados) na vertical e na horizontal do tabuleiro
        board_w = 8
        board_h = 6

        #tamanho em mm do quadrado
        tam_quad = 29

        #determina o tempo (s) de espera para mudar o tabuleiro de posicao apos uma deteccao
        time_step = 2

        for i in range(iteracoes):
            mtx, dist, R, T = calibration(WebCam, tam_quad, board_h, board_w, time_step, max_images)

            print("Iteracao ", i+1, " feita", end='')
            if(i != (iteracoes - 1)):
                print(", preparando proxima iteracao\n")

            start_time = time.time()
            rotation_matrix = np.zeros(shape=(3,3))
            cv2.Rodrigues(R[0], rotation_matrix)
            R11 = np.append(R11, rotation_matrix[0][0])
            R12 = np.append(R12, rotation_matrix[0][1])
            R13 = np.append(R13, rotation_matrix[0][2])
            R21 = np.append(R21, rotation_matrix[1][0])
            R22 = np.append(R22, rotation_matrix[1][1])
            R23 = np.append(R23, rotation_matrix[1][2])
            R31 = np.append(R31, rotation_matrix[2][0])
            R32 = np.append(R32, rotation_matrix[2][1])
            R33 = np.append(R33, rotation_matrix[2][2])

            PI_11 = np.append(PI_11, mtx[0][0])
            PI_13 = np.append(PI_13, mtx[0][2])
            PI_22 = np.append(PI_22, mtx[1][1])
            PI_23 = np.append(PI_23, mtx[1][2])

            PD_1 = np.append(PD_1, dist[0][0])
            PD_2 = np.append(PD_2, dist[0][1])
            PD_3 = np.append(PD_3, dist[0][2])
            PD_4 = np.append(PD_4, dist[0][3])
            PD_5 = np.append(PD_5, dist[0][4])

            T1 = np.append(T1, T[0][0])
            T2 = np.append(T2, T[0][1])
            T3 = np.append(T3, T[0][2])

            if(i != (iteracoes - 1)):
                while(time.time() - start_time < 2):
                    pass

        TranParam = T1.mean(), T2.mean(), T3.mean()

        ExtParam[0][0] = R11.mean()
        ExtParam[0][1] = R12.mean()
        ExtParam[0][2] = R13.mean()
        ExtParam[0][3] = TranParam[0]
        ExtParam[1][0] = R21.mean()
        ExtParam[1][1] = R22.mean()
        ExtParam[1][2] = R23.mean()
        ExtParam[1][3] = TranParam[1]
        ExtParam[2][0] = R31.mean()
        ExtParam[2][1] = R32.mean()
        ExtParam[2][2] = R33.mean()
        ExtParam[2][3] = TranParam[2]

        IntParam[0][0] = PI_11.mean()
        IntParam[0][2] = PI_13.mean()
        IntParam[1][1] = PI_22.mean()
        IntParam[1][2] = PI_23.mean()

        DistParam = PD_1.mean(), PD_2.mean(), PD_3.mean(), PD_4.mean(), PD_5.mean()

        DesvPadExt[0][0] = R11.std(ddof=1)
        DesvPadExt[0][1] = R12.std(ddof=1)
        DesvPadExt[0][2] = R13.std(ddof=1)
        DesvPadExt[0][3] = T1.std(ddof=1)
        DesvPadExt[1][0] = R21.std(ddof=1)
        DesvPadExt[1][1] = R22.std(ddof=1)
        DesvPadExt[1][2] = R23.std(ddof=1)
        DesvPadExt[1][3] = T2.std(ddof=1)
        DesvPadExt[2][0] = R31.std(ddof=1)
        DesvPadExt[2][1] = R32.std(ddof=1)
        DesvPadExt[2][2] = R33.std(ddof=1)
        DesvPadExt[2][3] = T3.std(ddof=1)

        DesvPadInt[0][0] = PI_11.std(ddof=1)
        DesvPadInt[0][2] = PI_13.std(ddof=1)
        DesvPadInt[1][1] = PI_22.std(ddof=1)
        DesvPadInt[1][2] = PI_23.std(ddof=1)

        DesvPadDist = PD_1.std(ddof=1), PD_2.std(ddof=1), PD_3.std(ddof=1), PD_4.std(ddof=1), PD_5.std(ddof=1)


        print("Matriz Intrinsecos: ")
        print("Media")
        print(IntParam)
        print("Desvio Padrao")
        print(DesvPadInt)
        print()

        print("Matriz Extrinsecos: ")
        print("Media")
        print(ExtParam)
        print("Desvio Padrao")
        print(DesvPadExt)
        print()

        print("Distorcao: ")
        print("Media")
        print(DistParam)
        print("Desvio Padrao")
        print(DesvPadDist)

        PI_11 = np.empty(0)
        PI_13 = np.empty(0)
        PI_22 = np.empty(0)
        PI_23 = np.empty(0)
        PD_1 = np.empty(0)
        PD_2 = np.empty(0)
        PD_3 = np.empty(0)
        PD_4 = np.empty(0)
        PD_5 = np.empty(0)
        R11 = np.empty(0)
        R12 = np.empty(0)
        R13 = np.empty(0)
        R21 = np.empty(0)
        R22 = np.empty(0)
        R23 = np.empty(0)
        R31 = np.empty(0)
        R32 = np.empty(0)
        R33 = np.empty(0)
        T1 = np.empty(0)
        T2 = np.empty(0)
        T3 = np.empty(0)

        print("Clique em 2 pontos para medir a distancia")
        correct_distortion(WebCam, IntParam, DistParam, ExtParam, 4)


if __name__ == '__main__':
    main()
