import numpy as np
import cv2
import time

def calibration(WebCam, tam_quad, board_h, board_w, time_step, max_images):
    '''
    Metodo para calibrar a camera, utiliza o padrao de calibracao forncido (pattern.pdf) para
calcular a matriz dos parametros intrinsecos da camera e os parametros de distorcao da mesma

    Parametros:
        -WebCam: Objeto do openCV que abriu a webcam do computador
        -tam_quad: Tamanho (mm) do quadrado no padrao impresso
        -board_h: Quantidade de cantos com intersecao de 4 quadrados na vertical
        -board_w: Quantidade de cantos com intersecao de 4 quadrados na horizontal
        -time_step: Tempo (s) de espera entre deteccoes para poder movimentar o padrao
        -max_images: Numero total de fotos tiradas do padrao para fazer a calibracao

    Retorno:
        -mtx: matriz dos parametros intrinsecos da camera calculados na calibracao
        -dist: parametros de distorcao da camera calculados na calibracao
    '''
    # Criterio de parada
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, tam_quad, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((board_h*board_w,3), dtype='float32')
    objp[:,:2] = np.mgrid[0:board_w,0:board_h].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    #Set start time for the detections
    start_time = time.time()

    #Contador para o numero de imagens detectadas
    detected_images = 0

    while detected_images != max_images:
        #Ve quant tempo tem desde a ultima deteccao
        elapsed = time.time() - start_time

        grab, img = WebCam.read()
        if not grab or img is None:
            break

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        cv2.imshow("Webcam", img)

        # Acha as bordas do tabuleiro de xadrez
        ret, corners = cv2.findChessboardCorners(gray, (board_w,board_h),None)

        # If found, add object points, image points (after refining them)
        if ret == True and elapsed > time_step:
            detected_images += 1
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Desenha na imagem e mostra os cantos do xadrez
            img = cv2.drawChessboardCorners(img, (board_w,board_h), corners2,ret)
            cv2.imshow('Corners Detected',img)

            #Apos detectar o xadrez em uma imagem, da um sleep de 2s para mudar o xadrez de posicao
            start_time = time.time()

        #Aperte a tecla 'q' para encerrar o programa
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break

    # destroi as janelas usadas para a calibracao
    cv2.destroyAllWindows()

    #Faz a calibracao da camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    print('Intrinsic parameters matrix:\n{}'.format(mtx))
    print('Distortion parameters:\n{}'.format(dist))

    return mtx, dist, rvecs, tvecs

def correct_distortion(WebCam, mtx, dist):
    '''
    Metodo para corrigir a distorcao na imagem da webcam e mostrar na tela a imagem original da camera e a
imagem sem distorcao

    Parametros:
        -WebCam: Objeto do openCV que abriu a webcam do computador
        -mtx: matriz dos parametros intrinsecos da camera calculados na calibracao
        -dist: parametros de distorcao da camera calculados na calibracao
    '''

    #Inicializa as janelas raw e undistorted
    cv2.namedWindow("raw")
    cv2.namedWindow("undistorted")

    grab, img = WebCam.read()
    h,  w = img.shape[:2]
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # Mapeamento para retirar a distorcao da imagem
    mapx, mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)

    print('Aperte q para sair')
    while True:
        grab, img = WebCam.read()
        if not grab:
            break

        cv2.imshow("raw", img)

        #remapeamento
        dst = cv2.remap(img,mapx, mapy,cv2.INTER_LINEAR)

        cv2.imshow('undistorted', dst)

        #Aperte a tecla 'q' para encerrar o programa
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    #Cria um objeto do openCV para abrir a camera
    WebCam = cv2.VideoCapture(0)

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

    mtx, dist = calibration(WebCam, tam_quad, board_h, board_w, time_step, max_images)
    correct_distortion(WebCam, mtx, dist)
