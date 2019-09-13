import sys
import numpy as np
import cv2
from camera_calibration import calibration
from camera_calibration import correct_distortion

coord1 = (0,0)
coord2 = (0,0)
firstclick = 0

def Draw_line(event,x,y,flags,param):
    global coord1, coord2, firstclick
 
    if (event==cv2.EVENT_LBUTTONDOWN) & (firstclick==0):
        coord1 = (x,y)
        firstclick = 1
    elif (event == cv2.EVENT_LBUTTONDOWN) & (firstclick==1):
        coord2 = (x,y)

    

if(str(sys.argv[1])) == '-r1':
    img = cv2.imread(str(sys.argv[2]))
    print("Faca os cliques e, em seguida, pressione ESC...")
    cv2.imshow("imagem", img)
    cv2.setMouseCallback('imagem',Draw_line)

    k = cv2.waitKey(0)
    
    if k == 27 & firstclick==1:
        cv2.destroyAllWindows()
    clone = img.copy()
    cv2.line(clone, coord1, coord2, (0,0,255), 1)
    cv2.imshow("Linha", clone)

    distancia = np.subtract(coord1, coord2)
    distancia = np.power(distancia, 2)
    distancia = np.sum(distancia)
    distancia = np.sqrt(distancia)

    print(distancia)

    k = cv2.waitKey()
    if k == 27:
        cv2.destroyAllWindows()
    
if (str(sys.argv[1]) == '-r2'):
    cam = cv2.VideoCapture(0)
    mtx, dist = calibration(cam, 28, 8, 6, 5, 5)
    correct_distortion(cam, mtx, dist)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                