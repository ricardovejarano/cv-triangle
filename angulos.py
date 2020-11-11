import cv2
import numpy as np

cap = cv2.VideoCapture(0)

#rangos del color azul
azulBajo = np.array([20,100,100],np.uint8)  #azul 
azulAlto = np.array([125,255,255],np.uint8)
while True:
    # captura fotograma a fotograma
    ret,frame = cap.read()
    if ret==True:
        frameHSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        #encontrar los rangos
        maskBlue = cv2.inRange(frameHSV,azulBajo,azulAlto)
        #mascara solo azul
        maskBluevis = cv2.bitwise_and(frame,frame, mask=maskBlue)
        
        #contorno
        #Buscamos los contornos de las bolas y los dibujamos
        contornos,_ = cv2.findContours(maskBlue, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contornos:
            area = cv2.contourArea(c)
            if area > 500:
                M = cv2.moments(c)
                if (M["m00"]==0): M["m00"]=1
                x = int(M["m10"]/M["m00"])
                y = int(M['m01']/M['m00'])
                cv2.circle(frame,(x,y),7,(0,0,255), -1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,'{},{}'.format(x,y),(x+10,y), font, 0.75,(0,255,0),1,cv2.LINE_AA)
                nuevocontorno = cv2.convexHull(c)
                cv2.drawContours(frame, [nuevocontorno], 0, (0,0,255), 3)
                cv2.drawContours(maskBlue, [nuevocontorno], 0, (0,0,255), 3)
                print (x,y)
        #visualizar mascara con el color azul
        cv2.imshow('maskBluevis',maskBluevis) #las partes azules
        cv2.imshow('maskBlue',maskBlue)       #color detectado se muestra en blanco
        cv2.imshow('frame',frame)           #imagen normal
        #cerrar ventanas con tecla s
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
cap.release()
cv2.destroyAllWindows()