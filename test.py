import cv2
import numpy as np
import math

# lowerBound=np.array([33,80,40])
# upperBound=np.array([102,255,255])

lowerBound=np.array([20,100,100],np.uint8)
upperBound=np.array([125,255,255],np.uint8)

cam= cv2.VideoCapture(0)
kernelOpen=np.ones((5,5))
kernelClose=np.ones((20,20))

# font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX,2,0.5,0,3,1)
font = cv2.FONT_HERSHEY_SIMPLEX
def drawLines(conts, img):
    if len(conts) == 3:
        M = cv2.moments(conts[0])
        M1 = cv2.moments(conts[1])
        M2 = cv2.moments(conts[2])
        if (M["m00"]==0): M["m00"]=1
        x = int(M["m10"]/M["m00"])
        y = int(M['m01']/M['m00'])
        x2 = int(M1["m10"]/M1["m00"])
        y2 = int(M1['m01']/M1['m00'])
        x3 = int(M2["m10"]/M2["m00"])
        y3 = int(M2['m01']/M2['m00'])
        cv2.line(img, (x, y), (x2, y2), (0, 255, 0), thickness=3, lineType=8)
        cv2.line(img, (x, y), (x3, y3), (0, 255, 0), thickness=3, lineType=8)
        cv2.line(img, (x2, y2), (x3, y3), (0, 255, 0), thickness=3, lineType=8)

        pointsX = [ x, x2, x3 ]
        pointsY = [ y, y2, y3 ]

        maxNumX = np.amax(pointsX)
        minNumX = np.amin(pointsX)

        maxNumY = np.amax(pointsY)
        minNumY = np.amin(pointsY)

        pointA = []
        pointB = []
        pointC = []

        for i in range(len(pointsX)):


            # C
            if maxNumX == pointsX[i]:
                pointC = [pointsX[i], pointsY[i]]
                cv2.putText(img,'{}'.format("C"),(pointC[0],pointC[1]-30), font, 0.75,(0,255,0),1,cv2.LINE_AA)
            elif maxNumY == pointsY[i]:
                pointA = [pointsX[i], pointsY[i]]
                cv2.putText(img,'{}'.format("A"),(pointA[0],pointA[1]-30), font, 0.75,(0,255,0),1,cv2.LINE_AA)
            else: 
                pointB = [pointsX[i], pointsY[i]]
                cv2.putText(img,'{}'.format("B"),(pointB[0],pointB[1]-30), font, 0.75,(0,255,0),1,cv2.LINE_AA)
        
        distanceBC = math.sqrt(math.pow((pointC[0]-pointB[0]),2) + math.pow((pointC[1]-pointB[1]),2))
        distanceBA = math.sqrt(math.pow((pointA[0]-pointB[0]),2) + math.pow((pointA[1]-pointB[1]),2))
        distanceAC = math.sqrt(math.pow((pointC[0]-pointA[0]),2) + math.pow((pointC[1]-pointA[1]),2))

        
        num = math.pow(distanceBC, 2) + math.pow(distanceBA, 2) - math.pow(distanceAC, 2)
        den = 2*distanceBC*distanceBA
        cosValue = num / den
        # angulo = (distanceBA/distanceBC)
        anguloB = math.acos(cosValue)* ( 180/ math.pi)
        # anguloB = (angulo*180) / math.pi
        cv2.putText(img,' = {}'.format(anguloB),(pointB[0]+ 10,pointB[1]-30), font, 0.75,(0,255,0),1,cv2.LINE_AA)


while True:
    ret, img=cam.read()
    img=cv2.resize(img,(340,220))

    #convert BGR to HSV
    imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # create the Mask
    mask=cv2.inRange(imgHSV,lowerBound,upperBound)
    #morphology
    maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
    maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)

    maskFinal=maskClose
    conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if len(conts) == 3:
        coords = drawLines(conts, img)
        # cv2.putText(img,'{}'.format("C"),(coords[0],coords[1]-30), font, 0.75,(0,255,0),1,cv2.LINE_AA)
    cv2.drawContours(img,conts,-1,(255,0,0),3)
    for i in range(len(conts)):
        M = cv2.moments(conts[i])
        if (M["m00"]==0): M["m00"]=1
        x = int(M["m10"]/M["m00"])
        y = int(M['m01']/M['m00'])
        cv2.putText(img,'{},{}'.format(x,y),(x+10,y), font, 0.75,(0,255,0),1,cv2.LINE_AA)
        cv2.circle(img,(x,y),7,(0,0,255), -1)
        x,y,w,h=cv2.boundingRect(conts[i])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255), 2)

    cv2.imshow("maskClose",maskClose)
    cv2.imshow("maskOpen",maskOpen)
    cv2.imshow("mask",mask)
    cv2.imshow("cam",img)
    cv2.waitKey(10)

