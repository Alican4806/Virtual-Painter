from pickletools import uint8
import cv2 as cv
import os
import numpy as np
import HandTrackingModule as htm


get = cv.VideoCapture(0) # The webcam is opened
##The camera demonsions are adjusted.##
wCam,hCam = 1280,720
get.set(3,wCam)
get.set(4,hCam)
brushThickness = 10
eraserThickness =50
##It is access Folder which include images.##
folderPath = 'Colors for pointer' # The path is  defined to reached the file 

images = os.listdir(folderPath)
overlayList =[]
for imgs in images:
    overlayList.append(cv.imread(f'{folderPath}/{imgs}'))
 
canvas = np.zeros((hCam,wCam,3),'uint8') # The canvas is defined.
xp , yp = 0, 0  # The start point is defined for drawing



header =overlayList[6]    # First situation is adjusted as eraser.
drawColor =(0,0,0) # The color of the eraser is adjusted black.
# print(overlayList)   
# print(overlayList[0].shape)
detector = htm.handDetector()
finger = [4,8,12,16,20]
result = cv.VideoWriter('Virtual Painter.mp4',cv.VideoWriter_fourcc(*'mp4V'),10,(wCam,hCam)) # The format is defined 

while True:
    
    # 1-)  Import image
    success , img = get.read()
    img = cv.flip(img,1) # Directions reversed to get rid of mirror effective.
    
    
    
    
    # 2- Find hand landmarks
    
    img = detector.findHands(img)
    lmList = detector.findPositon(img,draw = False) # The information is gotten for the positon of the hand.
    # print(lmList)
    
    
    
    
    
    if len(lmList) !=0:
        # tip of the middle and index fingers
        # print(lmList)
        
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
        
        
        # 3-) Check whether fingers are up.
        fingers = detector.fingersUp()
        # 4-) If selection mode -two fingers are up
        if fingers[1] and fingers[2]:
            print('selection mode on')
            xp , yp = 0, 0
            if 130 > y1:
                if x1<(wCam/6):
                    header = overlayList[0]
                    drawColor = (0,0,255) # The colors are selected
                elif (wCam/6) < x2 <(wCam/3):
                    header = overlayList[1]
                    drawColor = (1,1,1) # For black color
                elif (wCam/6) < x2 <(wCam/2):
                    header = overlayList[2]
                    drawColor = (255,0,0)# For blue color
                elif (wCam/2) < x2 <(2*wCam/3):
                    header = overlayList[3]
                    drawColor = (0,255,255)# For yellow color
                elif (2*wCam/3) < x2 <(5*wCam/6):
                    header = overlayList[4]
                    drawColor = (0,255,0)# For green color
                elif (5*wCam/6) < x2 <wCam:
                    header = overlayList[5]
                    drawColor = (0,0,0)# For color of the eraser
            cv.rectangle(img,(x1-5,y1+10),(x2+5,y2-10),drawColor,cv.FILLED)   # Selection mode is shown as rectangle.
                
        # 5-) If drawing mode - index finger is up    
        elif fingers[1] == True and fingers[2]==False: # It is controlled whether the index finger is up
            print('Drawing mode on')
            cv.circle(img,(x1,y1),12,drawColor,cv.FILLED) # Drawing mode is shown as circle.
            
            if xp == 0 and yp == 0: # The start point is adjusted
                xp , yp = x1 , y1 # The location of the circle is assigned to the starting location
                
            elif drawColor == (0,0,0):
                
                cv.line(img,(xp,yp),(x1,y1),drawColor,eraserThickness)
                cv.line(canvas,(xp,yp),(x1,y1),drawColor,eraserThickness)

            else:
                    
                cv.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
                cv.line(canvas,(xp,yp),(x1,y1),drawColor,brushThickness) # The line is drawing on the canvas.
            
            
            xp , yp = x1 , y1 # The line is not continuous due to this expression.
    # Last of all, the drawing  is provided
    
    imgGray = cv.cvtColor(canvas,cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGray,0,255,cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv,cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img,imgInv)
    img = cv.bitwise_or(img,canvas)
    
    
    got = cv.resize(header,(wCam,120),3)
    h,w,c = got.shape
    img[0:h,0:w] = got        
            
            
                
    #img = cv.addWeighted(img,0.5,canvas,0.5,0) # These pictures are summed
       
    
    # cv.imshow('image',img)
    # cv.imshow('canvas image',canvas)
    # cv.imshow('Inv',imgInv)
    
    if success == True: 
        
    
        result.write(img) # The video is saved
        
        cv.imshow('image',img)
        if cv.waitKey(20) & 0xFF == ord('a'):
            break
get.release()    
cv.destroyAllWindows()