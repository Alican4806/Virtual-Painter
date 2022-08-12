'''
We will make module which called HandTrackingModule.py to able to use next projects

'''

import cv2 as cv
import mediapipe as mp
import time


class handDetector(): # We created the class to be functional 
    
    
    # This part is initialization part that is run first.
    def __init__(self, mode= False, maxHands = 2, modelCom=1 , detectionCon= 0.5, trackCon = 0.5):# We copy from Hands class to be more flexible.
        self.mode = mode
        self.maxHands = maxHands            # These parts are OOP.
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelCom = modelCom

        self.mpHands = mp.solutions.hands # we reached hands module 
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.modelCom, # we assign default parameters
                                        self.detectionCon, self.trackCon) # we occur object from Hands function # we don't give any value of the Hands function because the false is given default

        self.mpDraw = mp.solutions.drawing_utils # the object is occured by drawing_utils module. # This object help us to draw points on the our hands.
        self.finger = [4,8,12,16,20]
            # Because of this module, we get rid of doing any mathematical operations.

    def findHands(self,img,draw=True): # We created new class to find hands.

        imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB) # We converted img to rgb as hands only works in rgb state.
        self.results = self.hands.process(imgRGB) # we reached process by hands object. # The process function will process the frame    # print(results.multi_hand_landmarks)
    #print(results.multi_hand_landmarks) # we control whether my hands are  on the screen.
    
        if self.results.multi_hand_landmarks: # we controlled whether our hands are on the screen with if conditon.
            for handLms in self.results.multi_hand_landmarks: # we get information from each hand
                 if draw:
                    self.mpDraw.draw_landmarks(img,handLms,
                                               self.mpHands.HAND_CONNECTIONS) # We are trying to draw the points on the hand with the information we receive.
                            # # second parameter shows which hand # third parameter also shows connection between each points.
                             

        return img               
    
    
    def findPositon(self,img, handNo = 0, draw = True): # We created a new function that called findPositon to find where hands are
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo] # We wrote down which hand was being talked about
            for id,lm in enumerate(myHand.landmark): # we used the enumarate function to order each point # For example WRIST = 0; like where does wrist locate in the screen?
                                            #print(id,lm) # First parameter is count, second parameter is directions that got from list or tuple etc.
                h,w,c = img.shape # we assigned shape of the img to variables seperately
                                            
                cx,cy = int(lm.x*w),int(lm.y*h) # the decimal values convert to integer values
                                        # we try to get pixel value which located as x and y
                                        
                # print(id,cx,cy) # we wrote values as pixel coordinates. Also we have written id values
                self.lmList.append([id,cx,cy])# if id ==4:
                
                if draw:
                    cv.circle(img,(cx,cy),5,(0,255,0),cv.FILLED) # We adjust size of the circle the color.           
                                                                
        return self.lmList                                        
     # The function which is needed Virtual painter, was added                                  
    def fingersUp(self):
        if len(self.lmList) != 0:
            indexs = []
            # For thumbs
            if self.lmList[self.finger[0]][1]<self.lmList[self.finger[0]-1][1]: # The size mark is changed due to reverse the directions
                indexs.append(1)
                # print('index is open')
            else:
                indexs.append(0)
                # print('index is close')
                
            # Another fingers
            for id in range(1,5):
                
                if self.lmList[self.finger[id]][2]<self.lmList[self.finger[id]-2][2]:
                    # print('index is open')
                    indexs.append(1)
                else:
                    # print('index is close')
                    indexs.append(0)
            return indexs
            
            
    
  
def main():
    pTime = 0 # It is previous time and it was 0 at started
    cTime = 0 # It is current time and it was 0 at started
    cap = cv.VideoCapture(0) # we wrote 0 to use my laptop's camera. If we want to use another camera on the usb, we have to write 1. 

    detector = handDetector()  # we created object that called detector.

    while True:
        success,img = cap.read() # 'success' variable is boolean and we got video from the my laptop with cap.read()
        img =detector.findHands(img) # we didn't give another values because it is given by default
        lmList = detector.findPositon(img)
        if len(lmList)!= 0: # if this state is provided, it will print it
            print(lmList[4]) # We is written fourth value of the list

      
        cTime = time.time() # we got current time as cTime
    
        fps = 1/ (cTime-pTime) # We found fps
    
        pTime = cTime # cTime assigned as previous time 
    
        cv.putText(img,str(int(fps)),(10,100),cv.FONT_HERSHEY_SIMPLEX,3,(255,0,255),3) # we write fps value to the screen.
    
        cv.imshow("Alican",img)
        if cv.waitKey(30) & 0xFF == ord('q'): # We used function that is 0xFF == ord('q') to close the laptop's camera 
            break 

          
    
if __name__=="__main__": # It firstly runs this part in python
    main()
     
