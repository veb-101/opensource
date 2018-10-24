#dlib based detector for real time
import cv2
import dlib
import numpy as np
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

predictor = dlib.shape_predictor(PREDICTOR_PATH)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def shapetocord(dsahpe):
    # initialize the list of (x, y)-coordinates
    cords = np.zeros((68, 2),dtype='int')

    # loop over the 68 facial landmarks and convert them to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        cords[i] = (dshape.part(i).x,dshape.part(i).y)  
        
    # return the list of (x, y) tuple-coordinates
    return cords

cap = cv2.VideoCapture(0)
frames=25
varr='none'
smm=False
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
while(True):
    ret, image = cap.read()
    #image = cv2.flip(image, 1 ) 
    cv2.putText(image, str(varr), (20, 20),fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.3,color=(0, 0, 255))
    if ret:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        #faces = detector(gray, 1)       

        for x,y,w,h in faces:
        #for face in faces:   
           
            
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) 
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h)) 
            dshape  = predictor(gray ,rect )
            #dshape  = predictor(gray ,face )

            cord = shapetocord(dshape)
            for i ,(x, y) in enumerate(cord): # looping over x and y cordinated and drawing them
                #cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
               # cv2.putText(image, str(i), (x, y),fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.2,color=(0, 0, 255))
                smile = (cord[54][0] - cord[48][0]) / (cord[15][0] - cord[1][0])
                if smile > 0.40 and smm==False: 
                     smm = True
                     saver =image.copy()  
                     
                #cv2.circle(image,tuple(cord[48]), 2, (0, 0, 255), -1)
                #cv2.circle(image,tuple(cord[54]), 2, (0, 0, 255), -1)


                varr=smile 

        cv2.imshow('frame',image)
        if smm and frames > 0:
            cv2.imshow('Saved',saver)
            frames-=1

        else:
            frames = 25
            smm = False
            #cv2.imwrite('smilesave.jpg',saver)  uncomment this to actually save the picture in disk
            cv2.destroyWindow('Saved')

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break    
  

cap.release()
cv2.destroyAllWindows()
