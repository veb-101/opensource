import numpy as np
import cv2
from skimage.filters import threshold_local

try:
    cap.release()
except: 
    pass


cap = cv2.VideoCapture(1) 
doc=False
while(True):
    ret, img = cap.read()
   # img = cv2.flip( img, 1 ) 
    #img = cv2.resize(img, (0,0), fx=0.1, fy=0.1)
    img = cv2.resize(img, (400,600))
    if cv2.waitKey(1) & 0xFF == ord('a'):
        doc = False
        
                   

    
    img2 = img.copy()

    rows, cols, chan=img2.shape

    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(imgray, (5, 5), 0)
    edges2 = cv2.Canny(gray, 75, 200)

    #edges1 = cv2.Canny(imgray,100,300)
    #edges1 = cv2.Canny(imgray,219,390)

    #edges2 = cv2.GaussianBlur(edges1,(5,5),0)

    _,contours, hierarchy = cv2.findContours(edges2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    #print(len(contours))
    #print(len(approx))        
    cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:10]#give me the 10 biggest contours
        #find the biggest area
    #c = max(contours, key = cv2.contourArea)

    for cc in cnts:
        

        cv2.drawContours(img, cc,-1 , (0,255,0), 3)
        epsilon = 0.09*cv2.arcLength(cc,True)
        approx = cv2.approxPolyDP(cc,epsilon,True)
        if len(approx) == 4:
                    if doc == False:
                        pts1 = np.float32(approx.reshape(-1,2))
                        pts2 = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]])
                        hold= pts1[1].copy()
                        pts1[1] = pts1[0].copy()
                        pts1[0] = hold
                        for i,xx in enumerate(pts1):
                        #cv2.circle(img2,tuple(xx), 55, (0,255,0), -1)
                             cv2.putText(img2,str(i),tuple(xx), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,255,155), 2, cv2.LINE_AA)



                        M = cv2.getPerspectiveTransform(pts1,pts2)

                        dst = cv2.warpPerspective(img2,M,(cols,rows))
                        #dst = cv2.adaptiveThreshold(dst,80,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,5,5)
                        #dst = cv2.adaptiveThreshold(dst,80,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,5)
                        #T = threshold_local(dst, 11, offset = 11, method = "gaussian")
                        #dst = (dst > T).astype("uint8") * 255

                        cv2.imshow('dst',dst)

                        cv2.imshow('image',img)
                        doc = True




    cv2.imshow('edges',edges2)
                #cv2.imshow('blurr',edges2)

    cv2.imshow('imaee',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

#cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()