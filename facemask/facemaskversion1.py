
import cv2
import dlib
import time
import numpy as np
import glob
from bleedface import fclass
from faceangle import faceangle
from dlibhelper import shapetocord
from dlibhelper import facialpoints as fp
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"  
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

face_list = []
for filename in glob.glob('aiclubnewn/*.png'): #assuming gif
    img = cv2.imread(filename)
   # im=Image.open(filename)
    face_list.append(img)


#snapchat eye + mouth filter + shades + rotation
from faceangle import faceangle

fa = faceangle(predictor, desiredFaceWidth=256,desiredLeftEye=(0.35,0.35))
f= fclass()
fname = fp()
from dlibhelper import rect_to_bb


 #rlogo33= image_list[0]
try:
   cap.release()
except:
   pass    
cap = cv2.VideoCapture(0)
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
conf= 0.5
elog=face_list[0]
fcount=0
ffill= 0
fps=0
noface= True
timer = 14  # timer for the face change
ncounter =timer
while 1:
            
            ret, img = cap.read()
            #print(img.shape)
            if ret:
                start_time = time.time()
                imgmain = cv2.flip( img, 1 )
                cv2.putText(imgmain, 'FPS: {:.2f}'.format(fps), (20, 20),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.3,color=(0, 0, 255))
                cv2.putText(imgmain, '{}'.format(ncounter), (180, 20),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.3,color=(0, 0, 255))
                gray = cv2.cvtColor(imgmain, cv2.COLOR_BGR2GRAY)
                #faces = detector(gray, 1)  
                faces= f.facer(imgmain,conf)
                #faces = np.array(faces)
                
                if len(faces) == 0:
                    if ncounter > 0:
                        ncounter -=1
                else:
                    if ncounter == 0:
                        ffill = np.random.randint(5)
                    ncounter = timer
                
                for (x,y,w,h) in faces:
                #for face in faces:

                         
                         #dshape  = predictor(gray ,face )
                         #(x, y, w, h) = rect_to_bb(face)

                         rect = dlib.rectangle(int(x), int(y), int(w), int(h)) 
                         dshape  = predictor(gray ,rect )
                         cord = shapetocord(dshape)
                         faceangle = fa.align(imgmain, gray, cord)
                        
                                     
                         if ffill > len(face_list):
                            ffill = 0
                         
                         if ffill < len(face_list):

                    
                             #flog= face_list[ffill]
                             img = face_list[ffill]
                             rows,cols,ch = img.shape

                             #play with the zoom factor, and center

                             M = cv2.getRotationMatrix2D((rows/2,cols/2),360-faceangle,1)
                             flog = cv2.warpAffine(img,M,(cols,rows))
       
                             
                             x,y,w,h = int(x), int(y), int(w-x), int(h-y)
                             if ffill == 4:
                                const = 200
                             else:
                                const = 30 
                             x = int(x -(const/2))
                             y = int(y-(const/2) )  
                             w= w+const
                             h=h+const 
                             #print(int(x), int(y), int(w-x), int(h-y))
                             rlogof = cv2.resize(flog, (w,h)) 
                             
                             #rlogo3 = cv2.resize(image_list[0], (w,h))

                             shiftx = 0
                             shifty= 0
                             #if ffill == 1 or ffill == 2:
                              #      shifty = 20
                             #else:
                             #   shifty = 10  
                                
                             rows,cols,chann = rlogof.shape

                             #play with the zoom factor, and center


                             roi = imgmain[y+shifty:y+shifty+h, x-shiftx:x-shiftx+w ]
                             #print(roi.shape,'roi shape')   

                             img2gray = cv2.cvtColor(rlogof,cv2.COLOR_BGR2GRAY)
                             #rett, mask = cv2.threshold(img2gray, 170, 255, cv2.THRESH_BINARY_INV)
                             rett, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)




                             mask_inv = cv2.bitwise_not(mask)

                             try :

                                 img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)


                                 # Take only region of logo from logo image.
                                 img2_fg = cv2.bitwise_and(rlogof,rlogof,mask = mask)

                                 combined = cv2.add(img1_bg,img2_fg)
                                
                                                               
                                # combined2 = colortransl(imgmain,combined)
                                # img2_fg2 = cv2.bitwise_and(combined2,combined2,mask = mask)
                                # combined3 = cv2.add(img1_bg,img2_fg2)
                                 #print(combined.shape,mask.shape,'comb and mask')
                                    
                                 imgmain[y+shifty:y+shifty+h, x-shiftx:x-shiftx+w ] = combined          
                             except Exception as e: 
                                #print(e)
                                pass



         
                     #    rlogo33= image_list[fcount]  #this is for the animation
                      #   if fcount != len(image_list) -1:
                       #     fcount += 1
                        # else:
                         #   fcount = 0
                         
                    
                        
          
            fps= (1.0 / (time.time() - start_time))
            cv2.imshow('img',imgmain)
            #cv2.imshow('img2',rlogof)

            k = cv2.waitKey(1) 
            if k == ord('q'):
                    break
            if k == ord('s'):
                ffill += 1


cap.release()
cv2.destroyAllWindows()