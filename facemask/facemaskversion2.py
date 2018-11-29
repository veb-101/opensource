import cv2
import dlib
import numpy as np
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat" 
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor(PREDICTOR_PATH)


def shapetocord(dsahpe):
    # initialize the list of (x, y)-coordinates
    cords = np.zeros((68, 2),dtype='int')

    # loop over the 68 facial landmarks and convert them to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        cords[i] = (dshape.part(i).x,dshape.part(i).y)  #E.g now dshape.part[5].x will give us the point on x axis for the 6th 
        #landmark
    # return the list of (x, y) tuple-coordinates
    return cords
	
	
	
	
from collections import OrderedDict

#FACIAL_LANDMARKS_IDXS =  {'key1':'item','key2':'item2'} 

FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])



face_list = []
for filename in glob.glob('facefill2/*.png'): #assuming gif
    img = cv2.imread(filename)
   # im=Image.open(filename)
    face_list.append(img)
	
	
	
	
eye_list = []
for filename in glob.glob('eyefill2/*.png'): #assuming gif
    img = cv2.imread(filename)
   # im=Image.open(filename)
    eye_list.append(img)




#snapchat eye + mouth filter + shades + rotation

import time
import numpy as np
import cv2
from bleedface import fclass

#cap.release()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
f= fclass()
rlogo = cv2.imread('eyeball.jpg')
rlogo33= image_list[0]
lis = ['left_eye','right_eye']
#rlogo = image_list[0]
try:
   cap.release()
except:
   pass    
cap = cv2.VideoCapture(0)
varr='none'
mopen=False
conf= 0.5
elog=face_list[0]
fcount=0
ffill= 0
efill = 0
#from colortranslocal import colortransl
from faceangle import faceangle
fa = faceangle(predictor, desiredFaceWidth=256,desiredLeftEye=(0.35,0.35))
fps=0
while 1:
            
            ret, img = cap.read()
            #print(img.shape)
            if ret:
                start_time = time.time()
                imgmain = cv2.flip( img, 1 )
                cv2.putText(imgmain, 'FPS: {:.2f}'.format(fps), (20, 20),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.3,color=(0, 0, 255))
                
                gray = cv2.cvtColor(imgmain, cv2.COLOR_BGR2GRAY)
                faces= f.facer(imgmain,conf)
                faces = np.array(faces)
                
                #if len(faces) > 0:
                for (x,y,w,h) in faces:

                         
                         
                         rect = dlib.rectangle(int(x), int(y), int(w), int(h)) 
                         dshape  = predictor(gray ,rect )
                         cord = shapetocord(dshape)
                         faceangle = fa.align(imgmain, gray, cord)
                         #eyesize = (cord[37][0] - cord[41][0]) / (cord[15][0] - cord[1][0])
                         #varr = eyesize
                            
                                                
                         if efill > len(eye_list):
                            efill= 0
                         
                         if efill < len(eye_list):

                    
                             img= eye_list[efill] 
                             rows,cols,ch = img.shape
                             M = cv2.getRotationMatrix2D((rows/2,cols/2),360-faceangle,1)
                             elog = cv2.warpAffine(img,M,(cols,rows))
                             (j, k) = FACIAL_LANDMARKS_IDXS['nose']
                             pts = cord[j:k]

                             x,y,w,h = cv2.boundingRect(pts)
                             #if efill == 0:
                             if efill == 2:
                                consty = int(h*2.7 )  #20
                                constx = int(w*9 )  #120
                             else:
                                consty = int(w*1.2 ) #20
                                constx = int(h*1.9 )  #120
                                
                             
                             x = int(x -(constx/2))
                             y = int(y-(consty/2) )  
                             w= w+constx
                             h=h+consty   
                             rlogof = cv2.resize(elog, (w,h)) 
                             #rlogo3 = cv2.resize(image_list[0], (w,h))

                             shiftx = 0
                             if efill == 0:
                                    shifty = -10
                             elif efill == 1:
                                shifty = -30
                             else:
                                
                               shifty = -120  #-10  
                                
                             rows,cols,channels = rlogof.shape
                             roi = imgmain[y+shifty:y+shifty+h, x-shiftx:x-shiftx+w ]
                             #print(roi.shape,'roi shape')   

                             img2gray = cv2.cvtColor(rlogof,cv2.COLOR_BGR2GRAY)
                             #rett, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY_INV)
                             rett, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)


                             mask_inv = cv2.bitwise_not(mask)

                             try :

                                 img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)


                                 # Take only region of logo from logo image.
                                 img2_fg = cv2.bitwise_and(rlogof,rlogof,mask = mask)

                                 combined = cv2.add(img1_bg,img2_fg)
                                 #combined2 = colortransl(imgmain,combined)
                                 #img2_fg2 = cv2.bitwise_and(combined2,combined2,mask = mask)
                                 #combined3 = cv2.add(img1_bg,img2_fg2)


                                
                                 #print(combined.shape,mask.shape,'comb and mask')
                                 imgmain[y+shifty:y+shifty+h, x-shiftx:x-shiftx+w ] = combined       
                             except Exception as e: 
                                #print(e)
                                pass

              
                       
                         if ffill > len(face_list):
                            ffill = 0
                         
                         if ffill < len(face_list):

                    
                             #flog= face_list[ffill]
                             img = face_list[ffill]
                             rows,cols,ch = img.shape

                             #play with the zoom factor, and center

                             M = cv2.getRotationMatrix2D((rows/2,cols/2),360-faceangle,1)
                             flog = cv2.warpAffine(img,M,(cols,rows))
                             
                             
                             #flog = cv2.getRotationMatrix2D((rows/2,cols/2),0,1)
                             #flog = cv2.warpAffine(flog,M,(cols,rows))
                             (j, k) = FACIAL_LANDMARKS_IDXS['mouth']
                             pts = cord[j:k]

                             x,y,w,h = cv2.boundingRect(pts)
                             if ffill == 0:
                                const = int(w*1.7 )#120
                             else:
                                const = int(w*1.4 )#80 
                             x = int(x -(const/2))
                             y = int(y-(const/2) )  
                             w= w+const
                             h=h+const 
                             
                             rlogof = cv2.resize(flog, (w,h)) 
                             
                             #rlogo3 = cv2.resize(image_list[0], (w,h))

                             shiftx = 0
                             if ffill == 1 or ffill == 2:
                                    shifty = 20
                             else:
                                shifty = 10  
                                
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



         
                         rlogo33= image_list[fcount]  #this is for the animation
                         if fcount != len(image_list) -1:
                            fcount += 1
                         else:
                            fcount = 0
                         
                    
                        
                        
                         if (cord[57][1] - cord[51][1])   > 36:  #jaw trigger
                                
                             (j, k) = FACIAL_LANDMARKS_IDXS['mouth']
                             pts = cord[j:k]
                             x,y,w,h = cv2.boundingRect(pts)
                             const=90
                             x = int(x -(const/2))
                             y = int(y-(const/2) )  
                             w= w+const
                             h=h+const   
                             rlogo3 = cv2.resize(rlogo33, (w,h)) 
                             #rlogo3 = cv2.resize(image_list[0], (w,h))


                             rows,cols,channels = rlogo3.shape
                             roi = imgmain[y+55:y+55+h, x-55:x-55+w ]
                             #print(roi.shape,'roi shape')   

                             img2gray = cv2.cvtColor(rlogo3,cv2.COLOR_BGR2GRAY)
                             rett, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

                             mask_inv = cv2.bitwise_not(mask)

                             try :

                                 img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)


                                 # Take only region of logo from logo image.
                                 img2_fg = cv2.bitwise_and(rlogo3,rlogo3,mask = mask)

                                 combined = cv2.add(img1_bg,img2_fg)
                                 #print(combined.shape,mask.shape,'comb and mask')
                                 imgmain[y+55:y+55+h, x-55:x-55+w ] = combined          
                             except: 
                                pass                        
   
                                
                        
                         for x in lis:
                             (j, k) = FACIAL_LANDMARKS_IDXS[x]
                             pts = cord[j:k]
                             x,y,w,h = cv2.boundingRect(pts)
                             #eyeroi = img[y:y + h, x:x + w]
                             varr = cord[37][1] - cord[19][1]
                             #varr = eyeroi.size  / (cord[15][0] - cord[1][0])
                             #print(rlogo.shape,'orignal logo shape')
                             const=30
                             x = int(x -(const/2))
                             y = int(y-(const/2) )  
                             w= w+const
                             h=h+const   
                             rlogo2 = cv2.resize(rlogo, (w,h)) 
                             #rlogo3 = cv2.resize(image_list[0], (w,h)) 
   
                                
                             #print(rlogo.shape,'resized logo shape')   


                             rows,cols,channels = rlogo2.shape
                             roi = imgmain[y:y+h, x:x+w ]
                             #print(roi.shape,'roi shape')   

                             img2gray = cv2.cvtColor(rlogo2,cv2.COLOR_BGR2GRAY)
                             rett, mask = cv2.threshold(img2gray, 247, 255, cv2.THRESH_BINARY_INV)

                             mask_inv = cv2.bitwise_not(mask)
                             if  int(varr) > 28:  #adjust this value to control eye triggers
                                 try :

                                     img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)


                                     # Take only region of logo from logo image.
                                     img2_fg = cv2.bitwise_and(rlogo2,rlogo2,mask = mask)

                                     combined = cv2.add(img1_bg,img2_fg)
                                     #print(combined.shape,mask.shape,'comb and mask')
                                     imgmain[y:y+h, x:x+w ] = combined          
                                 except: 
                                    pass
            cv2.namedWindow('img', cv2.WINDOW_NORMAL)
          
            fps= (1.0 / (time.time() - start_time))
            cv2.imshow('img',imgmain)
            k = cv2.waitKey(1) 
            if k == ord('a'):
                    break
            if k == ord('s'):
                ffill += 1
            if k == ord('d'):
                efill += 1    


cap.release()
cv2.destroyAllWindows()	