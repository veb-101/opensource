import glob
#path =r"C:\Users\DELL\devworkspace\drumsounds\drums"  #you can also specify complete path
rlist = []
drumsl =[] 
ringsl = []
for filename in glob.glob('drumsounds/ringsound/*.wav'): 
    ringsl.append(filename)
print(ringsl)    

for filename in glob.glob('drumsounds/drums/*.wav'): 
    drumsl.append(filename)
print(drumsl)    

def checkin(x,y,listd):
    found = False
    for i, it in enumerate(listd):
       # print (y > it[0],y < it[1], x > it[2], x < it[3])
        if y > it[0] and y < it[1] and x > it[2] and x < it[3]:
            found = True
            break

    return found,i
	
	
def padd(x,y):
    found = False
    padder=60
    try:
        for i, it in enumerate(d1):
           # print (y > it[0],y < it[1], x > it[2], x < it[3])
            if y > it[0] - padder and y < it[1] +padder and x > it[2] - padder and x < it[3] +padder:
                found = True
                return found,i
                break
        for i, it in enumerate(r1):
           # print (y > it[0],y < it[1], x > it[2], x < it[3])
            if y > it[0] - padder and y < it[1] +padder and x > it[2] - padder and x < it[3] +padder:
                found = True
                return found,i
                break
    except:
        pass

    return found




import numpy as np
import cv2
import winsound

d1 = [[368, 480, 182, 273] , [368, 480, 348 ,441], [348, 480, 2 , 117] ,  [343, 480, 513, 640]]
r1 = [ [80 ,163 ,2 ,80] , [2 ,50 ,144 ,231]   ,  [0 ,49 ,385 ,486] ,  [88 ,156 ,564 ,640]]
    
touching= False    
xd1=0
yd1=0
hd1 = 480
wd1 = 640
kernel = np.ones((5,5),np.uint8)

drum2left = cv2.imread('drumpics/drumcanvas.png')
testr = cv2.resize(drum2left, (wd1, hd1))


cap = cv2.VideoCapture(0)
cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
while(True):
    ret, frame = cap.read()
    frame = cv2.flip( frame, 1 ) 
    if ret:
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # define range of blue color in HSV
        lower_red = np.array([150,70,173])
        upper_red = np.array([179,189,255])


        #Threshold the HSV image to get only red colors
        mask = cv2.inRange(hsv, lower_red, upper_red)
        mask = cv2.erode(mask,kernel,iterations = 1)
        mask = cv2.dilate(mask,kernel,iterations =2)
        _,contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) >0:
            cnt = sorted(contours, key=cv2.contourArea,reverse=True)
            #cnt = max(contours, key = cv2.contourArea)
            for c in cnt[:2]:
                
                x,y,w,h = cv2.boundingRect(c)
                area = cv2.contourArea(c)
                if area > 150:
                    midx = int(x+(w/2))
                    midy = int(y +(h/2))
                   # cv2.circle(frame,(midx,midy), 5, (0,255,0), -1)

                    #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    vald   = checkin(midx,midy,d1)
                    if vald[0]:

                        if touching == False:
                            winsound.PlaySound(drumsl[vald[1]], winsound.SND_ASYNC)
                            touching = True
                            #print(vald[1])

                    else:
                        vald   = checkin(midx,midy,r1)
                        if vald[0]:
                            if touching == False:
                                winsound.PlaySound(ringsl[vald[1]],winsound.SND_ASYNC)
                                #print(vald[1])
                                touching = True

                        else:
                            ptest= padd(midx,midy)
                            if ptest:
                                touching =False


                
                #print(area)
        # frame[yd1:yd1+hd1, xd1:xd1+wd1] =  testr
        img2gray = cv2.cvtColor(testr,cv2.COLOR_BGR2GRAY)

        roi = frame[yd1:yd1+hd1, xd1:xd1+wd1] 
        rett, mask = cv2.threshold(img2gray, 247, 255, cv2.THRESH_BINARY_INV)

        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

        img2_fg = cv2.bitwise_and(testr,testr,mask = mask)

        combined = cv2.add(img1_bg,img2_fg)
        #print(combined.shape,mask.shape,'comb and mask')
        frame[yd1:yd1+hd1, xd1:xd1+wd1] = combined          

                   
       #print(frame.shape) 

    cv2.imshow('image2',frame)
    k = cv2.waitKey(1) 
    if k == ord('q'):
        break
        #cv2.imwrite('roombak.jpg',frame)
    elif k == ord("c"):
            r = cv2.selectROI(frame)   # r gives cols, rows, width and height in this order           
            print(int(r[1]),int(r[1]+r[3]), int(r[0]),int(r[0]+r[2]))
            print(r)
            

cap.release()
cv2.destroyAllWindows()	
