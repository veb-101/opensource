import cv2
import numpy as np
class fclass:

    # Constructor
    def __init__(self ):
      self.net =  cv2.dnn.readNetFromCaffe("caffeandproto/deploy.prototxt.txt", "caffeandproto/res10_300x300_ssd_iter_140000.caffemodel")



    def facer(self,image,conf):
        flist=[]
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()  #this is a forward pass through the network

        for i in range(0, detections.shape[2]):
            #print(detections.shape[2])

            confidence = detections[0, 0, i, 2]

            # filter out weak detections by setting your confidence try to put confidence above 0.5
            if confidence < conf:
                continue

            faces = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = faces.astype("int")
            flist.append((startX, startY, endX, endY))
        return flist    
