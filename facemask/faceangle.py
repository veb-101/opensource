import numpy as np
import cv2


class faceangle:
    def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
        desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight
 
        # if the desired face height is None, set it to be the
        # desired face width (normal behavior
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth
            
    def align(self, image, gray, cords):
        # convert the landmark (x, y)-coordinates to a NumPy array
       
        from collections import OrderedDict
#An OrderedDict is a dictionary subclass that remembers the order in which its contents are added.
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
            
        shape = cords
 
        # extract the left and right eye (x, y)-coordinates
        (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]
        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
 
        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180    
        
        return angle
        
      