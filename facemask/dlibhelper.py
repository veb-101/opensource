
from collections import OrderedDict

import numpy as np



def facialpoints():
    FACIAL_LANDMARKS_IDXS = OrderedDict([
            ("mouth", (48, 68)),
            ("right_eyebrow", (17, 22)),
            ("left_eyebrow", (22, 27)),
            ("right_eye", (36, 42)),
            ("left_eye", (42, 48)),
            ("nose", (27, 35)),
            ("jaw", (0, 17))
        ])    
    return FACIAL_LANDMARKS_IDXS  







def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)




def shapetocord(dshape):
            # initialize the list of (x, y)-coordinates
               cords = np.zeros((68, 2),dtype='int')

            # loop over the 68 facial landmarks and convert them to a 2-tuple of (x, y)-coordinates
               for i in range(0, 68):
                   cords[i] = (dshape.part(i).x,dshape.part(i).y)  #E.g now dshape.part[5].x will give us the point on x axis for the 6th 
                #landmark
            # return the list of (x, y) tuple-coordinates
               return cords
