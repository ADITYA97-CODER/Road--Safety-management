import cv2
import numpy as np
from object_detection import ObjectDetection
import math
import time
import torch 
import os
count = 0
model = torch.hub.load('ultralytics/yolov5', 'custom', path='source_code/best.pt')
classes= model.names

def score_frame(frame,model):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        frame = [frame]
        results = model(frame)
        #print(results.pandas().xyxy)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord
path= os.listdir('images')
for x in path:
        frame = cv2.imread('images/'+x)
        labels,cord = score_frame(frame , model)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        n= len(labels)
        for i in range(n):
         row = cord[i]
         x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
         f = frame[y1:y2,x1:x2]
         cv2.imwrite('image/'+str(count)+'.png',f)
         count+=1