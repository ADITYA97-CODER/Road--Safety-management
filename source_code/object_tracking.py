import cv2
import numpy as np
from object_detection import ObjectDetection
import math
import time
import torch 
model = torch.hub.load('ultralytics/yolov5', 'custom', path='source_code/yolov5s.pt')
model2 = torch.hub.load('ultralytics/yolov5', 'custom', path='source_code/helmet.pt')
classes= model.names
classes2= model2.names

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

# Initialize Object Detection
#od = ObjectDetection()

cap = cv2.VideoCapture("video.mp4")

# Initialize count
count = 0
center_points_prev_frame = []
vehicle_entering_a1 =[]
vehicle_entering_a2= []
tracking_objects = {}
track_id = 0
speed=[]
exist1 = False
exist2 = False

while True:
    ret, frame = cap.read()
    frame  = cv2.resize(frame,(700,799))

    count += 1
    area_1 = [(663, 400), (842, 395), (902, 472), (601, 478)]
    area_2 = [(557, 531), (940, 525), (1075, 676), (448, 689)]
    
    
    if not ret:
        break
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    # Point current frame
    center_points_cur_frame = []
    # Detect objects on frame
    labels , boxes = score_frame(frame,model=model)

    n= len(labels)
    for i in range(n):
        row = boxes[i]
        x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        #cv2.putText(frame, classes[int(labels[i])]+str(float(row[4])), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,5), 2)
        if classes[int(labels[i])]=="person":
            f = frame[y1:y2,x1:x2]
            if f.shape[0] and f.shape[1] >0:
             labels1, boxes1 = score_frame(f,model=model2)
             n1= len(labels1)

             for i in range(n1):
               row1 = boxes1[i]
              # x1, y1, x2, y2 = int(row[0]*f.shape[1]), int(row[1]*f.shape[0]), int(row[2]*f.shape[1]), int(row[3]*f.shape[0])
               x11, y11, x22, y22 = int(row1[0]*f.shape[1]), int(row1[1]*f.shape[0]), int(row1[2]*f.shape[1]), int(row1[3]*f.shape[0])
               cv2.putText(frame, classes2[int(labels1[i])]+str(float(row[4])), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,5), 2)
               #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
               frame[y1:y2,x1:x2] = f
               #cv2.imshow("l",f)


        cx = int(((x1+x2)/2))
        cy= int(((y1+y2)/2))
        center_points_cur_frame.append((cx, cy))
        #print("FRAME N°", count, " ", x, y, w, h)
        result1 = cv2.pointPolygonTest(np.array(area_1, np.int32), (cx, cy), False)
        result2 = cv2.pointPolygonTest(np.array(area_2, np.int32), (cx, cy), False)
        #if result1>=0 or result2>=0:
        # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Only at the beginning we compare previous and current frame
    if count <= 2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < 20:
                    tracking_objects[track_id] = pt
                    track_id += 1
    else:

        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                # Update IDs position
                if distance < 20:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue

            # Remove IDs lost
            if not object_exists:
                tracking_objects.pop(object_id)

        # Add new IDs found
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1

    for object_id, pt in tracking_objects.items():
        result1 = cv2.pointPolygonTest(np.array(area_1, np.int32), (int(pt[0]), int(pt[1])), False)

        result2 = cv2.pointPolygonTest(np.array(area_2, np.int32), (int(pt[0]), int(pt[1])), False)
        if result1>=0 or result2>=0:


        
         for s in speed:
             if s[0] == object_id:
                 cv2.putText(frame, str(int(s[1]))+'km/h', (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)
        if result1>=0:
            for x in vehicle_entering_a1:
                id = x[0]
                if id==object_id:
                    exist1=True
                    break
                else:
                    exist1=False
            if exist1==False:
                vehicle_entering_a1.append([object_id,time.time()])
            #cv2.circle(frame, pt, 5, (0, 0, 255), -1)
                


        elif result2>=0:
            for x in vehicle_entering_a2:
                id = x[0]
                if id==object_id:
                    exist2=True
                    break
                else:
                    exist2=False
            if exist2==False:
                vehicle_entering_a2.append([object_id,time.time()])
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)
    for y in vehicle_entering_a1:
        for x in vehicle_entering_a2:
            if x[0] == y[0]:
                print(x)

                if (x[1]-y[1])>0:
                 s = 30/(x[1]-y[1])*3.6
                 speed.append([x[0], s])
                


    #print(vehicle_entering_a1)
    #print(center_points_cur_frame)

    for area in [area_1, area_2]:
     cv2.polylines(frame, [np.array(area, np.int32)], True, (15,220,10), 6)
    cv2.imshow("Frame", frame)

    # Make a copy of the points
    center_points_prev_frame = center_points_cur_frame.copy()

    key = cv2.waitKey(1)
    if key == 27:
        break
print(vehicle_entering_a2)
cap.release()
cv2.destroyAllWindows()
