
import numpy as np
import cv2
import os
thres = 0.5 # Threshold to detect object
nms_threshold = 0.2 #(0.1 to 1) 1 means no suppress , 0.1 means high suppress 
# Setting video source. use cv2.VideoCapture(0) to use the first available camera.
cap = cv2.VideoCapture("./samples/street_vid.mp4")

# Check what type of feed is used.
if str(cap).rfind(".mp4"):
    print("[INFO]    Looking for Objects in video file....\n")
else:
    print("[INFO]    Looking for Objects in live feed.")

    # Set screen dimensions
cap.set(cv2.CAP_PROP_FRAME_WIDTH,580) #width  280
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,420) #height 120
cap.set(cv2.CAP_PROP_BRIGHTNESS,150) #brightness 

coco_file = os.path.join("model_data", "coco.names")

classNames = []
with open(coco_file,'r') as f:
    classNames = f.read().splitlines()

font = cv2.FONT_HERSHEY_PLAIN
#font = cv2.FONT_HERSHEY_COMPLEX
Colors = np.random.uniform(0, 255, size=(len(classNames), 3))
# Set variables for config files.
weightsPath = os.path.join("model_data", "frozen_inference_graph.pb")
configPath = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))

    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)

    # Draw rectangles around detected objects
    if len(classIds) != 0:
        for i in indices:
            box = bbox[i]
            confidence = str(round(confs[i], 1))
            color = Colors[classIds[i]-1]
            x,y,w,h = box[0],box[1],box[2],box[3]
            cv2.rectangle(img, (x,y), (x+w,y+h), color, thickness=2)
            #cv2.imwrite(f"./objects/object-{classNames[classIds[i]-1]}--{i}.png", img)
            cv2.putText(img, classNames[classIds[i]-1]+" "+confidence,(x+10,y+20),
                            font,1,color,2)

    cv2.imshow("Object Recognition V1.0",img)
    if cv2.waitKey(1) == ord("q"):
        print("[INFO]    EXIT!")
        exit()