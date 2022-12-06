# Object-Recognition
 Software to recoginize objects and people.


# Requirements
Python 3.9.x

## Pip requirements
opencv-python==4.5.4.58
opencv-contrib-python==4.5.4.58

# How to run on video:
In detector.py at line 8  | cap = cv2.VideoCapture("./samples/street_vid.mp4")
Copy the full path to your video file or place it under the samples folder and replace street_vid.mp4 to your_file.mp4

# How to run on camera:
In detector.py at line 8 | cap = cv2.VideoCapture("./samples/street_vid.mp4")
replace that line with:

cap = cv2.VideoCapture(0).

0 = webcam
1 = any attached camera other then the first webcam.