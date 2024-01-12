import os
from ultralytics import YOLO

ROOT_DIR = '/home/user/Desktop/ARUCO_OBJECT_DETECTION'

model = YOLO("yolov8n.yaml") 

results = model.train(data=os.path.join(ROOT_DIR, "aruco4.yaml"), epochs=500) 