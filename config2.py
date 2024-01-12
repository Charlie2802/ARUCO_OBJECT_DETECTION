import os
from ultralytics import YOLO

ROOT_DIR = '/Users/aaditya/Desktop/aruco-markers-2-3'

model = YOLO("yolov8n.yaml") 

results = model.train(data=os.path.join(ROOT_DIR, "aruco2.yaml"), epochs=1) 