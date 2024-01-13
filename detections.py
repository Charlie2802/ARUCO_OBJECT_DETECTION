from ultralytics import YOLO
import cv2

model = YOLO('/Users/aaditya/Desktop/aruco-markers-2-4/runs/detect/train3/weights/best.pt')  # load a custom model

results = model('/Users/aaditya/Desktop/aruco-markers-2-4/train/images/aruco_3.jpg') 
from PIL import Image

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    c=boxes.xyxy
    bbox =c.numpy()
    bbox=bbox.tolist()[0]
    x_min, y_min, x_max, y_max = map(int, bbox)
    print(x_min)
   # print(x_min)
###     https://docs.ultralytics.com/modes/predict/#thread-safe-inference   ####
#### use the abhove mentioned link for various kind of predictions ###


