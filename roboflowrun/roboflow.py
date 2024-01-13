from roboflow import Roboflow
rf = Roboflow(api_key="91285SVTjJdMWL7x9AcG")
project = rf.workspace().project("aruco-markers-2")
model = project.version(2).model

# infer on a local image
print(model.predict("/Users/aaditya/Desktop/ARUCO_MARKER/aruco_images2/test/images/aruco.jpgjkdkk", confidence=40, overlap=30).json())

# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())