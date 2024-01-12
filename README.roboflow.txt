
aruco markers 2 - v2 PERFECT_ANNOTATION
==============================

This dataset was exported via roboflow.com on January 12, 2024 at 7:18 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 230 images.
Aruco are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Random shear of between -10° to +10° horizontally and -10° to +10° vertically

The following transformations were applied to the bounding boxes of each image:
* Random rotation of between -45 and +45 degrees
* Random shear of between -10° to +10° horizontally and -10° to +10° vertically


