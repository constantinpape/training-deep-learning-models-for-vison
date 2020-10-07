# Object detection on Pascal VOC 2012 using pre-trained DETR

In this exercise we're going to use a per-trained for the object detection
task on [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/devkit_doc.pdf) dataset.
As has been shown during the lecture, detection models (e.g. R-CNN) rely on many hand-designed components like a non-maximum suppression procedure or anchor generation.

Very recently a conceptually simple model which removes the need for the aforementioned manual components has been
developed, see [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872).
In this exercise we're going to use a simplified version of [DETR model](https://github.com/facebookresearch/detr)
model provided by Facebook Research pre-trained on the [COCO object detection dataset](https://cocodataset.org/#home).
We're going to use this model detect objects in the images from the validation set of [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/devkit_doc.pdf) dataset.
We're going to base our exercise on the excellent [notebook](https://github.com/facebookresearch/detr#notebooks) from Facebook Reserach,
Use the following [notebook](../day4/detr_demo.ipynb) as a starting point.

## Before you start

download the Pascal VOC validation set from the [here](https://oc.embl.de/index.php/s/bkBUhSajTPP0lUP) and save it in your Google Drive. 
The archive is 2GB in size so the upload will take a while. 

# Exercise
1. Given the images from the `VOCSegmentation` dataset in the notebook, use the pre-trained DETR model to detect object.
Show bounding boxes together with the ground truth segmentation masks on 20 randomly selected images from Pascal VOC dataset.
Quantitatively how does the model perform? Do you see any irregularities between the predicted bounding boxes and the ground truth masks?

2. Quantify the object detection performance on the Pascal VOC 2012 dataset using the Mean Average Precision metric. 
Given a function which returns the ground truth bounding boxes together with their corresponding classes (see `find_boxes`) 
and bounding box predictions given by the DETR model, compute the `mAP` score at different 'intersection over union' (IoU) thresholds (e.g. 0.4, 0.5, 0.75) 
on the **entire** Pascal VOC 2012 dataset.
A detection is a true positive if it has IoU with a ground-truth box greater than a given threshold. 
Details of how to implement `mAP` for object detection ca be found 
e.g. [here](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173).

**Hint**
bear in mind that COCO dataset contains 81 classes, whereas Pascal VOC contains 20 classes. For images where DETR model
returns one of the 61 classes not present in the Pascal VOC, simply ignore the predicted bounding box instead of counting
it as a False Positive.

