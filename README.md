# Training Deep Learning Models for Vision - Compact Course

A compact course on using deep learning methods for computer vision.
Based on material from the [EMBL deep learning course](https://github.com/kreshuklab/teaching-dl-course-2019).

See the [course website](https://hci.iwr.uni-heidelberg.de/ial/adl) for details on the version of this course taught at Heidelberg University in October 2020.

## Requirements

In order to follow this course you should have basic knowledge of machine learning and/or computer vision and have python programming experience.
The course will be taught using [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#), in order to use it you need a google account and about 1GB of free space on your [Google Drive](https://www.google.com/drive/). 


## Content

### Day 1:

- Deep Learning for Vision
    - Basics of Machine learning, focus on supervised learning
    - Basics of Neural Networks: MLPs and SGD
    - Deep Learning Frameworks and pytorch

Practical part:
- Data preperation for pytorch
- Image Classification (Logistic Regression -> MLP)


### Day 2:

- Deep Learning for Vision:
    - Introduction to CNNs
    - More on traiing and data augmentations
    - Advanced architectures: ResNet, ...

Practical part:
- Image classification with CNN on CIFAR10
- Data Augmentation and advanced architectures on CIFAR10


### Day 3:

- Advanced topics:
    - Image segmentation and denoising with U-Net (in detail)
    - Other applications: object detection (depth estimation, ...)
    - Intro to biomedical image analysis
    - More on loss functions

Practical part:
- (CNN and Data Augmentation on CIFAR10 continued)
- 2d segmentation on DSB


### Day 4:

- Advanced topics:
    - More on architectures: Res-Net, Dense-Net

Practical part: Choice of larger exercise (tentative)
- Build the best classification network + evaluation on Fashion MNIST
- Build the best segmentation network + post-processing for instance segmentation on DSB or ISBI2012 Challenge
- Bring your own data / small project

### Day 5:

Work on exercise and presentation of exercise results.
