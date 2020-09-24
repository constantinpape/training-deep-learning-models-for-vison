# Day 3

## Content
On the third day, we will focus on fully convolutional neural networks for semantic segmentation.
in particular the U-Net architecture. The lectures will cover ([you can find the slides here](https://docs.google.com/presentation/XXXX)):
- A quick introduction to semantic segmentation with convolutional neural networks
- U-Net architecture for bioimage segmentation
- Variants of the U-Net architecture (residual blocks, normalization layers, upsampling layers)
- Other applications: Image denoising with U-Net
- More on loss functions

### Exercises:

* unet_pytorch - follow the notebook and solve all the tasks to run a unet on Kaggle nuclei dataset
* noise2noise_pytorch - pytorch implementation of "Noise2Noise: Learning Image Restoration without Clean Data"

## Additional materials

 * [Python Classes Basic Tutorial](https://www.w3schools.com/python/python_classes.asp)
 * [Python Classes Basic Tutorial 2](https://www.learnpython.org/en/Classes_and_Objects)
 * [Python Classes In-depth Tutorial](https://jeffknupp.com/blog/2014/06/18/improve-your-python-python-classes-and-object-oriented-programming/)
 * [Popular 3D U-Net implementation](https://github.com/wolny/pytorch-3dunet)
 * [Intuition Behind UNet](https://towardsdatascience.com/u-net-b229b32b4a71)
 * [Overview of Deep Learning models for Image Segmentation](https://medium.com/@arthur_ouaknine/review-of-deep-learning-algorithms-for-image-semantic-segmentation-509a600f7b57)


## References

* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
* [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/abs/1606.06650)
* [Superhuman Accuracy on the SNEMI3D Connectomics Challenge](https://arxiv.org/pdf/1706.00120.pdf)
* [Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations](https://arxiv.org/pdf/1707.03237.pdf)
* [Noise2Noise: Learning Image Restoration without Clean Data](https://arxiv.org/abs/1803.04189)