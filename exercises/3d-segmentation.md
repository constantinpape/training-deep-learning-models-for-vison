# Instance segmentation for bio-image analysis in 3d

Segmenting the structures of interest is an important task for analysing microscopy images.
We have already covered this [in an exercise on the second day](https://github.com/constantinpape/training-deep-learning-models-for-vison/tree/master/day3#exercises).
However, for many applications the microscopy data is 3 dimensional; and the segmentation task more difficult.

For this exercise, we will work on a publicly available dataset that shows plant cells imaged with confocal microscopy during their development.
We have extraxted a small subset of the Arabidopsis Thaliana Ovules dataset which has to be dowloaded
from [here](https://oc.embl.de/index.php/s/8giJ7SnNfknMzHO). The 3d images are saved in the HDF5 file format
and can be easily read into numpy arrays using the [h5py](https://docs.h5py.org/en/stable/) library.

Each HDF5 file contains two datasets:
- `raw` containing the 3d images from the confocal microscope
- `label` containing the dense ground truth labels for each cell (each cell has it's own label)

Below you can see the raw image data `raw`, the segmentation labels `labels` and the boundaries that we will use as training target
for a single image plane.

![raw](https://user-images.githubusercontent.com/4263537/94693558-29d5eb00-0334-11eb-833e-9f1ded0cb620.png)
![labels](https://user-images.githubusercontent.com/4263537/94693568-2cd0db80-0334-11eb-8053-709d0cfbdef1.png)
![boundaries](https://user-images.githubusercontent.com/4263537/94693573-2e9a9f00-0334-11eb-97f1-7056506cbc38.png)

The goal of this exercise is to implement a 3d U-Net that learns to segment boundaries and work on
post-processing to obtain a segmentation from the network results.


## Exercise

- Write labels to boundaries (`LabelToBoundary`) transformation in order to recover the target labels for the semantic segmentation task.
Use [find_boundaries function](https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.find_boundaries))
- Implement `OvuleDataset` class which returns randomly sampled patches of a given size from the dataset, e.g.: 
```python
ds = OvuleDataset(root_dir, phase='train', patch_size=(64, 64, 64))

for input_image, boundary_labels in ds:
    # show sample input and target patches
    pass
```

- Implement 3D data augmentations, e.g. rotations, random flipping and additive gaussian noise on the input
Your chain of transformation for the training phase should look similar to:
```python
train_transform = torchvision.transforms.Compose([
    Normalize(),
    Rotate(random_seed=0), # make sure that input and target validations have the same seed, i.e. input and target are always rotated by the same degree
    RandomFlip(random_seed=0), # make sure input and target are randomly flipped in the same way
    AdditiveGaussianNoise(sigma=0.5),
    ToTensor()
])

val_transform = torchvision.transforms.Compose([
    Rotate(random_seed=0), # make sure that input and target validations have the same seed, i.e. input and target are always rotated by the same degree
    RandomFlip(random_seed=0), # make sure input and target are randomly flipped in the same way
    LabelToBounday(),
    ToTensor()
])

```
- Train a 3D U-Net for the semantic segmentation task of predicting the cell boundaries. In order to implement the 3D-Unet,
take the base model from the [excercise](../day3/unet_pytorch.ipynb) and replace `Conv2d` layers by `Conv3d` layers
as well as `MaxPool2d`, by `MaxPool3d`. Use [BinaryCrossEntropy](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html),
Adam optimizer with the initial learning rate of `0.0001` and train for 5000 iterations. 
See [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/abs/1606.06650) for more details
(Optional) Add [GroupNorm](https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html#torch.nn.GroupNorm)
layers before each convolutional layer in order to speed up the training convergance and improve the network performance.
- Try postprocessing strategies to go from boundaries to segmentation, e.g. thresholding the boundary probability maps and running connected components.
What is the problem with this simple approach?
- Using the network trained on the training set, predict the cell boundaries on the test set of the Ovules dataset. Visualize the results
e.g. show the a couple of z-slices of the input image and predicted boundaries and the target boundaries.
