# Instance segmentation for bio-image analysis in 3d

Segmenting the structures of interest is an important task for analysing microscopy images.
We have already covered this [in an exercise on the second day](https://github.com/constantinpape/training-deep-learning-models-for-vison/tree/master/day3#exercises).
However, for many applications the microscopy data is 3 dimensional which makes the segmentation task more difficult.
For this exercise, we will use publicly available data from confocal microscopy that
shows plant cells during development. You can find different volumes for training, validation and testing [here](https://oc.embl.de/index.php/s/8giJ7SnNfknMzHO).

Below you can see a zoom-in with the raw data and the segmentation labels as well as the corresponding boundary labels.

![raw](https://user-images.githubusercontent.com/4263537/94693558-29d5eb00-0334-11eb-833e-9f1ded0cb620.png)
![labels](https://user-images.githubusercontent.com/4263537/94693568-2cd0db80-0334-11eb-8053-709d0cfbdef1.png)
![boundaries](https://user-images.githubusercontent.com/4263537/94693573-2e9a9f00-0334-11eb-97f1-7056506cbc38.png)

The goal of this exercise is to implement a 3d U-Net that learns to segment boundaries and work on
post-processing to obtain a segmentation from the network results.


## Exercise

In this exercise you should:
- Explore the plant dataset
- Write a dataloader and transformations for this dataset
- Based on the 2D U-Net implementation from the previous exercise, implement a 3D U-Net
- Train 2D U-Net and 3d U-Net to predict cell boundaries and compare the results 

Once you have boundary predictions, you can explore how to obtain an instance segmentation.
Note that full 3d instance segmentation is a challenging task, but you should be able to 
produce decent 2d segmentations.

**Hints:**

The dataset contains a lot of background. You should exclude pathches with too much background during the training.

For going from boundaries to a segmentation, [connected components](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html) or [watersheds](https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html) can be used.

Some other usefuld functionality: [find_boundaries](https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.find_boundaries), [mark_boundaries](https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.mark_boundaries).
