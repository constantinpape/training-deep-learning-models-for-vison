# Instance segmentation for bio-image analysis in 3d

Segmenting the structures of interest is an important task for analysing microscopy images.
We have already covered this [in an exercise on the second day](https://github.com/constantinpape/training-deep-learning-models-for-vison/tree/master/day3#exercises).
However, for many applications the microscopy data is 3 dimensional; and the segmentation task more difficult.
For this exercise, we will use publicly available data from confocal microscopy that
shows plant cells during development, you can find it [here](https://osf.io/w38uf/).


![raw](https://user-images.githubusercontent.com/4263537/94693558-29d5eb00-0334-11eb-833e-9f1ded0cb620.png)
![labels](https://user-images.githubusercontent.com/4263537/94693568-2cd0db80-0334-11eb-8053-709d0cfbdef1.png)
![boundaries](https://user-images.githubusercontent.com/4263537/94693573-2e9a9f00-0334-11eb-97f1-7056506cbc38.png)

exercise draft:
- write transformations (labels to boundaries), dataset + augmentations
- try different architectures: 2d, iso 3d, 3d adapted to aniso (or just resize the data)
- try postprocessing strategies to go from boundaries to segmentation in 2d and 3d

discuss with adrian
- what's the anisotropy factor
- what are good datasets to take?
- what about the unlabeled part / how to treat treat the transitions to unlabled
- what's the degree of anisotropy
- postprocessing strategies?


## Exercise
