# Contrastive learning for visual representations

One of the largest obstacles to training neural networks for new tasks in computer vision
is their need for large amounts of labeled training data.
Manually labeling of datasets is a very time consuming task, so generating new labels is tedious or (if it is outsourced) expensive.

However, collecting the raw image data is usually much easier, so large amounts of image data without labels can easily be generated. 
The goal of unsupervised representation learning is to use this unlabeled data to 
learn a descriptive representation of the data that can be fine-tuned for the actual task given only a small amount of labeled data.

Here, we will implement a method for unsupervised representation learning based on the recent [SimCLR paper](https://arxiv.org/abs/2002.05709).


## SimCLR A simple Framework for Contrastive Learning of Visual Representations

The SimCLR method works by minimizing a contrastive loss function between pairs of images.
The idea is to compute the loss in such a way that related images are closer in the representation space
than unrelated images. In a given training step, this is achieved by (see also image below):
- applying two randomly parametrized transformations (drawn from the same distribution of transformations) `t`,`t'` to the input image
- appplying the (learnable) function `f` to the two transformed images to obtain the representations `h_i`, `h_j`.
- applying a small learnable projection function `g` to the representations to obtain `z_i`, `z_j`.
- maximizing the agreement between `z_i` and `z_j` using the contrastive loss function.

![overview](https://user-images.githubusercontent.com/4263537/94680396-f8ecba80-0321-11eb-81db-61874fc87c38.png)

SimCLR only computes the loss explicitly between positive pairs, i.e. the pairs of images obtained by applying two differently parametrized transformations to the same input image. Given a minibatch of N images (2N images after transformations), for a given positive pair `ij` all other  2 (N - 1) transformed images are treated as negative examples.
This is formalized by the normalized temperature-scaled cross entropy, or NT-Xent, loss:

![loss](https://user-images.githubusercontent.com/4263537/94681766-410cdc80-0324-11eb-87b7-08b854a83fa5.png)

where `sim(z_i, z_j)` is the cosine simjarity of projection `z_i` and `z_j` and `tau` is a temperature parameter.

The paper examines several choices for the transformations and finds the following combination to work the best:
- taking a random crop of the image and resizing it back to the original size
- randomly applying a horizontal flip
- applying random color jittering 
- blurring the image with a randomly sampled sigma factor

It uses a Resnet50  (we will use a Resnet18 to lower the training effort) as function `f` and uses the output after the average pooling layer as representation, discarding the final fully connected layer. 
For the  projection `g` a small network consisting of 2 fully connected layers with an intermediate ReLU activation is used. Empirically, computing the loss on the projection `z` instead of the representation `h` yields significantly better representations; the size of the projection 
space does not have a large influence, for example 256 is a good choice.

Overall, this algorithm summarizes the SimCLR training:

![training](https://user-images.githubusercontent.com/4263537/94684528-774c5b00-0328-11eb-9afd-ef9bc270bf28.png)

After the representation has been trained, it can for exampled be used to train a small MLP or linear model on a fraction of the
labeled training data. This model should yield a fairly high accuracy, comparable to larger models trained fully supervised on the whole training data.
Note that this network should be trained on the representation `h`, not the projection `z`.

In addition, the paper performs a set of experiments that compare different parameter choices and show that the
proposed method can yield very accurate results on ImageNet when using only a small fraction of the labeled training data.
For more details, [check out the paper](https://arxiv.org/abs/2002.05709).

## Exercise

While the ImageNet results in the SimCLR paper require a lot of computation, the method itself
is fairly simple, and can be applied to smaller datasets with much less resources.
For this exercise:
- Choose a small image dataset such as [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) or [STL10](https://cs.stanford.edu/~acoates/stl10/). 
- Implement the unsupervised training procedure, where you sample positive pairs by applying a succession of transformations to the same image and compute the contrastive loss. Use it to train a Resnet18 to output a visual representation space.
- Train a linear model or small MLP using the learned representations as input. Only use a small fraction of the labeled data for supervison and do not update the representation during traing.
- Compare the results to a supervised model learned on the fully labeled dataset.

After implementing the basic SimCLR framework, you can explore how the following settings influence the quality of the learned representations:
- the fraction of labeled data used for the small classification model
- the number of epochs used for training
- the transformations used for generating positive pairs
- the batch size used during training

You can also try to visualize the learned representation using a dimensionalty reduction method like T-SNE or UMAP.

**Hints**:

Torchvision already provides the necessary [transforms](https://pytorch.org/docs/stable/torchvision/transforms.html) and many [datasets](https://pytorch.org/docs/stable/torchvision/datasets.html).

You can freeze a model (i.e. exclude its parameters during training) by setting `requires_grad=False` for its parameters:
```python
for param in model.parameters():
    param.requires_grad = False
```
When you instantiate the optimizer, you also need to exclude these parameters:
```python
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

```
For the loss function, you can use the [cosine similarity](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.cosine_similarity) to compute the similarity between the pairs of transformed images.
