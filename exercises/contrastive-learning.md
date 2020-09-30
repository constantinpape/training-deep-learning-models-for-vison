# Contrastive learning for visual representations

One of the largest obstacles to training neural networks for new tasks in computer vision
is their need for large amounts of labeled training data.
Manually labeling datasets is a very time consuming and tedious task, so generating
new labels is tedious or (if it is outsourced) expensive.

However, collecting the raw image is usually much easier, so large amounts of image data without labels are often present. The goal of unsupervised representation learning is to use this unlabeled data to 
learn a descriptive representation of the data that can be used to fine-tune for the actual
task given only a small amount of labeled data.

Here, we will implement a method for unsupervised representation learning based on the recent [SimCLR paper](https://arxiv.org/abs/2002.05709).


## SimCLR A simple Framework for Contrastive Learning of Visual Representations


For more details, [check out the paper](https://arxiv.org/abs/2002.05709).


## Exercise

While the ImageNet results in the SimCLR paper require a lot of computation, the method itself
is fairly simple, and can be applied to smaller datasets with much less resources.
For this exercise, implement SimCLR:
- Choose a small image dataset such as [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) or [STL10](https://cs.stanford.edu/~acoates/stl10/). 
- Implement the unsupervised training procedure, where you sample positive pairs by applying a succession of transformations to the same image and compute the contrastive loss.
- Train a linear model or small MLP using the learned representations as input. Only use a small fraction of the labeled data for supervison and do not update the representation during traing.
- Compare the results to a supervised model learned on the fully labeled dataset.

After implementing the basic SimCLR framework, you can explore how the following settings influence the quality of the learned representations:
- the fraction of labeled data used for the small classification model
- the number of epochs used for training
- the transformations used for generating positive pairs
- the batch size used during training

You can also try to visualize the learned representation using a dimensionalty reduction method like T-SNE or UMAP.

Hints:
Torchvision already provides the necessary [transforms](https://pytorch.org/docs/stable/torchvision/transforms.html) and many [datasets](https://pytorch.org/docs/stable/torchvision/datasets.html).
You can freeze a model (i.e. exclude its parameters during training) by recursively setting
`requires_grad=False` for its parameters:
```python
for param in model.parameters():
    param.requires_grad = False
```
When you instantiate the optimizer, you also need to exclude these parameters:
```python
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
```
For the loss function, you can use [cosine similarity](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.cosine_similarity) to compute the similarity between the pairs of transformed images.
