import os
import numpy as np

import torch
from torch.utils.data import Dataset

from imageio import imread
from tqdm import tqdm


#
# helper functions to load and split the data
#

def load_cifar(data_dir):
    images = []
    labels = []

    categories = os.listdir(data_dir)
    categories.sort()

    for label_id, category in tqdm(enumerate(categories), total=len(categories)):
        category_dir = os.path.join(data_dir, category)
        image_names = os.listdir(category_dir)
        for im_name in image_names:
            im_file = os.path.join(category_dir, im_name)
            images.append(np.asarray(imread(im_file)))
            labels.append(label_id)

    # from list of arrays to a single numpy array by stacking along new "batch" axis
    images = np.concatenate([im[None] for im in images], axis=0)
    labels = np.array(labels)

    return images, labels


def make_cifar_train_val_split(images, labels, validation_fraction=0.15):

    # shuffle the data
    n_images = len(images)
    indices = np.arange(n_images)
    np.random.shuffle(indices)

    # split into training and validation data
    validation_fraction = 0.15
    split_index = int(validation_fraction * n_images)
    train_indices = indices[:-split_index]
    val_indices = indices[-split_index:]

    train_images, val_images = images[train_indices], images[val_indices]
    train_labels, val_labels = labels[train_indices], labels[val_indices]
    assert len(train_images) == len(train_labels)
    assert len(val_images) == len(val_labels)
    assert len(train_images) + len(val_images) == n_images

    return train_images, train_labels, val_images, val_labels


#
# transformations and datasets
#

def to_channel_first(image, target):
    """ Transform images with color channel last (WHC) to channel first (CWH)
    """
    # put channel first
    image = image.transpose((2, 0, 1))
    return image, target


def normalize(image, target, channel_wise=True):
    eps = 1.e-6
    image = image.astype('float32')
    chan_min = image.min(axis=(1, 2), keepdims=True)
    image -= chan_min
    chan_max = image.max(axis=(1, 2), keepdims=True)
    image /= (chan_max + eps)
    return image, target


# finally, we need to transform our input from a numpy array to a torch tensor
def to_tensor(image, target):
    return torch.from_numpy(image), torch.tensor([target], dtype=torch.int64)


# we also need a way to compose multiple transformations
def compose(image, target, transforms):
    for trafo in transforms:
        image, target = trafo(image, target)
    return image, target


class DatasetWithTransform(Dataset):
    """ Our minimal dataset class. It holds data and target
    as well as optional transforms that are applied to data and target
    on the fly when data is requested via the [] operator.
    """
    def __init__(self, data, target, transform=None):
        assert isinstance(data, np.ndarray)
        assert isinstance(target, np.ndarray)
        self.data = data
        self.target = target
        if transform is not None:
            assert callable(transform)
        self.transform = transform

    # exposes the [] operator of our class
    def __getitem__(self, index):
        data, target = self.data[index], self.target[index]
        if self.transform is not None:
            data, target = self.transform(data, target)
        return data, target

    def __len__(self):
        return self.data.shape[0]


def make_cifar_datasets(images, labels, transform=None, validation_fraction=0.15):
    (train_images, train_labels,
     val_images, val_labels) = make_cifar_train_val_split(images, labels, validation_fraction)
    train_dataset = DatasetWithTransform(train_images, train_labels, transform=transform)
    val_dataset = DatasetWithTransform(val_images, val_labels, transform=transform)
    return train_dataset, val_dataset


#
# training and validation functions
#

def train(model, loader,
          loss_function, optimizer,
          epoch, log_frequency=10):
    """ Train model for one epoch.
    """

    # set model to train mode
    model.train()

    # iterate over the training batches provided by the loader
    n_batches = len(loader)
    log_frequency_ = n_batches // log_frequency
    for batch_id, (x, y) in enumerate(loader):

        # set the gradients to zero, to start with "clean" gradients
        # in this training iteration
        optimizer.zero_grad()

        # apply the model and get our prediction
        prediction = model(x)

        # calculate the loss (negative log likelihood loss)
        loss_value = loss_function(prediction, y)

        # calculate the gradients (`loss.backward()`)
        # and apply them to the model weights (`optimizer.ste()`)
        loss_value.backward()
        optimizer.step()

        # report the training progress
        if batch_id % log_frequency_ == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_id * len(x), n_batches * len(x),
                  100. * batch_id / n_batches, loss_value.item()))
