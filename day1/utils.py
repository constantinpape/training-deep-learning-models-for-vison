import os
import matplotlib.pyplot as plt
from functools import partial
from itertools import product

import numpy as np
import torch
from torch.utils.data import Dataset
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split

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
    (train_images, val_images,
     train_labels, val_labels) = train_test_split(images, labels, shuffle=True,
                                                  test_size=validation_fraction,
                                                  stratify=labels)
    assert len(train_images) == len(train_labels)
    assert len(val_images) == len(val_labels)
    assert len(train_images) + len(val_images) == len(images)
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


def get_default_cifar_transform():
    trafos = [to_channel_first, normalize, to_tensor]
    trafos = partial(compose, transforms=trafos)
    return trafos


def make_cifar_datasets(cifar_dir, transform=None, validation_fraction=0.15):
    images, labels = load_cifar(os.path.join(cifar_dir, 'train'))
    (train_images, train_labels,
     val_images, val_labels) = make_cifar_train_val_split(images, labels, validation_fraction)

    if transform is None:
        transform = get_default_cifar_transform()

    train_dataset = DatasetWithTransform(train_images, train_labels, transform=transform)
    val_dataset = DatasetWithTransform(val_images, val_labels, transform=transform)
    return train_dataset, val_dataset


def make_cifar_test_dataset(cifar_dir, transform=None):
    images, labels = load_cifar(os.path.join(cifar_dir, 'test'))

    if transform is None:
        transform = get_default_cifar_transform()

    dataset = DatasetWithTransform(images, labels, transform=transform)
    return dataset


#
# training and validation functions
#

def train(model, loader,
          loss_function, optimizer,
          device, epoch,
          tb_logger, log_image_interval=100):
    """ Train model for one epoch.

    Parameters:
    model - the model we are training
    loader - the data loader that provides the training data
        (= pairs of images and labels)
    loss_function - the loss function that will be optimized
    optimizer - the optimizer that is used to update the network parameters
        by backpropagation of the loss
    device - the device used for training. this can either be the cpu or gpu
    epoch - which trainin eppch are we in? we keep track of this for logging
    tb_logger - the tensorboard logger, it is used to communicate with tensorboard
    log_image_interval - how often do we send images to tensborboard?
    """

    # set model to train mode
    model.train()

    # iterate over the training batches provided by the loader
    n_batches = len(loader)
    for batch_id, (x, y) in enumerate(loader):

        # send data and target tensors to the active device
        x = x.to(device)
        y = y.to(device)

        # set the gradients to zero, to start with "clean" gradients
        # in this training iteration
        optimizer.zero_grad()

        # apply the model to get the prediction
        prediction = model(x)

        # calculate the loss (negative log likelihood loss)
        # the target tensor is returned with a singleton axis, i.e.
        # they have the shape (n_batches, 1). However, the loss function
        # expects them to come in shape (n_batches,). That's why we need
        # to index with [:, 0] here.
        loss_value = loss_function(prediction, y[:, 0])

        # calculate the gradients (`loss.backward()`)
        # and apply them to the model parameters according
        # to our optimizer (`optimizer.step()`)
        loss_value.backward()
        optimizer.step()

        # log the loss value to tensorboard
        step = epoch * n_batches + batch_id
        tb_logger.add_scalar(tag='train-loss',
                             scalar_value=loss_value.item(),
                             global_step=step)

        # check if we log images, and if we do then send the
        # current image to tensorboard
        if log_image_interval is not None and step % log_image_interval == 0:
            # TODO make logging more pretty, see
            # https://www.tensorflow.org/tensorboard/image_summaries
            tb_logger.add_images(tag='input',
                                 img_tensor=x.to('cpu'),
                                 global_step=step)


def validate(model, loader, loss_function,
             device, step, tb_logger=None):
    """
    Validate the model predictions.

    Parameters:
    model - the model to be evaluated
    loader - the loader providing images and labels
    loss_function - the loss function
    device - the device used for prediction (cpu or gpu)
    step - the current training step. we need to know this for logging
    tb_logger - the tensorboard logger. if 'None', logging is disabled
    """
    # set the model to eval mode
    model.eval()
    n_batches = len(loader)

    # we record the loss and the predictions / labels for all samples
    mean_loss = 0
    predictions = []
    labels = []

    # the model parameters are not updated during validation,
    # hence we can disable gradients in order to save memory
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)

            # update the loss
            mean_loss += loss_function(prediction, y[:, 0]).item()

            # compute the most likely class predictions
            # note that 'max' returns a tuple with the
            # index of the maximun value (which correponds to the predicted class)
            # as second entry
            prediction = prediction.max(1, keepdim=True)[1]

            # store the predictions and labels
            predictions.append(prediction[:, 0].to('cpu').numpy())
            labels.append(y[:, 0].to('cpu').numpy())

    # predictions and labels to numpy arrays
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)

    # log the validation results if we have a tensorboard
    if tb_logger is not None:

        accuracy_error = 1. - metrics.accuracy_score(labels, predictions)
        mean_loss /= n_batches

        # TODO log more advanced things like confusion matrix, see
        # https://www.tensorflow.org/tensorboard/image_summaries

        tb_logger.add_scalar(tag="validation-error",
                             global_step=step,
                             scalar_value=accuracy_error)
        tb_logger.add_scalar(tag="validation-loss",
                             global_step=step,
                             scalar_value=mean_loss)

    # return all predictions and labels for further evaluation
    return predictions, labels


def make_confusion_matrix(labels, predictions, categories, ax):
    cm = metrics.confusion_matrix(labels, predictions)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Confusion matrix")
    plt.colorbar(im)
    tick_marks = np.arange(len(categories))
    plt.xticks(tick_marks, categories, rotation=45)
    plt.yticks(tick_marks, categories)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
