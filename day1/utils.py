import os
from imageio import imread
from tqdm import tqdm


def load_cifar(data_dir):
    images = []
    labels = []
    categories = os.listdir(data_dir)
    for label_id, category in tqdm(enumerate(categories), total=len(categories)):
        category_dir = os.path.join(data_dir, category)
        image_names = os.listdir(category_dir)
        for im_name in image_names:
            im_file = os.path.join(category_dir, im_name)
            images.append(imread(im_file))
            labels.append(label_id)
    return images, labels
