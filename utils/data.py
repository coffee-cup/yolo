import os

import numpy as np

from utils import voc_common

ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
}

# (Images, Objects) statistics on every class.
TRAIN_STATISTICS = {
    'none': (0, 0),
    'aeroplane': (670, 865),
    'bicycle': (552, 711),
    'bird': (765, 1119),
    'boat': (508, 850),
    'bottle': (706, 1259),
    'bus': (421, 593),
    'car': (1161, 2017),
    'cat': (1080, 1217),
    'chair': (1119, 2354),
    'cow': (303, 588),
    'diningtable': (538, 609),
    'dog': (1286, 1515),
    'horse': (482, 710),
    'motorbike': (526, 713),
    'person': (4087, 8566),
    'pottedplant': (527, 973),
    'sheep': (325, 813),
    'sofa': (507, 566),
    'train': (544, 628),
    'tvmonitor': (575, 784),
    'total': (11540, 27450),
}

SPLITS_TO_SIZES = {
    'train': 11540,
}

SPLITS_TO_STATISTICS = {
    'train': TRAIN_STATISTICS,
}

NUM_CLASSES = 20


def load_data(record_path, split_name):
    """Function to load data from CIFAR10.

    Parameters
    ----------
    record_path : string
        Path to the tf.Record path containing the data

    split_name : string
        "trainval", "train", "val", or "test"

    Returns
    -------
        A `Dataset` namedtuple.
    """
    dataset = voc_common.get_split(split_name, record_path, SPLITS_TO_SIZES,
                                   ITEMS_TO_DESCRIPTIONS, NUM_CLASSES)

    return dataset
