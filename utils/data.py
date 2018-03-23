import os

import numpy as np

from utils import pascal_tf_record, voc_common

ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
}

SPLITS_TO_SIZES = {
    'train': 5717,
    'val': 5823,
    'trainval': 11540,
}

NUM_CLASSES = 20


def tf_record_exist(record_file):
    """Returns whether or not the tf record file exists"""
    return os.path.exists(record_file)


def load_data(data_dir, record_file, year, split_name):
    """Function to load data from CIFAR10.

    Parameters
    ----------
    data_dir : string
        Path to directory containing pascal voc data

    record_file : string
        Path to the tf.record path containing the data

    year : string
        Year we want from the pascal voc data (2012)

    annotations_dir : string
         Relative directory to data_dir of image annotation xml files

    split_name : string
        "trainval", "train", "val", or "test"

    Returns
    -------
        A `Dataset` namedtuple.
    """

    record_file = record_file.format(split_name)

    # If record file does not exists, create it
    if not tf_record_exist(record_file):
        pascal_tf_record.create_record_file(data_dir, record_file, year,
                                            split_name)

    return voc_common.get_split(split_name, record_file, SPLITS_TO_SIZES,
                                ITEMS_TO_DESCRIPTIONS, NUM_CLASSES)
