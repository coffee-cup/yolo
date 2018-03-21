import os

import tensorflow as tf
import utils.dataset_util

slim = tf.contrib.slim

VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}


def get_split(split_name, record_path, split_to_sizes, items_to_descriptions,
              num_classes):
    """Gets a dataset tuple with instructions for reading Pascal VOC dataset.

    Args:
      split_name: A train/test split name.
      record_path: The tf.Record file to get the data from

    Returns:
      A `Dataset` namedtuple.
    """

    reader = tf.TFRecordReader

    # Features in Pascal VOC TFRecords.
    keys_to_features = {
        # Image file
        'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature(
            (), tf.string, default_value='jpeg'),
        'image/key/sha256': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/source_id': tf.FixedLenFeature((), tf.string, default_value=''),

        # Image features
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),

        # Detection features
        'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),

        # Classes
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),

        # Bounding box
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
    }

    items_to_handlers = {
        'image':
        slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'shape':
        slim.tfexample_decoder.Tensor('image/shape'),
        'object/bbox':
        slim.tfexample_decoder.BoundingBox(['ymin', 'xmin', 'ymax', 'xmax'],
                                           'image/object/bbox/'),
        'object/label':
        slim.tfexample_decoder.Tensor('image/object/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                      items_to_handlers)

    labels_to_names = {}
    for k, v in VOC_LABELS.items():
        labels_to_names[v[0]] = k

    return slim.dataset.Dataset(
        data_sources=record_path,
        reader=reader,
        decoder=decoder,
        num_samples=split_to_sizes[split_name],
        items_to_descriptions=items_to_descriptions,
        num_classes=num_classes,
        labels_to_names=labels_to_names)
