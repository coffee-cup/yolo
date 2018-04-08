import os

import numpy as np

import tensorflow as tf
import utils.dataset_util

slim = tf.contrib.slim

IMAGE_H, IMAGE_W = 416, 416
GRID_H, GRID_W = 13, 13
BOX = 5
CLASSES = 20
YOLO_ANCHORS = np.array(((0.57273, 0.677385), (1.87446, 2.06253),
                         (3.33843, 5.47434), (7.88282, 3.52778), (9.77052,
                                                                  9.16828)))

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


def preprocess_true_boxes(true_boxes):
    anchors = YOLO_ANCHORS
    height, width = IMAGE_H, IMAGE_W
    num_anchors = len(anchors)

    assert height % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    assert width % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'

    conv_height = height // 32
    conv_width = width // 32

    num_box_params = true_boxes.shape[1]
    detectors_mask = np.zeros(
        (conv_height, conv_width, num_anchors, 1), dtype=np.float32)
    matching_true_boxes = np.zeros(
        (conv_height, conv_width, num_anchors, num_box_params),
        dtype=np.float32)

    for box in true_boxes:
        # scale box to convolutional feature spatial dimensions
        box_class = box[4:5]

        box = box[0:4] * np.array(
            [conv_width, conv_height, conv_width, conv_height])

        i = np.floor(box[1]).astype('int')
        j = np.floor(box[0]).astype('int')

        best_iou = 0
        best_anchor = 0
        for k, anchor in enumerate(anchors):
            # Find IOU between box shifted to origin and anchor box
            box_maxes = box[2:4] / 2.
            box_mins = -box_maxes
            anchor_maxes = (anchor / 2.)
            anchor_mins = -anchor_maxes

            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[0] * intersect_wh[1]
            box_area = box[2] * box[3]
            anchor_area = anchor[0] * anchor[1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)
            if iou > best_iou:
                best_iou = iou
                best_anchor = k

        if best_iou > 0:
            detectors_mask[i, j, best_anchor] = 1
            # adjusted_box = np.array(
            #     [
            #         box[0] - j, box[1] - i,
            #         np.log(box[2] / anchors[best_anchor][0]),
            #         np.log(box[3] / anchors[best_anchor][1]), box_class
            #     ],
            #     dtype=np.float32)
            x = box[0] - j
            y = box[1] - i
            w = box[2] / anchors[best_anchor][0]
            h = box[3] / anchors[best_anchor][1]
            adjusted_box = np.array([x, y, w, h, box_class])

            matching_true_boxes[i, j, best_anchor] = adjusted_box

    return detectors_mask, matching_true_boxes


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
        'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),

        # Bounding box
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/count': tf.FixedLenFeature([], tf.int64),
        'image/object/detectors_mask': tf.VarLenFeature(dtype=tf.float32),
        'image/object/matching_true_boxes': tf.VarLenFeature(dtype=tf.float32)
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
        'object/count':
        slim.tfexample_decoder.Tensor('image/object/count'),
        'object/detectors_mask':
        slim.tfexample_decoder.Tensor('image/object/detectors_mask'),
        'object/matching_true_boxes':
        slim.tfexample_decoder.Tensor('image/object/matching_true_boxes')
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
