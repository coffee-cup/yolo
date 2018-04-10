import os

import numpy as np
from PIL import Image

import cv2
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

labels_to_names = {}
for k, v in VOC_LABELS.items():
    labels_to_names[v[0]] = k


def preprocess_true_boxes(true_boxes):
    '''true_boxes is list of bounding boxes in form [x, y, w, h]'''
    anchors = YOLO_ANCHORS
    height, width = IMAGE_H, IMAGE_W
    num_anchors = len(anchors)
    num_boxes = len(true_boxes)

    y_true = np.zeros(
        (GRID_H, GRID_W, num_anchors, 4 + 1 + CLASSES), dtype=np.float32)

    for box_index, box in enumerate(true_boxes):
        box_class = int(box[4:5])

        # Scale box by grid
        box[0] *= GRID_W  # center x
        box[1] *= GRID_H  # center y
        box[2] *= GRID_W  # width
        box[3] *= GRID_H  # height
        box = box[0:4]

        grid_x = np.floor(box[1]).astype('int')
        grid_y = np.floor(box[0]).astype('int')

        best_iou = -1
        best_anchor = -1
        shifted_box = np.array([0, 0, box[2], box[3]])

        for k, anchor in enumerate(anchors):
            # Find IOU between box shifted to origin and anchor box
            box_maxes = shifted_box[2:4] / 2.
            box_mins = -box_maxes
            anchor_maxes = (anchor / 2.)
            anchor_mins = -anchor_maxes

            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[0] * intersect_wh[1]
            box_area = shifted_box[2] * shifted_box[3]
            anchor_area = anchor[0] * anchor[1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)
            if iou > best_iou:
                best_iou = iou
                best_anchor = k

        if best_iou > 0:
            box = np.array([
                box[0] - grid_y, box[1] - grid_x,
                np.log(box[2] / anchors[best_anchor][0]),
                np.log(box[3] / anchors[best_anchor][1])
            ])
            y_true[grid_y, grid_x, best_anchor, 0:4] = box
            y_true[grid_y, grid_x, best_anchor, 4] = 1
            y_true[grid_y, grid_x, best_anchor, 5 + box_class - 1] = 1

    return y_true


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
        'image/object/y_true': tf.VarLenFeature(dtype=tf.float32)
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
        'object/y_true':
        slim.tfexample_decoder.Tensor('image/object/y_true')
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                      items_to_handlers)

    return slim.dataset.Dataset(
        data_sources=record_path,
        reader=reader,
        decoder=decoder,
        num_samples=split_to_sizes[split_name],
        items_to_descriptions=items_to_descriptions,
        num_classes=num_classes,
        labels_to_names=labels_to_names)


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, confidence=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.confidence = confidence
        self.classes = classes

        self.label = 0
        if classes is not None:
            self.label = np.argmax(self.classes) + 1
            self.score = self.classes[self.label - 1]

    def obj_name(self):
        return labels_to_names[self.label]

    def coord_array(self):
        return np.array([self.xmin, self.ymin, self.xmax, self.ymax])

    def __str__(self):
        return str(self.coord_array()) + ' ' + self.obj_name()


def draw_boxes(image, boxes):
    image = (image * 255 / np.max(image)).astype('uint8')
    image_h, image_w, _ = image.shape

    for box in boxes:
        xmin = int(box.xmin * image_w)
        ymin = int(box.ymin * image_h)
        xmax = int(box.xmax * image_w)
        ymax = int(box.ymax * image_h)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(
            image,
            box.obj_name() + ' ' + str(box.score), (xmin, ymin - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            1e-3 * image_h, (0, 255, 0),
            2,
            lineType=cv2.LINE_AA)

    return image


def combine_images(images):
    '''Draw 2 images side by side.'''
    images = [Image.fromarray(i) for i in images]
    widths = [i.size[0] for i in images]
    heights = [i.size[1] for i in images]

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    return new_im


def save_image(image, filename):
    '''Save a numpy array (height, width, 3) between [0, 1] as an image to filename'''
    # formatted = (image * 255 / np.max(image)).astype('uint8')
    im = Image.fromarray(image)
    im.save(filename)


def decode_netout(netout, obj_threshold=0.3):
    boxes = []

    num_anchors = len(YOLO_ANCHORS)

    # netout[..., 4] = sigmoid(netout[..., 4])
    # netout[..., 5:] = netout[..., 4][..., np.newaxis] * softmax(
    #     netout[..., 5:])
    # netout[..., 5:] *= netout[..., 5:] > obj_threshold

    for row in range(GRID_H):
        for col in range(GRID_W):
            for b in range(num_anchors):
                classes = netout[row, col, b, 5:]

                if np.sum(classes) > 0:
                    x, y, w, h = netout[row, col, b, :4]

                    x = (sigmoid(x) + row) / GRID_W
                    y = (sigmoid(y) + col) / GRID_H
                    w = YOLO_ANCHORS[b][0] * np.exp(w) / GRID_W
                    h = YOLO_ANCHORS[b][1] * np.exp(h) / GRID_H
                    confidence = netout[row, col, b, 4]

                    # Only include box if class label is not 0 (background)
                    box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2,
                                   confidence, classes)
                    boxes.append(box)

    boxes = [box for box in boxes if box.score > obj_threshold]
    return boxes


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)

    if np.min(x) < t:
        x = x / np.min(x) * t

    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)
