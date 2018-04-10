import numpy as np

import tensorflow as tf
from utils import tf_image
from utils.voc_common import (BOX, CLASSES, GRID_H, GRID_W, IMAGE_H, IMAGE_W,
                              YOLO_ANCHORS)

slim = tf.contrib.slim


def process_bboxes_and_labels(bboxes, labels):
    '''
    Convert bboxes in form ymin, xmin, ymax xmax
    to xcenter, ycenter, width, height, class label
    '''
    # boxes are in the form ymin, xmin, ymax, xmax
    # we want xcenter, ycenter, width, height, class
    ymin = tf.reshape(bboxes[:, 0], (-1, 1))
    xmin = tf.reshape(bboxes[:, 1], (-1, 1))
    ymax = tf.reshape(bboxes[:, 2], (-1, 1))
    xmax = tf.reshape(bboxes[:, 3], (-1, 1))

    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    width = xmax - xmin
    height = ymax - ymin

    # Scale to grid
    # center_x *= GRID_W
    # center_y *= GRID_H
    # width *= GRID_W
    # height *= GRID_H

    labels = tf.cast(tf.reshape(labels, (-1, 1)), tf.float32)
    bboxes_labels = tf.concat([center_x, center_y, width, height, labels], 1)

    return bboxes_labels


def tf_summary_image(image, bboxes, name='image'):
    """Add image with bounding boxes to sumamry."""
    image = tf.expand_dims(image, 0)
    bboxes = tf.expand_dims(bboxes, 0)
    image_with_box = tf.image.draw_bounding_boxes(image, bboxes)
    tf.summary.image(name, image_with_box)


def preprocess_for_train(image,
                         labels,
                         bboxes,
                         y_true,
                         scope='preprocessing_train'):
    """Preprocesses the given image for training.
    Note that the actual resizing scale is sampled from
        [`resize_size_min`, `resize_size_max`].
    Args:
        image: A `Tensor` representing an image of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        resize_side_min: The lower bound for the smallest side of the image for
            aspect-preserving resizing.
        resize_side_max: The upper bound for the smallest side of the image for
            aspect-preserving resizing.
    Returns:
        A preprocessed image.
    """
    with tf.name_scope(scope, [image, labels, bboxes]):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        # Convert to float scaled [0, 1].
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        tf_summary_image(image, bboxes, 'image_with_bboxes')

        # Distort image and bounding boxes.
        # image = image
        # image, labels, bboxes, distort_bbox = \
        #     distorted_bounding_box_crop(image, labels, bboxes,
        #                                 min_object_covered=MIN_OBJECT_COVERED,
        #                                 aspect_ratio_range=CROP_RATIO_RANGE)

        # Resize image to output size.
        out_shape = (IMAGE_H, IMAGE_W)
        image = tf_image.resize_image(
            image,
            out_shape,
            method=tf.image.ResizeMethod.BILINEAR,
            align_corners=False)
        tf_summary_image(image, bboxes, 'image_resized')

        # Randomly flip the image horizontally.
        image, bboxes = tf_image.random_flip_left_right(image, bboxes)

        # Randomly distort the colors. There are 4 ways to do it.
        # image = apply_with_random_selector(
        #     image,
        #     lambda x, ordering: distort_color(x, ordering, fast_mode),
        #     num_cases=4)
        # tf_summary_image(image, bboxes, 'image_color_distorted')

        # Normalize to [-1, 1]
        # image = tf.multiply(image, 1. / 127.5)
        # image = tf.subtract(image, 1.0)

    bboxes_labels = process_bboxes_and_labels(bboxes, labels)

    # Reshape boxes and true output
    num_anchors = len(YOLO_ANCHORS)
    y_true = tf.reshape(y_true, (GRID_H, GRID_W, num_anchors, 4 + 1 + CLASSES))

    return image, bboxes_labels, y_true


def preprocess_for_validation(image,
                              labels,
                              bboxes,
                              y_true,
                              scope='preprocessing_val'):
    """Preprocesses the given image for validation."""
    with tf.name_scope(scope, [image, labels, bboxes]):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        # Convert to float scaled [0, 1].
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # Resize image to output size.
        out_shape = (IMAGE_H, IMAGE_W)
        image = tf_image.resize_image(
            image,
            out_shape,
            method=tf.image.ResizeMethod.BILINEAR,
            align_corners=False)

    bboxes_labels = process_bboxes_and_labels(bboxes, labels)

    # Reshape boxes and true output
    num_anchors = len(YOLO_ANCHORS)
    y_true = tf.reshape(y_true, (GRID_H, GRID_W, num_anchors, 4 + 1 + CLASSES))

    return image, bboxes_labels, y_true
