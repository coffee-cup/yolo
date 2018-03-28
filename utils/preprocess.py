import numpy as np

import tensorflow as tf
from utils import tf_image

slim = tf.contrib.slim

out_shape = (416, 416)


def tf_summary_image(image, bboxes, name='image'):
    """Add image with bounding boxes to sumamry."""
    image = tf.expand_dims(image, 0)
    bboxes = tf.expand_dims(bboxes, 0)
    image_with_box = tf.image.draw_bounding_boxes(image, bboxes)
    tf.summary.image(name, image_with_box)


def preprocess_for_train(image, labels, bboxes, scope='preprocessing_train'):
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

    return image, labels, bboxes


def preprocess_for_validation(image,
                              labels,
                              bboxes,
                              scope='preprocessing_validation'):
    """Preprocesses the given image for validation."""
    with tf.name_scope(scope, [image, labels, bboxes]):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        # Convert to float scaled [0, 1].
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # Resize image to output size.
        image = tf_image.resize_image(
            image,
            out_shape,
            method=tf.image.ResizeMethod.BILINEAR,
            align_corners=False)

    return image, labels, bboxes
