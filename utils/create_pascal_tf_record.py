# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=/home/user/VOCdevkit \
        --year=VOC2012 \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import, division, print_function

import hashlib
import io
import logging
import os

import PIL.Image
from lxml import etree

import dataset_util
import tensorflow as tf
from config import get_config, print_usage

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


def dict_to_tf_example(data,
                       dataset_directory,
                       image_subdirectory='JPEGImages'):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
        data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
        dataset_directory: Path to root directory holding PASCAL dataset

        image_subdirectory: String specifying subdirectory within the
        PASCAL dataset directory holding the actual image data.

    Returns:
        example: The converted tf.Example.

    Raises:
        ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """

    # Get full image path
    img_path = os.path.join(data['folder'], image_subdirectory,
                            data['filename'])
    full_path = os.path.join(dataset_directory, img_path)

    # Encode jpg image
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    difficult = []
    truncated = []

    # For each detection in the image
    for obj in data['object']:

        # Difficulty
        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)

        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        # Normalized bounding boxes
        xmin.append(float(obj['bndbox']['xmin']) / width)
        ymin.append(float(obj['bndbox']['ymin']) / height)
        xmax.append(float(obj['bndbox']['xmax']) / width)
        ymax.append(float(obj['bndbox']['ymax']) / height)

        # Classes
        label = obj.find('name').text
        classes.append(int(VOC_LABELS[label][0]))
        classes_text.append(label)

        classes_text.append(obj['name'].encode('utf8'))
        classes.append(class_text_to_int(obj['name']))
        truncated.append(int(obj['truncated']))

    print('-- Adding class {}'.format(classes_text))

    features = {
        'image/filename':
        dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/encoded':
        dataset_util.bytes_feature(encoded_jpg),
        'image/format':
        dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/width':
        dataset_util.int64_feature(width),
        'image/height':
        dataset_util.int64_feature(height),
        'image/key/sha256':
        dataset_util.bytes_feature(key.encode('utf8')),
        'image/source_id':
        dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/object/difficult':
        dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated':
        dataset_util.int64_list_feature(truncated),
        'image/object/class/text':
        dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label':
        dataset_util.int64_list_feature(classes),

        # Bounding box
        'image/object/bbox/xmin':
        dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax':
        dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin':
        dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax':
        dataset_util.float_list_feature(ymax),
    }

    return tf.train.Example(features=tf.train.Features(feature=features))


def main(config):
    years = ['VOC2007', 'VOC2012']
    if config.year != 'merged':
        years = [config.year]

    writer = tf.python_io.TFRecordWriter(config.output_path)

    for year in years:
        logging.info('Reading from PASCAL %s dataset.', year)
        examples_path = os.path.join(config.data_dir, year, 'ImageSets',
                                     'Main', config.set + '.txt')
        annotations_dir = os.path.join(data_dir, year, config.annotations_dir)
        examples_list = dataset_util.read_examples_list(examples_path)
        for idx, example in enumerate(examples_list):
            if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(examples_list))
            path = os.path.join(annotations_dir, example + '.xml')
            with tf.gfile.GFile(path, 'r') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

            tf_example = dict_to_tf_example(data, config.data_dir)
            writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    config, unparsed = get_config()

    if len(unparsed) > 0:
        print_usage()
        exit(1)
