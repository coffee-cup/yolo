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
import os

import PIL.Image
from lxml import etree

import tensorflow as tf
from config import get_config, print_usage
from tqdm import trange
from utils import dataset_util, voc_common
from utils.dataset_util import (bytes_feature, bytes_list_feature,
                                float_list_feature, int64_feature,
                                int64_list_feature)

slim = tf.contrib.slim

VOC_LABELS = voc_common.VOC_LABELS


def _dict_to_tf_example(data,
                        dataset_directory,
                        image_subdirectory='JPEGImages'):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
        data: dict holding PASCAL XML fields for a single image (obtained by
        running recursive_parse_xml_to_dict)
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
    depth = int(data['size']['depth'])
    shape = [height, width, depth]

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
        difficult_b = bool(int(obj['difficult']))

        # Ignore difficult objects for now
        if difficult_b:
            continue

        # Difficulty
        difficult.append(int(difficult_b))

        if obj.get('truncated'):
            truncated.append(int(obj.get('truncated')))
        else:
            truncated.append(0)

        # Normalized bounding boxes
        xmin.append(float(obj['bndbox']['xmin']) / width)
        ymin.append(float(obj['bndbox']['ymin']) / height)
        xmax.append(float(obj['bndbox']['xmax']) / width)
        ymax.append(float(obj['bndbox']['ymax']) / height)
        object_count = len(xmin)

        # Classes
        classes.append(int(VOC_LABELS[obj['name']][0]))
        classes_text.append(obj['name'].encode('utf8'))

    features = {
        # Image file
        'image/filename': bytes_feature(data['filename'].encode('utf8')),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature('jpeg'.encode('utf8')),
        'image/key/sha256': bytes_feature(key.encode('utf8')),
        'image/source_id': bytes_feature(data['filename'].encode('utf8')),

        # Image features
        'image/width': int64_feature(width),
        'image/height': int64_feature(height),
        'image/channels': int64_feature(depth),
        'image/shape': int64_list_feature(shape),

        # Detection features
        'image/object/difficult': int64_list_feature(difficult),
        'image/object/truncated': int64_list_feature(truncated),

        # Classes
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),

        # Bounding box
        'image/object/bbox/xmin': float_list_feature(xmin),
        'image/object/bbox/xmax': float_list_feature(xmax),
        'image/object/bbox/ymin': float_list_feature(ymin),
        'image/object/bbox/ymax': float_list_feature(ymax),
        'image/object/count': int64_feature(object_count)
    }

    return tf.train.Example(features=tf.train.Features(feature=features))


def create_record_file(data_dir, output_file, year, split_name):
    years = ['VOC2007', 'VOC2012']
    if year != 'merged':
        years = [year]

    # Create tf.Record writer
    writer = tf.python_io.TFRecordWriter(output_file)

    for year in years:
        print('Creating TFRecord file from PASCAL {} {} dataset'.format(
            year, split_name))

        examples_path = os.path.join(data_dir, year, 'ImageSets', 'Main',
                                     split_name + '.txt')
        annotations_dir = os.path.join(data_dir, year, 'Annotations')
        examples_list = dataset_util.read_examples_list(examples_path)

        for idx in trange(0, len(examples_list)):
            example = examples_list[idx]

            # Find and parse annotation xml file
            path = os.path.join(annotations_dir, example + '.xml')
            with tf.gfile.GFile(path, 'r') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

            # Create tf.Example and add to tf.Record
            tf_example = _dict_to_tf_example(data, data_dir)
            writer.write(tf_example.SerializeToString())

    writer.close()
    print('Saved tf Record to {}\n'.format(output_file))


if __name__ == '__main__':
    config, unparsed = get_config()

    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)
