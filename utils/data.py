import os

import numpy as np

import tensorflow as tf
import voc_utils
from utils.external import unpickle

slim = tf.contrib.slim

# feature={
#     'image/height':
#     dataset_util.int64_feature(height),
#     'image/width':
#     dataset_util.int64_feature(width),
#     'image/filename':
#     dataset_util.bytes_feature(data['filename'].encode('utf8')),
#     'image/source_id':
#     dataset_util.bytes_feature(data['filename'].encode('utf8')),
#     'image/key/sha256':
#     dataset_util.bytes_feature(key.encode('utf8')),
#     'image/encoded':
#     dataset_util.bytes_feature(encoded_jpg),
#     'image/format':
#     dataset_util.bytes_feature('jpeg'.encode('utf8')),
#     'image/object/bbox/xmin':
#     dataset_util.float_list_feature(xmin),
#     'image/object/bbox/xmax':
#     dataset_util.float_list_feature(xmax),
#     'image/object/bbox/ymin':
#     dataset_util.float_list_feature(ymin),
#     'image/object/bbox/ymax':
#     dataset_util.float_list_feature(ymax),
#     'image/object/class/text':
#     dataset_util.bytes_list_feature(classes_text),
#     'image/object/class/label':
#     dataset_util.int64_list_feature(classes),
#     'image/object/difficult':
#     dataset_util.int64_list_feature(difficult_obj),
#     'image/object/truncated':
#     dataset_util.int64_list_feature(truncated),
#     'image/object/view':
#     dataset_util.bytes_list_feature(poses),
# }))


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    eaxmple = tf.parse_single_example(
        serialized_example,
        features={
            'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
            'image/filename':
            tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/height':
            tf.FixedLenFeature([1], tf.int64),
            'image/width':
            tf.FixedLenFeature([1], tf.int64),
            'image/channels':
            tf.FixedLenFeature([1], tf.int64),
            'image/shape':
            tf.FixedLenFeature([3], tf.int64),
            'image/object/bbox/xmin':
            tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin':
            tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax':
            tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax':
            tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/label':
            tf.VarLenFeature(dtype=tf.int64),
            'image/object/bbox/difficult':
            tf.VarLenFeature(dtype=tf.int64),
            'image/object/bbox/truncated':
            tf.VarLenFeature(dtype=tf.int64),
        })

    with tf.name_scope('load_image'):
        imagefile = tf.read_file(example['image/filename'])
        image = tf.image.decode_jpeg(imagefile, channels=3)

    label = tf.cast(example['image/object/class/label'])

    return image, label


def inputs(train, record_file, batch_size, num_epochs):
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=num_epochs)

        image, label = read_and_decode(filename_queue)


def get_data(record_file):
    reader = tf.TFRecordReader
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature(
            (), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
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
        slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        'object/difficult':
        slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
        'object/truncated':
        slim.tfexample_decoder.Tensor('image/object/bbox/truncated'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                      items_to_handlers)

    labels_to_names = t


def load_data(data_dir, data_type):
    """Function to load data from CIFAR10.

    Parameters
    ----------
    data_dir : string
        Absolute path to the directory containing the extracted CIFAR10 files.

    data_type : string
        Either "train" or "test", which loads the entire train/test data in
        concatenated form.

    Returns
    -------
    data : ndarray (uint8)
        Data from the CIFAR10 dataset corresponding to the train/test
        split. The datata should be in NHWC format.

    labels : ndarray (int)
        Labels for each data. Integers ranging between 0 and 9.

    """

    voc = voc_utils.PascalVOC(r'./')
    annotations = voc.get_annotations(dataset=data_type)

    data = []
    label = []

    for row in annotations:
        print(row)
        break

    # if data_type == "train":
    #     data = []
    #     label = []
    #     for _i in range(5):
    #         file_name = os.path.join(data_dir, "data_batch_{}".format(_i + 1))
    #         cur_dict = unpickle(file_name)
    #         data += [np.array(cur_dict[b"data"])]
    #         label += [np.array(cur_dict[b"labels"])]
    #     # Concat them
    #     data = np.concatenate(data)
    #     label = np.concatenate(label)

    # elif data_type == "test":
    #     data = []
    #     label = []
    #     cur_dict = unpickle(os.path.join(data_dir, "test_batch"))
    #     data = np.array(cur_dict[b"data"])
    #     label = np.array(cur_dict[b"labels"])

    # else:
    #     raise ValueError("Wrong data type {}".format(data_type))

    # N=number of images, H=height, W=widht, C=channels. Note that this
    # corresponds to Tensorflow format that we will use later.
    # data = np.transpose(np.reshape(data, (-1, 3, 32, 32)), (0, 2, 3, 1))

    return data, label
