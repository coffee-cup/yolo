import os

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils.preprocess import preprocess_for_train


class Yolo(object):
    """Yolo network"""

    def __init__(self, config):
        self.config = config

        self._build_placeholder()
        self._build_preprocessing()
        self._build_model()
        self._build_loss()
        self._build_optim()
        self._build_eval()
        self._build_summary()
        self._build_writer()

    def _build_placeholder(self):
        """Build placeholders."""
        pass

    def _build_preprocessing(self):
        """Build preprocessing related graph."""
        pass

    def _build_model(self):
        """ Arguments required for darknet :
            net, classes, num_anchors, training=False, center=True"""
        pass

    def _build_loss(self):
        """Build our loss."""
        pass

    def _build_optim(self):
        """Build optimizer related ops and vars."""
        pass

    def _build_eval(self):
        """Build the evaluation related ops"""
        pass

    def _build_summary(self):
        """Build summary ops."""
        self.summary_op = tf.summary.merge_all()

    def _build_writer(self):
        """Build the writers and savers"""
        # Create summary writers (one for train, one for validation)
        self.summary_tr = tf.summary.FileWriter(
            os.path.join(self.config.log_dir, "train"))
        self.summary_va = tf.summary.FileWriter(
            os.path.join(self.config.log_dir, "valid"))

        # Create savers (one for current, one for best)
        # self.saver_cur = tf.train.Saver()
        # self.saver_best = tf.train.Saver()

        # # Save file for the current model
        # self.save_file_cur = os.path.join(self.config.log_dir, "model")

        # # Save file for the best model
        # self.save_file_best = os.path.join(self.config.save_dir, "model")

    def train(self, dataset_train):
        print('\n--- Training')

        with tf.Graph().as_default():
            # Create dataset provider
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset_train, num_readers=1, shuffle=True)

            [images, shape, labels, bboxes] = provider.get(
                ['image', 'shape', 'object/label', 'object/bbox'])

            # Preprocess
            images, labels, bboxes = preprocess_for_train(
                images, labels, bboxes)

            # Need to rebuild summary
            self._build_summary()

            # Create batche
            # batch = tf.train.batch(
            #     [images, labels, bboxes],
            #     batch_size=self.config.batch_size,
            #     num_threads=1,
            #     capactiy=4 * config.batch_size,
            #     allow_smaller_final_batch=True)

            # Run TensorFlow Session
            with tf.Session() as sess:
                print('Initializing...')
                sess.run([
                    tf.local_variables_initializer(),
                    tf.global_variables_initializer()
                ])

                print(self.summary_op)
                s = sess.run(self.summary_op)
                self.summary_tr.add_summary(s)
                self.summary_tr.flush()
