import os

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tqdm import trange
from utils.preprocess import preprocess_for_train


class Yolo(object):
    """Yolo network"""

    def __init__(self, config, dataset_train, dataset_val):
        self.config = config

        # load dataset provider
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset_train, num_readers=1, shuffle=True)

        [image, shape, labels, bboxes, object_count] = provider.get([
            'image', 'shape', 'object/label', 'object/bbox', 'object/count'
        ])

        # Preprocess
        image, labeles, bboxes = preprocess_for_train(
            image, labels, bboxes)

        print('image {}'.format(image))
        print('labels {}'.format(labels))
        print('bboxes {}'.format(bboxes))

        # Need to rebuild summary
        self._build_summary()

        # Create batches
        batch_size = self.config.batch_size
        batch = tf.train.batch(
            [image, labels, bboxes],
            batch_size=batch_size,
            num_threads=1,
            capacity=1 * batch_size,
            dynamic_pad=True,
            allow_smaller_final_batch=True)

        self.image_in = batch[0]
        self.label_in = batch[1]
        self.bboxes_in = batch[2]


        self._build_placeholder()
        self._build_preprocessing()
        self.model = self._build_model()
        self.loss = self._build_loss()
        self.optimizer = self._build_optim()
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
        with tf.variable_scope("Network", reuse=tf.AUTO_REUSE):
            n_filters = 32

            cur_in = tf.layers.conv2d(self.image_in,n_filters,7,1,padding="same",activation=tf.nn.relu)
            cur_in = tf.layers.max_pooling2d(cur_in,(2,2),2,padding="same")

            n_filters = n_filters*2
            cur_in = tf.layers.conv2d(cur_in,n_filters,3,1,padding="same",activation=tf.nn.relu)
            cur_in = tf.layers.max_pooling2d(cur_in,(2,2),2,padding="same")

            n_filters = n_filters*2
            cur_in = tf.layers.conv2d(cur_in,n_filters,3,1,padding="same",activation=tf.nn.relu)
            cur_in = tf.layers.conv2d(cur_in,n_filters/2.0,1,1,padding="same",activation=tf.nn.relu)
            cur_in = tf.layers.conv2d(cur_in,n_filters,3,1,padding="same",activation=tf.nn.relu)
            cur_in = tf.layers.max_pooling2d(cur_in,(2,2),2,padding="same")

            n_filters = n_filters*2
            cur_in = tf.layers.conv2d(cur_in,n_filters,3,1,padding="same",activation=tf.nn.relu)
            cur_in = tf.layers.conv2d(cur_in,n_filters/2.0,1,1,padding="same",activation=tf.nn.relu)
            cur_in = tf.layers.conv2d(cur_in,n_filters,3,1,padding="same",activation=tf.nn.relu)
            cur_in = tf.layers.max_pooling2d(cur_in,(2,2),2,padding="same")

            n_filters = n_filters*2
            cur_in = tf.layers.conv2d(cur_in,n_filters,3,1,padding="same",activation=tf.nn.relu)
            cur_in = tf.layers.conv2d(cur_in,n_filters/2.0,1,1,padding="same",activation=tf.nn.relu)
            cur_in = tf.layers.conv2d(cur_in,n_filters,3,1,padding="same",activation=tf.nn.relu)
            cur_in = tf.layers.conv2d(cur_in,n_filters/2.0,1,1,padding="same",activation=tf.nn.relu)
            cur_in = tf.layers.conv2d(cur_in,n_filters,3,1,padding="same",activation=tf.nn.relu)
            cur_in = tf.layers.max_pooling2d(cur_in,(2,2),2,padding="same")

            n_filters = n_filters*2
            cur_in = tf.layers.conv2d(cur_in,n_filters,3,1,padding="same",activation=tf.nn.relu)
            cur_in = tf.layers.conv2d(cur_in,n_filters/2.0,1,1,padding="same",activation=tf.nn.relu)
            cur_in = tf.layers.conv2d(cur_in,n_filters,3,1,padding="same",activation=tf.nn.relu)
            cur_in = tf.layers.conv2d(cur_in,n_filters/2.0,1,1,padding="same",activation=tf.nn.relu)
            cur_in = tf.layers.conv2d(cur_in,n_filters,3,1,padding="same",activation=tf.nn.relu)

            cur_in = tf.layers.conv2d(cur_in,6,1,1,padding="same",activation=tf.nn.relu)
            cur_in = tf.layers.average_pooling2d(cur_in,(7,7),1,padding="same")
            cur_in = tf.nn.softmax(cur_in)
            return cur_in

    def _build_loss(self):
        """Build our loss."""

        with tf.variable_scope("Loss", reuse=tf.AUTO_REUSE):
            pred_boxes = self.model[:,:,:,0:4]
            pred_area = tf.reduce_prod(tf.subtract(pred_boxes[:,:,:,0:2],pred_boxes[:,:,:,2:4]), axis=-1)
            real_area = tf.reduce_prod(tf.subtract(self.bboxes_in[:,:,0:2],self.bboxes_in[:,:,2:4]), axis=-1)

            max_min = tf.expand_dims(tf.maximum(tf.expand_dims(tf.reshape(pred_boxes[:,:,:,0:2],(tf.shape(pred_boxes)[0],13*13,2))[:,0,:], 1),self.bboxes_in[:,:,0:2]), 0)
            min_max = tf.expand_dims(tf.minimum(tf.expand_dims(tf.reshape(pred_boxes[:,:,:,2:4],(tf.shape(pred_boxes)[0],13*13,2))[:,0,:], 1),self.bboxes_in[:,:,2:4]), 0)

            union = tf.expand_dims(tf.add(tf.reshape(pred_area, (tf.shape(pred_area)[0], 13*13))[:,0],real_area), 0)

            for i in range(1,169):
                max_min = tf.concat( [max_min, tf.expand_dims(tf.maximum(tf.expand_dims(tf.reshape(pred_boxes[:,:,:,0:2],(tf.shape(pred_boxes)[0],13*13,2))[:,i,:], 1), self.bboxes_in[:,:,0:2]), 0)], 0)
                min_max = tf.concat( [min_max, tf.expand_dims(tf.minimum(tf.expand_dims(tf.reshape(pred_boxes[:,:,:,2:4],(tf.shape(pred_boxes)[0],13*13,2))[:,i,:], 1), self.bboxes_in[:,:,2:4]), 0)], 0)

                union = tf.concat(tf.expand_dims(tf.add(tf.expand_dims(tf.reshape(pred_area, (tf.shape(pred_area)[0], 13*13))[:,i], 1),real_area), 0), 0)

            intersection = tf.reduce_prod(tf.subtract(min_max,max_min),axis=-1)

            iou = tf.subtract(tf.div(intersection, union), 1)

            tf.summary.scalar("loss", tf.reduce_sum(tf.reduce_sum(iou)))
            return iou

    def _build_optim(self):
        """Build optimizer related ops and vars."""
        with tf.variable_scope("Optim", reuse=tf.AUTO_REUSE):
            self.global_step = tf.get_variable("global_step",shape=(),dtype=tf.int32,initializer=tf.zeros_initializer)
            adam = tf.train.AdamOptimizer()
            return adam.minimize(self.loss, global_step=self.global_step)

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

    def train(self):
        print('\n--- Training')

        # Run TensorFlow Session
        with tf.Session() as sess:
            print('Initializing...')
            sess.run([
                tf.local_variables_initializer(),
                tf.global_variables_initializer()
            ])
            tf.train.start_queue_runners(sess)

            #images, labels, bboxes = sess.run(batch)
            #print(images.shape)
            #print(labels.shape)
            #print(bboxes.shape)
            for idx_epoch in trange(self.config.max_iter):
                _, sop, gstp = sess.run([self.optimizer, self.summary_op, self.global_step])
                self.summary_tr.add_summary(sop, gstp)
                self.summary_tr.flush()
            #print(labels)
