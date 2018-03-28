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

        # load dataset provider
        provider_v = slim.dataset_data_provider.DatasetDataProvider(
            dataset_val, num_readers=1, shuffle=True)

        [image_v, shape_v, labels_v, bboxes_v, object_count_v] = provider_v.get([
            'image', 'shape', 'object/label', 'object/bbox', 'object/count'
        ])

        # Preprocess
        image, labeles, bboxes = preprocess_for_train(
            image, labels, bboxes)

        image_v, labeles_v, bboxes_v = preprocess_for_train(
            image_v, labels_v, bboxes_v)

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


        batch_v = tf.train.batch(
            [image_v, labels_v, bboxes_v],
            batch_size=batch_size,
            num_threads=1,
            capacity=1 * batch_size,
            dynamic_pad=True,
            allow_smaller_final_batch=True)

        self.image_val = batch_v[0]
        self.label_val = batch_v[1]
        self.bboxes_val = batch_v[2]

        self.image_tr = batch[0]
        self.label_tr = batch[1]
        self.bboxes_tr = batch[2]

        self.image_in = self.image_tr
        self.label_in = self.label_tr
        self.bboxes_in = self.bboxes_tr
        self.best_box = None


        self._build_placeholder()
        self._build_preprocessing()
        self.model = self._build_model()
        self.loss = self._build_loss()
        self.optimizer = self._build_optim()
        self._build_eval()
        self._build_summary()
        self._build_writer()

    def swapSet(self):
        if self.image_in == self.image_tr:
            #switch to validation
            self.image_in = self.image_val
            self.label_in = self.label_val
            self.bboxes_in = self.bboxes_val
        else:
            #switch to train
            self.image_in = self.image_tr
            self.label_in = self.label_tr
            self.bboxes_in = self.bboxes_tr

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
            real_boxes = self.bboxes_in[:,:,0:4]

            pred_boxes = tf.reshape(pred_boxes, (-1, 1, 4))
            real_boxes = tf.reshape(real_boxes, (1, -1, 4))

            x0_p = tf.minimum(pred_boxes[:, :, 0], pred_boxes[:, :, 2])
            x1_p = tf.maximum(pred_boxes[:, :, 0], pred_boxes[:, :, 2])
            y0_p = tf.minimum(pred_boxes[:, :, 1], pred_boxes[:, :, 3])
            y1_p = tf.maximum(pred_boxes[:, :, 1], pred_boxes[:, :, 3])

            x0_r = tf.minimum(real_boxes[:, :, 0], real_boxes[:, :, 2])
            x1_r = tf.maximum(real_boxes[:, :, 0], real_boxes[:, :, 2])
            y0_r = tf.minimum(real_boxes[:, :, 1], real_boxes[:, :, 3])
            y1_r = tf.maximum(real_boxes[:, :, 1], real_boxes[:, :, 3])

            x0 = tf.maximum(x0_p, x0_r)
            x1 = tf.minimum(x1_p, x1_r)
            y0 = tf.maximum(y0_p, y0_r)
            y1 = tf.minimum(y1_p, y1_r)

            inter = (x1 - x0) * (y1 - y0)
            union = (x1_p - x0_p) * (y1_p - y0_p) + (x1_p - x0_p) * (y1_p - y0_p) - inter

            iou = inter / (union + 1)

            min_iou = tf.reduce_min(iou, axis=1)
            loss = tf.reduce_mean(min_iou)

            box_iou_stack = tf.concat([tf.reshape(pred_boxes, (tf.shape(pred_boxes)[0],4)), tf.expand_dims(min_iou,1)], axis=1)
            print(box_iou_stack.shape)
            imin = tf.argmin(box_iou_stack,0)
            self.best_box = box_iou_stack[imin[4], :]

            tf.summary.scalar("loss", loss)
            return loss

    def _build_optim(self):
        """Build optimizer related ops and vars."""
        with tf.variable_scope("Optim", reuse=tf.AUTO_REUSE):
            self.global_step = tf.get_variable("global_step",shape=(),dtype=tf.int32,initializer=tf.zeros_initializer)
            adam = tf.train.AdamOptimizer(learning_rate=1e-4)
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

            self.swapSet()
            bBox = sess.run([self.best_box])
            print("WIN")
            print(bBox)
