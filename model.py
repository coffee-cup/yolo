import os

import numpy as np
from tqdm import trange

import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import *
from utils.preprocess import (preprocess_for_train, preprocess_for_validation,
                              process_bboxes_and_labels)
from utils.voc_common import (BOX, CLASSES, GRID_H, GRID_W, IMAGE_H, IMAGE_W,
                              YOLO_ANCHORS)

OBJ_THRESHOLD = 0.3
NMS_THRESHOLD = 0.3

NO_OBJECT_SCALE = 1.0
OBJECT_SCALE = 5.0
COORD_SCALE = 1.0
CLASS_SCALE = 1.0


class Yolo(object):
    '''Yolo network'''

    def __init__(self, config, dataset_train, dataset_val):
        self.config = config

        # Load dataset provider
        provider_tr = slim.dataset_data_provider.DatasetDataProvider(
            dataset_train, num_readers=1, shuffle=True)

        provider_va = slim.dataset_data_provider.DatasetDataProvider(
            dataset_val, num_readers=1, shuffle=False)

        [image_tr, labels_tr, bboxes_tr, detectors_mask,
         true_boxes] = provider_tr.get([
             'image', 'object/label', 'object/bbox', 'object/detectors_mask',
             'object/matching_true_boxes'
         ])

        [image_va, labels_va,
         bboxes_va] = provider_va.get(['image', 'object/label', 'object/bbox'])

        # Preprocess
        image, bboxes, detectors_mask, true_boxes = preprocess_for_train(
            image_tr, labels_tr, bboxes_tr, detectors_mask, true_boxes)

        image_va, labels_va, bboxes_va = preprocess_for_validation(
            image_va, labels_va, bboxes_va, IMAGE_H)

        print('image {}'.format(image_tr))
        print('labels {}'.format(labels_tr))
        print('bboxes {}'.format(bboxes_tr))
        print('mask {}'.format(detectors_mask))
        print('true boxes {}'.format(true_boxes))

        # Create batches
        batch_size = self.config.batch_size
        self.batch_tr = tf.train.batch(
            [image, bboxes, detectors_mask, true_boxes],
            batch_size=batch_size,
            num_threads=1,
            capacity=1 * batch_size,
            dynamic_pad=True,
            allow_smaller_final_batch=True)

        self.batch_va = tf.train.batch(
            [image_va, labels_va, bboxes_va],
            # batch_size=batch_size,
            batch_size=1,
            num_threads=1,
            capacity=1,
            dynamic_pad=True,
            allow_smaller_final_batch=True)

        self._build_placeholder()
        self._build_preprocessing()
        self.model = self._build_model()
        self.loss = self._build_loss()
        # self.optimizer = self._build_optim()
        # self._build_eval()
        self._build_summary()
        self._build_writer()

    def _build_placeholder(self):
        '''Build placeholders.'''
        self.images_in = tf.placeholder(
            tf.float32, shape=(None, IMAGE_W, IMAGE_H, 3))
        self.bboxes_in = tf.placeholder(tf.float32, shape=(None, None, 5))

        num_anchors = len(YOLO_ANCHORS)
        self.detectors_mask = tf.placeholder(
            tf.float32, shape=(None, GRID_H, GRID_W, num_anchors, 1))
        self.true_boxes = tf.placeholder(
            tf.float32, shape=(None, GRID_H, GRID_W, num_anchors, 5))

    def _build_preprocessing(self):
        '''Build preprocessing related graph.'''
        pass

    def _build_model(self):
        ''' Arguments required for darknet :
            net, classes, num_anchors, training=False, center=True'''

        def conv_layer(x, filters, kernel_size, name):
            with tf.variable_scope(name):
                x = tf.layers.conv2d(
                    x,
                    filters,
                    kernel_size,
                    padding='same',
                    kernel_initializer=tf.contrib.layers.
                    xavier_initializer_conv2d(),
                    bias_initializer=tf.zeros_initializer())
                x = tf.nn.relu(x)

            return x

        def pool_layer(x, size, stride, name):
            with tf.name_scope(name):
                x = tf.layers.max_pooling2d(x, size, stride, padding='same')

            return x

        def passthrough_layer(a, b, filters, depth, size, name):
            b = conv_layer(b, filters, depth, name)
            b = tf.space_to_depth(b, size)
            y = tf.concat([a, b], axis=3)

            return y

        with tf.variable_scope('Network', reuse=tf.AUTO_REUSE):
            cur_in = conv_layer(self.images_in, 32, 3, 'conv1')
            cur_in = pool_layer(cur_in, 2, 2, 'maxpool1')
            cur_in = conv_layer(cur_in, 64, 3, 'conv2')
            cur_in = pool_layer(cur_in, 2, 2, 'maxpool2')

            cur_in = conv_layer(cur_in, 128, 3, 'conv3')
            cur_in = conv_layer(cur_in, 64, 1, 'conv4')
            cur_in = conv_layer(cur_in, 128, 3, 'conv5')
            cur_in = pool_layer(cur_in, 2, 2, 'maxpool5')

            cur_in = conv_layer(cur_in, 256, 3, 'conv6')
            cur_in = conv_layer(cur_in, 128, 1, 'conv7')
            cur_in = conv_layer(cur_in, 256, 3, 'conv8')
            cur_in = pool_layer(cur_in, 2, 2, 'maxpool8')

            cur_in = conv_layer(cur_in, 512, 3, 'conv9')
            cur_in = conv_layer(cur_in, 256, 1, 'conv10')
            cur_in = conv_layer(cur_in, 512, 3, 'conv11')
            cur_in = conv_layer(cur_in, 256, 1, 'conv12')
            passthrough = conv_layer(cur_in, 512, 3, 'conv13')
            cur_in = pool_layer(passthrough, 2, 2, 'maxpool13')

            cur_in = conv_layer(cur_in, 1024, 3, 'conv14')
            cur_in = conv_layer(cur_in, 512, 1, 'conv15')
            cur_in = conv_layer(cur_in, 1024, 3, 'conv16')
            cur_in = conv_layer(cur_in, 512, 1, 'conv17')
            cur_in = conv_layer(cur_in, 1024, 3, 'conv18')

            cur_in = conv_layer(cur_in, 1024, 3, 'conv19')
            cur_in = conv_layer(cur_in, 1024, 3, 'conv20')
            cur_in = passthrough_layer(cur_in, passthrough, 64, 3, 2, 'conv21')
            cur_in = conv_layer(cur_in, 1024, 3, 'conv22')
            cur_in = conv_layer(cur_in, BOX * (4 + 1 + CLASSES), 1, 'conv23')

            y = tf.reshape(
                cur_in,
                shape=(-1, GRID_H, GRID_W, BOX, 4 + 1 + CLASSES),
                name='y')

            return y

    def _build_loss(self):
        with tf.variable_scope('Loss', reuse=tf.AUTO_REUSE):
            bboxes = self.bboxes_in
            y_true = self.true_boxes
            detectors_mask = self.detectors_mask
            batch_size = tf.shape(self.bboxes_in)[0]

            y_pred = self.model

            print('model output shape {}'.format(self.model.shape))
            print('bboxes in shape {}'.format(self.bboxes_in.shape))
            print('y_true shape {}'.format(y_true.shape))

            mask_shape = tf.shape(y_true)[:4]
            print('mask shape {}'.format(mask_shape))

            cell_x = tf.to_float(
                tf.reshape(
                    tf.tile(tf.range(GRID_W), [GRID_H]),
                    (1, GRID_H, GRID_W, 1, 1)))
            cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

            cell_grid = tf.tile(
                tf.concat([cell_x, cell_y], -1), [batch_size, 1, 1, 5, 1])

            coord_mask = tf.zeros(mask_shape)
            conf_mask = tf.zeros(mask_shape)
            class_mask = tf.zeros(mask_shape)

            ###
            # Adjust Prediction
            ###

            # adjust x and y
            pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

            # adjust w and h
            pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(
                YOLO_ANCHORS, [1, 1, 1, BOX, 2])

            # adjust confidence
            pred_box_conf = tf.sigmoid(y_pred[..., 4])

            # adjust class probabilities
            pred_box_class = y_pred[..., 5:]

            print('pred box xy  {}'.format(pred_box_xy.shape))
            print('pred box wh  {}'.format(pred_box_wh.shape))
            print('pred conf    {}'.format(pred_box_conf.shape))
            print('pred classes {}'.format(pred_box_class.shape))

            ###
            # Adjust Ground Truth
            ###

            # adjust x and y
            true_box_xy = y_true[..., 0:2]

            # adjust w and h
            true_box_wh = y_true[..., 2:4]

            # adjust confidence
            true_wh_half = true_box_wh / 2.
            true_mins = true_box_xy - true_wh_half
            true_maxes = true_box_xy + true_wh_half

            pred_wh_half = pred_box_wh / 2.
            pred_mins = pred_box_xy - pred_wh_half
            pred_maxes = pred_box_xy + pred_wh_half

            intersect_mins = tf.maximum(pred_mins, true_mins)

            return intersect_mins

    def _build_loss_2(self):
        '''Build our loss.'''

        with tf.variable_scope('Loss', reuse=tf.AUTO_REUSE):
            pred_boxes = self.model[:, :, :, 0:4]
            real_boxes = self.bboxes_in[:, :, 0:4]

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
            union = (x1_p - x0_p) * (y1_p - y0_p) + (x1_p - x0_p) * (
                y1_p - y0_p) - inter

            iou = inter / (union + 1)

            min_iou = tf.reduce_min(iou, axis=1)
            loss = tf.reduce_mean(min_iou)

            box_iou_stack = tf.concat(
                [
                    tf.reshape(pred_boxes, (tf.shape(pred_boxes)[0], 4)),
                    tf.expand_dims(min_iou, 1)
                ],
                axis=1)

            batch_size = tf.shape(self.bboxes_in)[0]

            # reshape box predictions

            # One bounding box prediction per cell
            self.boxes = tf.reshape(pred_boxes, (batch_size, -1, 4))

            # The iou prediction for the bounding boxes
            self.min_iou = tf.reshape(min_iou, (batch_size, -1, 1))

            # The indicies for the best bounding box in each batch
            self.best_index = tf.argmin(
                tf.reshape(min_iou, (batch_size, -1, 1)), axis=1)

            batch_range = tf.expand_dims(tf.range(batch_size), 1)

            # Get the top bounding box per batch
            indicies = tf.concat(
                [batch_range, tf.cast(self.best_index, tf.int32)], axis=1)
            self.best_boxes = tf.gather_nd(self.boxes, indicies)

            # self.best_boxes = tf.reduce_min(self.boxes_iou, axis=[4, 2])
            # self.thing = tf.concat([self.boxes, self.min_iou], axis=1)
            # self.best_boxes = tf.gather_nd(
            #     self.boxes, [np.arange(batch_size), self.best_index])

            # imin = tf.argmin(box_iou_stack, 1)
            # self.best_box = box_iou_stack[imin[4], :][0:4]
            # self.best_box = tf.reshape(self.best_box, (1, 1, 4))

            tf.summary.scalar('loss', loss)
            return loss

    def _build_optim(self):
        '''Build optimizer related ops and vars.'''
        with tf.variable_scope('Optim', reuse=tf.AUTO_REUSE):
            self.global_step = tf.get_variable(
                'global_step',
                shape=(),
                dtype=tf.int32,
                initializer=tf.zeros_initializer)
            adam = tf.train.AdamOptimizer(
                learning_rate=self.config.learning_rate)
            return adam.minimize(self.loss, global_step=self.global_step)

    def _build_eval(self):
        '''Build the evaluation related ops'''
        with tf.variable_scope('Eval', tf.AUTO_REUSE):
            images_with_boxes = tf.image.draw_bounding_boxes(
                self.images_in, self.boxes)
            tf.summary.image('prediction', images_with_boxes)

    def _build_summary(self):
        '''Build summary ops.'''
        self.summary_op = tf.summary.merge_all()

    def _build_writer(self):
        '''Build the writers and savers'''
        self.summary_tr = tf.summary.FileWriter(
            os.path.join(self.config.log_dir, 'train'))
        self.summary_va = tf.summary.FileWriter(
            os.path.join(self.config.log_dir, 'valid'))

        # Create savers (one for current, one for best)
        # self.saver_cur = tf.train.Saver()
        # self.saver_best = tf.train.Saver()

        # # Save file for the current model
        # self.save_file_cur = os.path.join(self.config.log_dir, 'model')

        # # Save file for the best model
        # self.save_file_best = os.path.join(self.config.save_dir, 'model')

    def train(self):
        print('\n--- Training')

        # Run TensorFlow Session
        conf = tf.ConfigProto()
        conf.gpu_options.allow_growth = True
        with tf.Session(config=conf) as sess:
            print('Initializing...')
            coord = tf.train.Coordinator()
            sess.run([
                tf.local_variables_initializer(),
                tf.global_variables_initializer()
            ])
            threads = tf.train.start_queue_runners(sess, coord=coord)

            # Add the session graph to training summary
            # so we can view in Tensorboard
            self.summary_tr.add_graph(sess.graph)

            for idx_epoch in trange(self.config.max_iter):
                print('\n\n')

                # Get training batch
                images_tr, bboxes_tr, detectors_mask, true_boxes = sess.run(
                    self.batch_tr)

                print('\n--- Boxes, shape {}'.format(bboxes_tr.shape))
                print(bboxes_tr)

                print('\n--- Mask, shape {}'.format(detectors_mask.shape))

                print('\n--- True Boxes, shape {}'.format(true_boxes.shape))

                res = sess.run(
                    self.loss,
                    feed_dict={
                        self.images_in: images_tr,
                        self.bboxes_in: bboxes_tr,
                        self.detectors_mask: detectors_mask,
                        self.true_boxes: true_boxes
                    })

                print('\n\n--- Output')
                print(res)

                # print('\n\n--- Output Shape')
                # print(res.shape)

                # print(res)
                # print(res[:, 0])

                # res = sess.run(
                #     fetches={
                #         # 'optimizer': self.optimizer,
                #         'global_step': self.global_step
                #         # 'summary': self.summary_op
                #     },
                #     feed_dict={
                #         self.images_in: images_tr,
                #         self.labels_in: labels_tr,
                #         self.bboxes_in: bboxes_tr
                #     })

                # self.summary_tr.add_summary(res['summary'], res['global_step'])
                # self.summary_tr.flush()

                break

                if idx_epoch == 0 or idx_epoch % self.config.val_freq == 0:
                    images_va, labels_va, bboxes_va = sess.run(self.batch_va)
                    res = sess.run(
                        fetches={
                            'summary': self.summary_op,
                            'global_step': self.global_step,
                            'best_boxes': self.best_boxes
                        },
                        feed_dict={
                            self.images_in: images_va,
                            self.labels_in: labels_va,
                            self.bboxes_in: bboxes_va
                        })

                    if self.config.print_boxes:
                        print(res['best_boxes'])

                    self.summary_va.add_summary(res['summary'],
                                                res['global_step'])
                    self.summary_va.flush()

            coord.request_stop()
            coord.join(threads)
            sess.close()
