import os

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tqdm import trange
from utils.preprocess import preprocess_for_train, preprocess_for_validation


class Yolo(object):
    '''Yolo network'''

    def __init__(self, config, dataset_train, dataset_val):
        self.config = config

        # Load dataset provider
        provider_tr = slim.dataset_data_provider.DatasetDataProvider(
            dataset_train, num_readers=1, shuffle=True)

        provider_va = slim.dataset_data_provider.DatasetDataProvider(
            dataset_val, num_readers=1, shuffle=False)

        [image_tr, labels_tr,
         bboxes_tr] = provider_tr.get(['image', 'object/label', 'object/bbox'])

        [image_va, labels_va,
         bboxes_va] = provider_va.get(['image', 'object/label', 'object/bbox'])

        # Preprocess
        image_tr, labels_tr, bboxes_tr = preprocess_for_train(
            image_tr, labels_tr, bboxes_tr, config.image_size)

        image_va, labels_va, bboxes_va = preprocess_for_validation(
            image_va, labels_va, bboxes_va, config.image_size)

        print('image {}'.format(image_tr))
        print('labels {}'.format(labels_tr))
        print('bboxes {}'.format(bboxes_tr))

        # Create batches
        batch_size = self.config.batch_size
        self.batch_tr = tf.train.batch(
            [image_tr, labels_tr, bboxes_tr],
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

        self.best_box = None

        self._build_placeholder()
        self._build_preprocessing()
        self.model = self._build_model()
        self.loss = self._build_loss()
        self.optimizer = self._build_optim()
        self._build_eval()
        self._build_summary()
        self._build_writer()

    def data_train(self):
        self.data_set(name='train')

    def data_val(self):
        self.data_set(name='val')

    def _build_placeholder(self):
        '''Build placeholders.'''
        self.images_in = tf.placeholder(
            tf.float32,
            shape=(None, self.config.image_size, self.config.image_size, 3))
        self.labels_in = tf.placeholder(
            tf.int64, shape=(
                None,
                None,
            ))
        self.bboxes_in = tf.placeholder(tf.float32, shape=(None, None, 4))

    def _build_preprocessing(self):
        '''Build preprocessing related graph.'''
        pass

    def _build_model(self):
        ''' Arguments required for darknet :
            net, classes, num_anchors, training=False, center=True'''
        with tf.variable_scope('network', reuse=tf.AUTO_REUSE):
            n_filters = 32

            cur_in = tf.layers.conv2d(
                self.images_in,
                n_filters,
                7,
                1,
                padding='same',
                activation=tf.nn.relu)
            cur_in = tf.layers.max_pooling2d(cur_in, (2, 2), 2, padding='same')

            n_filters = n_filters * 2
            cur_in = tf.layers.conv2d(
                cur_in, n_filters, 3, 1, padding='same', activation=tf.nn.relu)
            cur_in = tf.layers.max_pooling2d(cur_in, (2, 2), 2, padding='same')

            n_filters = n_filters * 2
            cur_in = tf.layers.conv2d(
                cur_in, n_filters, 3, 1, padding='same', activation=tf.nn.relu)
            cur_in = tf.layers.conv2d(
                cur_in,
                n_filters / 2.0,
                1,
                1,
                padding='same',
                activation=tf.nn.relu)
            cur_in = tf.layers.conv2d(
                cur_in, n_filters, 3, 1, padding='same', activation=tf.nn.relu)
            cur_in = tf.layers.max_pooling2d(cur_in, (2, 2), 2, padding='same')

            n_filters = n_filters * 2
            cur_in = tf.layers.conv2d(
                cur_in, n_filters, 3, 1, padding='same', activation=tf.nn.relu)
            cur_in = tf.layers.conv2d(
                cur_in,
                n_filters / 2.0,
                1,
                1,
                padding='same',
                activation=tf.nn.relu)
            cur_in = tf.layers.conv2d(
                cur_in, n_filters, 3, 1, padding='same', activation=tf.nn.relu)
            cur_in = tf.layers.max_pooling2d(cur_in, (2, 2), 2, padding='same')

            n_filters = n_filters * 2
            cur_in = tf.layers.conv2d(
                cur_in, n_filters, 3, 1, padding='same', activation=tf.nn.relu)
            cur_in = tf.layers.conv2d(
                cur_in,
                n_filters / 2.0,
                1,
                1,
                padding='same',
                activation=tf.nn.relu)
            cur_in = tf.layers.conv2d(
                cur_in, n_filters, 3, 1, padding='same', activation=tf.nn.relu)
            cur_in = tf.layers.conv2d(
                cur_in,
                n_filters / 2.0,
                1,
                1,
                padding='same',
                activation=tf.nn.relu)
            cur_in = tf.layers.conv2d(
                cur_in, n_filters, 3, 1, padding='same', activation=tf.nn.relu)
            cur_in = tf.layers.max_pooling2d(cur_in, (2, 2), 2, padding='same')

            n_filters = n_filters * 2
            cur_in = tf.layers.conv2d(
                cur_in, n_filters, 3, 1, padding='same', activation=tf.nn.relu)
            cur_in = tf.layers.conv2d(
                cur_in,
                n_filters / 2.0,
                1,
                1,
                padding='same',
                activation=tf.nn.relu)
            cur_in = tf.layers.conv2d(
                cur_in, n_filters, 3, 1, padding='same', activation=tf.nn.relu)
            cur_in = tf.layers.conv2d(
                cur_in,
                n_filters / 2.0,
                1,
                1,
                padding='same',
                activation=tf.nn.relu)
            cur_in = tf.layers.conv2d(
                cur_in, n_filters, 3, 1, padding='same', activation=tf.nn.relu)

            cur_in = tf.layers.conv2d(
                cur_in, 6, 1, 1, padding='same', activation=tf.nn.relu)
            cur_in = tf.layers.average_pooling2d(
                cur_in, (7, 7), 1, padding='same')
            cur_in = tf.nn.softmax(cur_in)
            return cur_in

    def _build_loss(self):
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
            print(batch_size)

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
            print('in eval')
            print('images in: {}'.format(self.images_in.shape))
            print('best boxes: {}'.format(self.best_boxes.shape))

            images_with_boxes = tf.image.draw_bounding_boxes(
                self.images_in, self.boxes)
            # tf.summary.image('predictions', images_with_boxes)
            # i = self.images_in[0, :, :, :]
            # i = tf.expand_dims(i, 0)
            # print('images {}'.format(i.shape))
            # print('boxes {}'.format(self.best_box.shape))
            # image_with_box = tf.image.draw_bounding_boxes(i, self.best_box)
            # tf.summary.image('validation', image_with_box)

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
            sess.run([
                tf.local_variables_initializer(),
                tf.global_variables_initializer()
            ])
            tf.train.start_queue_runners(sess)

            for idx_epoch in trange(self.config.max_iter):
                # Get training batch
                images_tr, labels_tr, bboxes_tr = sess.run(self.batch_tr)

                res = sess.run(
                    fetches={
                        'optimizer': self.optimizer,
                        'global_step': self.global_step,
                        'best_boxes': self.best_boxes,
                        'boxes': self.boxes,
                        'min_iou': self.min_iou,
                        'best_index': self.best_index
                    },
                    feed_dict={
                        self.images_in: images_tr,
                        self.labels_in: labels_tr,
                        self.bboxes_in: bboxes_tr
                    })

                print('\nFUCKKK')
                print('best boxes: {}'.format(res['best_boxes'].shape))
                print('images: {}'.format(images_tr.shape))
                print('boxes!: {}'.format(res['boxes'].shape))
                print('min_iou: {}'.format(res['min_iou'].shape))
                print('best_index: {}'.format(res['best_index']))

                print('\n\nHELLOOOO\n')

                boxes = res['boxes']
                indexs = tf.expand_dims(
                    tf.range(res['best_index'].shape[0]), 1)
                print('best index shape: {}'.format(res['best_index'].shape))
                print('range shape: {}'.format(indexs.shape))

                inds = tf.concat([indexs, res['best_index']], axis=1)
                row = tf.gather_nd(boxes, inds)

                print('row: {}'.format(sess.run(row)))
                print(sess.run(inds))

                # self.summary_tr.add_summary(res['summary'], res['global_step'])
                # self.summary_tr.flush()

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
                    self.summary_va.add_summary(res['summary'],
                                                res['global_step'])
                    self.summary_va.flush()

            sess.close()
