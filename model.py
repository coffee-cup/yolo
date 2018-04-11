import os

import numpy as np
from tqdm import trange

import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import *
from utils.preprocess import (preprocess_for_train, preprocess_for_validation,
                              process_bboxes_and_labels)
from utils.voc_common import (BOX, CLASSES, GRID_H, GRID_W, IMAGE_H, IMAGE_W,
                              YOLO_ANCHORS, combine_images, decode_netout,
                              draw_boxes, preprocess_true_boxes, save_image)

OBJ_THRESHOLD = 0.3
NMS_THRESHOLD = 0.3

CLASS_WEIGHTS = np.ones(CLASSES, dtype='float32')
OBJ_THRESHOLD = 0.3  #0.5
NMS_THRESHOLD = 0.3  #0.45

NO_OBJECT_SCALE = 1.0
OBJECT_SCALE = 5.0
COORD_SCALE = 1.0
CLASS_SCALE = 1.0

WARM_UP_BATCHES = 20


class Yolo(object):
    '''Yolo network'''

    def __init__(self, config, dataset_train, dataset_val, debug=False):
        self.config = config
        self.debug = debug

        # Load dataset provider
        provider_tr = slim.dataset_data_provider.DatasetDataProvider(
            dataset_train, num_readers=1, shuffle=True)

        provider_va = slim.dataset_data_provider.DatasetDataProvider(
            dataset_val, num_readers=1, shuffle=True)

        [image_tr, labels_tr, bboxes_tr, y_true_tr] = provider_tr.get(
            ['image', 'object/label', 'object/bbox', 'object/y_true'])

        [image_va, labels_va, bboxes_va, y_true_va] = provider_va.get(
            ['image', 'object/label', 'object/bbox', 'object/y_true'])

        # Preprocess
        image_tr, bboxes_tr, y_true_tr = preprocess_for_train(
            image_tr, labels_tr, bboxes_tr, y_true_tr)

        image_va, bboxes_va, y_true_va = preprocess_for_validation(
            image_va, labels_va, bboxes_va, y_true_va)

        print('image {}'.format(image_tr))
        print('labels {}'.format(labels_tr))
        print('bboxes {}'.format(bboxes_tr))

        # Create batches
        batch_size = self.config.batch_size
        self.batch_tr = tf.train.batch(
            [image_tr, bboxes_tr, y_true_tr],
            batch_size=batch_size,
            num_threads=4,
            capacity=2 * batch_size,
            dynamic_pad=True,
            allow_smaller_final_batch=True)

        self.batch_va = tf.train.batch(
            [image_va, bboxes_va, y_true_va],
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
        self.optimizer = self._build_optim()
        # self._build_eval()
        self._build_summary()
        self._build_writer()

    def _build_placeholder(self):
        '''Build placeholders.'''
        self.images_in = tf.placeholder(
            tf.float32, shape=(None, IMAGE_W, IMAGE_H, 3))
        self.bboxes_in = tf.placeholder(tf.float32, shape=(None, None, 5))

        num_anchors = len(YOLO_ANCHORS)
        self.y_true = tf.placeholder(
            tf.float32,
            shape=(None, GRID_H, GRID_W, num_anchors, 4 + 1 + CLASSES))

        self.training = tf.placeholder(tf.bool, shape=())

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
                    use_bias=False,
                    kernel_initializer=tf.truncated_normal_initializer(
                        0.0, 0.01),
                    kernel_regularizer=slim.l2_regularizer(0.0005),
                    bias_initializer=tf.zeros_initializer())
                x = tf.layers.batch_normalization(x, training=self.training)
                x = tf.nn.leaky_relu(x, alpha=0.2)

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
            y_true = self.y_true
            y_pred = self.model

            batch_size = tf.shape(self.bboxes_in)[0]
            true_boxes = bboxes[..., 0:4]
            true_boxes = tf.reshape(true_boxes, (batch_size, 1, 1, 1, -1, 4))

            mask_shape = tf.shape(y_true)[:4]
            total_recall = tf.Variable(0.)
            seen = tf.Variable(0.)

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

            num_anchors = len(YOLO_ANCHORS)

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
            intersect_maxes = tf.minimum(pred_maxes, true_maxes)
            intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

            true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
            pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

            union_areas = pred_areas + true_areas - intersect_areas
            iou_scores = tf.truediv(intersect_areas, union_areas)

            true_box_conf = iou_scores * y_true[..., 4]

            # adjust class probabilities
            true_box_class = tf.argmax(y_true[..., 5:], -1)

            ###
            # Determine the masks
            ###

            # position of the ground truth boxes
            coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * COORD_SCALE

            # penalize predictors + penalize boxes with low IOU
            true_xy = true_boxes[..., 0:2]
            true_wh = true_boxes[..., 2:4]

            true_wh_half = true_wh / 2.
            true_mins = true_xy - true_wh_half
            true_maxes = true_xy + true_wh_half

            pred_xy = tf.expand_dims(pred_box_xy, 4)
            pred_wh = tf.expand_dims(pred_box_wh, 4)

            pred_wh_half = pred_wh / 2.
            pred_mins = pred_xy - pred_wh_half
            pred_maxes = pred_xy + pred_wh_half

            intersect_mins = tf.maximum(pred_mins, true_mins)
            intersect_maxes = tf.minimum(pred_maxes, true_maxes)
            intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

            true_areas = true_wh[..., 0] * true_wh[..., 1]
            pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

            union_areas = pred_areas + true_areas - intersect_areas
            iou_scores = tf.truediv(intersect_areas, union_areas)

            best_ious = tf.reduce_max(iou_scores, axis=4)
            conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (
                1 - y_true[..., 4]) * NO_OBJECT_SCALE

            # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
            conf_mask = conf_mask + y_true[..., 4] * OBJECT_SCALE

            ### class mask: simply the position of the ground truth boxes (the predictors)
            class_mask = y_true[..., 4] * tf.gather(
                CLASS_WEIGHTS, true_box_class) * CLASS_SCALE

            ###
            # Warm up training
            ###
            no_boxes_mask = tf.to_float(coord_mask < COORD_SCALE / 2.)
            seen = tf.assign_add(seen, 1.)

            true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, WARM_UP_BATCHES),
                          lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
                                   true_box_wh + tf.ones_like(true_box_wh) * np.reshape(YOLO_ANCHORS, [1,1,1,BOX,2]) * no_boxes_mask,
                                   tf.ones_like(coord_mask)],
                          lambda: [true_box_xy,
                                   true_box_wh,
                                   coord_mask])

            # Finalize the loss
            nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
            nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
            nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

            loss_xy = tf.reduce_sum(
                tf.square(true_box_xy - pred_box_xy) * coord_mask) / (
                    nb_coord_box + 1e-6) / 2.
            loss_wh = tf.reduce_sum(
                tf.square(true_box_wh - pred_box_wh) * coord_mask) / (
                    nb_coord_box + 1e-6) / 2.
            loss_conf = tf.reduce_sum(
                tf.square(true_box_conf - pred_box_conf) * conf_mask) / (
                    nb_conf_box + 1e-6) / 2.
            loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=true_box_class, logits=pred_box_class)
            loss_class = tf.reduce_sum(loss_class * class_mask) / (
                nb_class_box + 1e-6)

            loss = loss_xy + loss_wh + loss_conf + loss_class

            nb_true_box = tf.reduce_sum(y_true[..., 4])
            nb_pred_box = tf.reduce_sum(
                tf.to_float(true_box_conf > 0.5) *
                tf.to_float(pred_box_conf > 0.3))

            current_recall = nb_pred_box / (nb_true_box + 1e-6)
            total_recall = tf.assign_add(total_recall, current_recall)

            if self.debug:
                loss = tf.Print(
                    loss, [loss_xy], message='Loss XY \t', summarize=1000)
                loss = tf.Print(
                    loss, [loss_wh], message='Loss WH \t', summarize=1000)
                loss = tf.Print(
                    loss, [loss_conf], message='Loss Conf \t', summarize=1000)
                loss = tf.Print(
                    loss, [loss_class],
                    message='Loss Class \t',
                    summarize=1000)
                loss = tf.Print(
                    loss, [loss], message='Total Loss \t', summarize=1000)

            tf.summary.scalar('loss', loss)
            tf.summary.scalar('recall', total_recall)
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
        self.saver_cur = tf.train.Saver()
        # self.saver_best = tf.train.Saver()

        # # Save file for the current model
        self.save_file_cur = os.path.join(self.config.log_dir, 'model')

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

            latest_checkpoint = tf.train.latest_checkpoint(self.config.log_dir)
            b_resume = latest_checkpoint is not None and self.config.allow_restore
            if b_resume:
                print('Restoring from {}...'.format(self.config.log_dir))
                self.saver_cur.restore(sess, latest_checkpoint)
                res = sess.run(fetches={'global_step': self.global_step})

                step = res['global_step']
            else:
                step = 0

            for idx_epoch in trange(step, self.config.max_iter):
                # Get training batch
                images_tr, bboxes_tr, y_true_tr = sess.run(self.batch_tr)

                # print('\n--- Boxes, shape {}'.format(bboxes_tr.shape))
                # print(bboxes_tr)

                # print('\n--- y_true, shape {}'.format(y_true.shape))

                b_write_summary = (
                    step is 0) or (step + 1) % self.config.report_freq is 0
                if b_write_summary:
                    fetches = {
                        'loss': self.loss,
                        'optimizer': self.optimizer,
                        'global_step': self.global_step,
                        'summary': self.summary_op
                    }
                else:
                    fetches = {'optimizer': self.optimizer}

                res = sess.run(
                    fetches=fetches,
                    feed_dict={
                        self.images_in: images_tr,
                        self.bboxes_in: bboxes_tr,
                        self.y_true: y_true_tr,
                        self.training: True
                    })

                if res.get('summary') is not None:
                    self.summary_tr.add_summary(res['summary'],
                                                res['global_step'])
                    self.summary_tr.flush()

                    self.saver_cur.save(
                        sess,
                        self.save_file_cur,
                        global_step=self.global_step,
                        write_meta_graph=False)

                # print('\n\n--- Loss')
                # print(res['loss'])

                # print('\n\n--- Loss Shape')
                # print(res['loss'].shape)

                # print(res)
                # print(res[:, 0])

                # image = images_tr[0]
                # boxes = bboxes_tr[0]

                # print(boxes)

                # netout = preprocess_true_boxes(boxes)
                # boxes = decode_netout(netout)
                # image = draw_boxes(image, boxes)
                # save_image(image, 'test.jpeg')

                if idx_epoch == 0 or idx_epoch % self.config.val_freq == 0:
                    images_va, bboxes_va, y_true_va = sess.run(self.batch_va)

                    res = sess.run(
                        fetches={
                            'summary': self.summary_op,
                            'global_step': self.global_step,
                            'netout': self.model
                        },
                        feed_dict={
                            self.images_in: images_va,
                            self.bboxes_in: bboxes_va,
                            self.y_true: y_true_va,
                            self.training: False
                        })

                    image = images_va[0]

                    # True
                    boxes_true = decode_netout(y_true_va[0])
                    image_true = draw_boxes(image, boxes_true)

                    # Pred
                    boxes_pred = decode_netout(res['netout'][0])
                    image_pred = draw_boxes(image, boxes_pred)

                    image_combined = combine_images([image_true, image_pred])
                    image_combined.save(
                        'eval_images/{}.jpeg'.format(idx_epoch))

                #     if self.config.print_boxes:
                #         print(res['best_boxes'])

                #     self.summary_va.add_summary(res['summary'],
                #                                 res['global_step'])
                #     self.summary_va.flush()

            coord.request_stop()
            coord.join(threads)
            sess.close()
