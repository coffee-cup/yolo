import os
import numpy as np
import pandas as pd
import tensorflow as tf
import utils

class Loss(dict):
    def __init__(self, model, mask, prob, coords, offset_xy_min, offset_xy_max, areas):
        self.model = model
        with tf.name_scope('true'):
            self.mask = tf.identity(mask, name='mask')
            self.prob = tf.identity(prob, name='prob')
            self.coords = tf.identity(coords, name='coords')
            self.offset_xy_min = tf.identity(offset_xy_min, name='offset_xy_min')
            self.offset_xy_max = tf.identity(offset_xy_max, name='offset_xy_max')
            self.areas = tf.identity(areas, name='areas')
        with tf.name_scope('iou') as name:
            _offset_xy_min = tf.maximum(model.offset_xy_min, self.offset_xy_min, name='_offset_xy_min')
            _offset_xy_max = tf.minimum(model.offset_xy_max, self.offset_xy_max, name='_offset_xy_max')
            _wh = tf.maximum(_offset_xy_max - _offset_xy_min, 0.0, name='_wh')
            _areas = tf.reduce_prod(_wh, -1, name='_areas')
            areas = tf.maximum(self.areas + model.areas - _areas, 1e-10, name='areas')
            iou = tf.truediv(_areas, areas, name=name)
        with tf.name_scope('mask'):
            best_box_iou = tf.reduce_max(iou, 2, True, name='best_box_iou')
            best_box = tf.to_float(tf.equal(iou, best_box_iou), name='best_box')
            mask_best = tf.identity(self.mask * best_box, name='mask_best')
            mask_normal = tf.identity(1 - mask_best, name='mask_normal')
        with tf.name_scope('dist'):
            iou_dist = tf.square(model.iou - mask_best, name='iou_dist')
            coords_dist = tf.square(model.coords - self.coords, name='coords_dist')
            prob_dist = tf.square(model.prob - self.prob, name='prob_dist')
        with tf.name_scope('objectives'):
            cnt = np.multiply.reduce(iou_dist.get_shape().as_list())
            self['iou_best'] = tf.identity(tf.reduce_sum(mask_best * iou_dist) / cnt, name='iou_best')
            self['iou_normal'] = tf.identity(tf.reduce_sum(mask_normal * iou_dist) / cnt, name='iou_normal')
            _mask_best = tf.expand_dims(mask_best, -1)
            self['coords'] = tf.identity(tf.reduce_sum(_mask_best * coords_dist) / cnt, name='coords')
            self['prob'] = tf.identity(tf.reduce_sum(_mask_best * prob_dist) / cnt, name='prob')

# HOW to use
# with tf.name_scope('weighted_loss'):
#    for key in self.objectives:
#        tf.add_to_collection(tf.GraphKeys.LOSSES, tf.multiply(self.objectives[key], self.config.getfloat(section + '_hparam', key), name='weighted_' + key))
# then later ...
# total_loss = tf.losses.get_total_loss(name="total_loss")
