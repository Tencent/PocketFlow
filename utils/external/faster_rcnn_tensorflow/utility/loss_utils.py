# -*- coding: utf-8 -*-
"""
@author: jemmy li
@contact: zengarden2009@gmail.com
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _smooth_l1_loss_base(bbox_pred, bbox_targets, sigma=1.0):
    '''

    :param bbox_pred: [-1, 4] in RPN. [-1, cls_num+1, 4] in Fast-rcnn
    :param bbox_targets: shape is same as bbox_pred
    :param sigma:
    :return:
    '''
    sigma_2 = sigma**2

    box_diff = bbox_pred - bbox_targets

    abs_box_diff = tf.abs(box_diff)

    smoothL1_sign = tf.stop_gradient(
        tf.to_float(tf.less(abs_box_diff, 1. / sigma_2)))
    loss_box = tf.pow(box_diff, 2) * (sigma_2 / 2.0) * smoothL1_sign \
               + (abs_box_diff - (0.5 / sigma_2)) * (1.0 - smoothL1_sign)
    return loss_box

def smooth_l1_loss_rpn(bbox_pred, bbox_targets, label, sigma=1.0):
    '''

    :param bbox_pred: [-1, 4]
    :param bbox_targets: [-1, 4]
    :param label: [-1]
    :param sigma:
    :return:
    '''
    value = _smooth_l1_loss_base(bbox_pred, bbox_targets, sigma=sigma)
    value = tf.reduce_sum(value, axis=1)  # to sum in axis 1
    rpn_select = tf.where(tf.greater(label, 0))

    # rpn_select = tf.stop_gradient(rpn_select) # to avoid
    selected_value = tf.gather(value, rpn_select)
    non_ignored_mask = tf.stop_gradient(
        1.0 - tf.to_float(tf.equal(label, -1))) # positve is 1.0 others is 0.0

    bbox_loss = tf.reduce_sum(selected_value) / tf.maximum(1.0, tf.reduce_sum(non_ignored_mask))

    return bbox_loss



def smooth_l1_loss_rcnn(bbox_pred, bbox_targets, label, num_classes, sigma=1.0):
    '''

    :param bbox_pred: [-1, (cfgs.CLS_NUM +1) * 4]
    :param bbox_targets:[-1, (cfgs.CLS_NUM +1) * 4]
    :param label:[-1]
    :param num_classes:
    :param sigma:
    :return:
    '''

    outside_mask = tf.stop_gradient(tf.to_float(tf.greater(label, 0)))

    bbox_pred = tf.reshape(bbox_pred, [-1, num_classes, 4])
    bbox_targets = tf.reshape(bbox_targets, [-1, num_classes, 4])

    value = _smooth_l1_loss_base(bbox_pred,
                                 bbox_targets,
                                 sigma=sigma)
    value = tf.reduce_sum(value, 2)
    value = tf.reshape(value, [-1, num_classes])

    inside_mask = tf.one_hot(tf.reshape(label, [-1, 1]),
                             depth=num_classes, axis=1)

    inside_mask = tf.stop_gradient(
        tf.to_float(tf.reshape(inside_mask, [-1, num_classes])))

    normalizer = tf.to_float(tf.shape(bbox_pred)[0])
    bbox_loss = tf.reduce_sum(
        tf.reduce_sum(value * inside_mask, 1)*outside_mask) / normalizer

    return bbox_loss


def sum_ohem_loss(cls_score, label, bbox_pred, bbox_targets,
                  num_classes, num_ohem_samples=256, sigma=1.0):
    '''

    :param cls_score: [-1, cls_num+1]
    :param label: [-1]
    :param bbox_pred: [-1, 4*(cls_num+1)]
    :param bbox_targets: [-1, 4*(cls_num+1)]
    :param num_ohem_samples: 256 by default
    :param num_classes: cls_num+1
    :param sigma:
    :return:
    '''

    cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label)  # [-1, ]
    # cls_loss = tf.Print(cls_loss, [tf.shape(cls_loss)], summarize=10, message='CLS losss shape ****')

    outside_mask = tf.stop_gradient(tf.to_float(tf.greater(label, 0)))
    bbox_pred = tf.reshape(bbox_pred, [-1, num_classes, 4])
    bbox_targets = tf.reshape(bbox_targets, [-1, num_classes, 4])

    value = _smooth_l1_loss_base(bbox_pred,
                                 bbox_targets,
                                 sigma=sigma)
    value = tf.reduce_sum(value, 2)
    value = tf.reshape(value, [-1, num_classes])

    inside_mask = tf.one_hot(tf.reshape(label, [-1, 1]),
                             depth=num_classes, axis=1)

    inside_mask = tf.stop_gradient(
        tf.to_float(tf.reshape(inside_mask, [-1, num_classes])))
    loc_loss = tf.reduce_sum(value * inside_mask, 1)*outside_mask
    # loc_loss = tf.Print(loc_loss, [tf.shape(loc_loss)], summarize=10, message='loc_loss shape***')

    sum_loss = cls_loss + loc_loss

    num_ohem_samples = tf.stop_gradient(tf.minimum(num_ohem_samples, tf.shape(sum_loss)[0]))
    _, top_k_indices = tf.nn.top_k(sum_loss, k=num_ohem_samples)

    cls_loss_ohem = tf.gather(cls_loss, top_k_indices)
    cls_loss_ohem = tf.reduce_mean(cls_loss_ohem)

    loc_loss_ohem = tf.gather(loc_loss, top_k_indices)
    normalizer = tf.to_float(num_ohem_samples)
    loc_loss_ohem = tf.reduce_sum(loc_loss_ohem) / normalizer

    return cls_loss_ohem, loc_loss_ohem

