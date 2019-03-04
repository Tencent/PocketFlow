# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import tensorflow as tf

def make_anchors(base_anchor_size, anchor_scales, anchor_ratios,
                 featuremap_height, featuremap_width,
                 stride, name='make_anchors'):
    '''
    :param base_anchor_size:256
    :param anchor_scales:
    :param anchor_ratios:
    :param featuremap_height:
    :param featuremap_width:
    :param stride:
    :return:
    '''
    with tf.variable_scope(name):
        base_anchor = tf.constant([0, 0, base_anchor_size, base_anchor_size], tf.float32)  # [x_center, y_center, w, h]

        ws, hs = enum_ratios(enum_scales(base_anchor, anchor_scales),
                             anchor_ratios)  # per locations ws and hs

        x_centers = tf.range(featuremap_width, dtype=tf.float32) * stride
        y_centers = tf.range(featuremap_height, dtype=tf.float32) * stride

        x_centers, y_centers = tf.meshgrid(x_centers, y_centers)

        ws, x_centers = tf.meshgrid(ws, x_centers)
        hs, y_centers = tf.meshgrid(hs, y_centers)

        anchor_centers = tf.stack([x_centers, y_centers], 2)
        anchor_centers = tf.reshape(anchor_centers, [-1, 2])

        box_sizes = tf.stack([ws, hs], axis=2)
        box_sizes = tf.reshape(box_sizes, [-1, 2])
        # anchors = tf.concat([anchor_centers, box_sizes], axis=1)
        anchors = tf.concat([anchor_centers - 0.5*box_sizes,
                             anchor_centers + 0.5*box_sizes], axis=1)
        return anchors


def enum_scales(base_anchor, anchor_scales):

    anchor_scales = base_anchor * tf.constant(anchor_scales, dtype=tf.float32, shape=(len(anchor_scales), 1))

    return anchor_scales


def enum_ratios(anchors, anchor_ratios):
    '''
    ratio = h /w
    :param anchors:
    :param anchor_ratios:
    :return:
    '''
    ws = anchors[:, 2]  # for base anchor: w == h
    hs = anchors[:, 3]
    sqrt_ratios = tf.sqrt(tf.constant(anchor_ratios))

    ws = tf.reshape(ws / sqrt_ratios[:, tf.newaxis], [-1, 1])
    hs = tf.reshape(hs * sqrt_ratios[:, tf.newaxis], [-1, 1])

    return hs, ws


