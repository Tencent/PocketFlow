# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np


def decode_boxes(encoded_boxes, reference_boxes, scale_factors=None):
    '''

    :param encoded_boxes:[N, 4]
    :param reference_boxes: [N, 4] .
    :param scale_factors: use for scale.

    in the first stage, reference_boxes  are anchors
    in the second stage, reference boxes are proposals(decode) produced by first stage
    :return:decode boxes [N, 4]
    '''

    t_xcenter, t_ycenter, t_w, t_h = tf.unstack(encoded_boxes, axis=1)
    if scale_factors:
        t_xcenter /= scale_factors[0]
        t_ycenter /= scale_factors[1]
        t_w /= scale_factors[2]
        t_h /= scale_factors[3]

    reference_xmin, reference_ymin, reference_xmax, reference_ymax = tf.unstack(reference_boxes, axis=1)
    # reference boxes are anchors in the first stage

    # reference_xcenter = (reference_xmin + reference_xmax) / 2.
    # reference_ycenter = (reference_ymin + reference_ymax) / 2.
    reference_w = reference_xmax - reference_xmin
    reference_h = reference_ymax - reference_ymin
    reference_xcenter = reference_xmin + reference_w/2.0
    reference_ycenter = reference_ymin + reference_h/2.0

    predict_xcenter = t_xcenter * reference_w + reference_xcenter
    predict_ycenter = t_ycenter * reference_h + reference_ycenter
    predict_w = tf.exp(t_w) * reference_w
    predict_h = tf.exp(t_h) * reference_h

    predict_xmin = predict_xcenter - predict_w / 2.
    predict_xmax = predict_xcenter + predict_w / 2.
    predict_ymin = predict_ycenter - predict_h / 2.
    predict_ymax = predict_ycenter + predict_h / 2.

    return tf.transpose(tf.stack([predict_xmin, predict_ymin,
                                  predict_xmax, predict_ymax]))


def encode_boxes(unencode_boxes, reference_boxes, scale_factors=None):
    '''

    :param unencode_boxes: [-1, 4]
    :param reference_boxes: [-1, 4]
    :return: encode_boxes [-1, 4]
    '''

    xmin, ymin, xmax, ymax = unencode_boxes[:, 0], unencode_boxes[:, 1], unencode_boxes[:, 2], unencode_boxes[:, 3]

    reference_xmin, reference_ymin, reference_xmax, reference_ymax = reference_boxes[:, 0], reference_boxes[:, 1], \
                                                                     reference_boxes[:, 2], reference_boxes[:, 3]

    # x_center = (xmin + xmax) / 2.
    # y_center = (ymin + ymax) / 2.
    w = xmax - xmin + 1e-8
    h = ymax - ymin + 1e-8
    x_center = xmin + w/2.0
    y_center = ymin + h/2.0

    # reference_xcenter = (reference_xmin + reference_xmax) / 2.
    # reference_ycenter = (reference_ymin + reference_ymax) / 2.
    reference_w = reference_xmax - reference_xmin + 1e-8
    reference_h = reference_ymax - reference_ymin + 1e-8
    reference_xcenter = reference_xmin + reference_w/2.0
    reference_ycenter = reference_ymin + reference_h/2.0
    # w + 1e-8 to avoid NaN in division and log below

    t_xcenter = (x_center - reference_xcenter) / reference_w
    t_ycenter = (y_center - reference_ycenter) / reference_h
    t_w = np.log(w/reference_w)
    t_h = np.log(h/reference_h)

    if scale_factors:
        t_xcenter *= scale_factors[0]
        t_ycenter *= scale_factors[1]
        t_w *= scale_factors[2]
        t_h *= scale_factors[3]

    return np.transpose(np.stack([t_xcenter, t_ycenter, t_w, t_h], axis=0))
