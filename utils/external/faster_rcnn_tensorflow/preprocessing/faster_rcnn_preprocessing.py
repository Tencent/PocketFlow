# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from utils.external.faster_rcnn_tensorflow.configs import cfgs

import tensorflow as tf

import numpy as np


def max_length_limitation(length, length_limitation):
    return tf.cond(tf.less(length, length_limitation),
                   true_fn=lambda: length,
                   false_fn=lambda: length_limitation)

def short_side_resize(img_tensor, labels, bboxes, target_shortside_len, length_limitation=1200):
    '''

    :param img_tensor:[h, w, c], gtboxes_and_label:[-1, 5].  gtboxes: [xmin, ymin, xmax, ymax]
    :param target_shortside_len:
    :param length_limitation: set max length to avoid OUT OF MEMORY
    :return:
    '''
    img_h, img_w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]

    new_h, new_w = tf.cond(tf.less(img_h, img_w),
                           true_fn=lambda: (target_shortside_len,
                                            max_length_limitation(target_shortside_len * img_w // img_h, length_limitation)),
                           false_fn=lambda: (max_length_limitation(target_shortside_len * img_h // img_w, length_limitation),
                                             target_shortside_len))

    img_tensor = tf.expand_dims(img_tensor, axis=0)
    img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])
    ymin, xmin, ymax, xmax = tf.unstack(bboxes, axis=1)

    img_h = tf.cast(img_h, tf.float32)
    img_w = tf.cast(img_w,tf.float32)
    new_h = tf.cast(new_h,tf.float32)
    new_w = tf.cast(new_w,tf.float32)

    ymin = ymin * img_h
    ymax = ymax * img_h
    xmin = xmin * img_w
    xmax = xmax * img_w

    new_xmin, new_ymin = xmin * new_w // img_w, ymin * new_h // img_h
    new_xmax, new_ymax = xmax * new_w // img_w, ymax * new_h // img_h

    img_tensor = tf.squeeze(img_tensor, axis=0)  # ensure image tensor rank is 3

    return img_tensor, labels, tf.cast(tf.transpose(tf.stack([new_xmin, new_ymin, new_xmax, new_ymax], axis=0)),dtype=tf.int32)


def short_side_resize_for_inference_data(img_tensor, target_shortside_len, length_limitation=1200):
    img_h, img_w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]

    new_h, new_w = tf.cond(tf.less(img_h, img_w),
                           true_fn=lambda: (target_shortside_len,
                                            max_length_limitation(target_shortside_len * img_w // img_h, length_limitation)),
                           false_fn=lambda: (max_length_limitation(target_shortside_len * img_h // img_w, length_limitation),
                                             target_shortside_len))

    img_tensor = tf.expand_dims(img_tensor, axis=0)
    img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])

    img_tensor = tf.squeeze(img_tensor, axis=0)  # ensure image tensor rank is 3
    return img_tensor

def flip_left_to_right(img_tensor, labels, bboxes):

  h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
  img_tensor = tf.image.flip_left_right(img_tensor)
  xmin, ymin, xmax, ymax= tf.unstack(bboxes, axis=1)
  new_xmax = w - xmin
  new_xmin = w - xmax

  return img_tensor, labels, tf.transpose(tf.stack([new_xmin, ymin, new_xmax, ymax], axis=0))

def random_flip_left_right(img_tensor, labels, bboxes):
    img_tensor, labels, bboxes= tf.cond(tf.less(tf.random_uniform(shape=[], minval=0, maxval=1), 0.5),
                                            lambda: flip_left_to_right(img_tensor, labels, bboxes),
                                            lambda: (img_tensor, labels, bboxes))

    return img_tensor, labels, bboxes


def preprocess_for_train(image, labels, bboxes, out_shape, data_format='channels_first', scope='ssd_preprocessing_train', output_rgb=True):
  img = tf.cast(image, tf.float32)
  img, labels, bboxes = short_side_resize(img_tensor = img, labels = labels, bboxes = bboxes,
                                             target_shortside_len =cfgs.IMG_SHORT_SIDE_LEN, length_limitation=cfgs.IMG_MAX_LENGTH)
  img, labels, bboxes = random_flip_left_right(img_tensor=img,labels = labels, bboxes = bboxes)

  img = img - tf.constant([[cfgs.PIXEL_MEAN]])  # sub pixel mean at last
  return img, labels, bboxes

def preprocess_for_eval(image, out_shape, data_format='channels_first', scope='ssd_preprocessing_eval', output_rgb=True):
  img = tf.cast(image, tf.float32)
  img = short_side_resize_for_inference_data(img_tensor=img,
                                             target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                             length_limitation=cfgs.IMG_MAX_LENGTH)
  img = img - tf.constant(cfgs.PIXEL_MEAN)
  return img

def preprocess_image(image, labels, bboxes, out_shape, is_training=False, data_format='channels_first', output_rgb=True):
  """Preprocesses the given image.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    labels: A `Tensor` containing all labels for all bboxes of this image.
    bboxes: A `Tensor` containing all bboxes of this image, in range [0., 1.] with shape [num_bboxes, 4].
    out_shape: The height and width of the image after preprocessing.
    is_training: Wether we are in training phase.
    data_format: The data_format of the desired output image.

  Returns:
    A preprocessed image.
  """
  if is_training:
    return preprocess_for_train(image, labels, bboxes, out_shape, data_format=data_format, output_rgb=output_rgb)
  else:
    return preprocess_for_eval(image, out_shape, data_format=data_format, output_rgb=output_rgb)