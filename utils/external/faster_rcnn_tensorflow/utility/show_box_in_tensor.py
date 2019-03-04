# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils.external.faster_rcnn_tensorflow.utility import draw_box_in_img

def only_draw_boxes(img_batch, boxes):

    boxes = tf.stop_gradient(boxes)
    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor = tf.cast(img_tensor, tf.float32)
    labels = tf.ones(shape=(tf.shape(boxes)[0], ), dtype=tf.int32) * draw_box_in_img.ONLY_DRAW_BOXES
    scores = tf.zeros_like(labels, dtype=tf.float32)
    img_tensor_with_boxes = tf.py_func(draw_box_in_img.draw_boxes_with_label_and_scores,
                                       inp=[img_tensor, boxes, labels, scores],
                                       Tout=tf.uint8)
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))  # [batch_size, h, w, c]

    return img_tensor_with_boxes

def draw_boxes_with_scores(img_batch, boxes, scores):

    boxes = tf.stop_gradient(boxes)
    scores = tf.stop_gradient(scores)

    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor = tf.cast(img_tensor, tf.float32)
    labels = tf.ones(shape=(tf.shape(boxes)[0],), dtype=tf.int32) * draw_box_in_img.ONLY_DRAW_BOXES_WITH_SCORES
    img_tensor_with_boxes = tf.py_func(draw_box_in_img.draw_boxes_with_label_and_scores,
                                       inp=[img_tensor, boxes, labels, scores],
                                       Tout=[tf.uint8])
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
    return img_tensor_with_boxes

def draw_boxes_with_categories(img_batch, boxes, labels):
    boxes = tf.stop_gradient(boxes)

    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor = tf.cast(img_tensor, tf.float32)
    scores = tf.ones(shape=(tf.shape(boxes)[0],), dtype=tf.float32)
    img_tensor_with_boxes = tf.py_func(draw_box_in_img.draw_boxes_with_label_and_scores,
                                       inp=[img_tensor, boxes, labels, scores],
                                       Tout=[tf.uint8])
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
    return img_tensor_with_boxes

def draw_boxes_with_categories_and_scores(img_batch, boxes, labels, scores):
    boxes = tf.stop_gradient(boxes)
    scores = tf.stop_gradient(scores)

    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor = tf.cast(img_tensor, tf.float32)
    img_tensor_with_boxes = tf.py_func(draw_box_in_img.draw_boxes_with_label_and_scores,
                                       inp=[img_tensor, boxes, labels, scores],
                                       Tout=[tf.uint8])
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
    return img_tensor_with_boxes

if __name__ == "__main__":
    print (1)

