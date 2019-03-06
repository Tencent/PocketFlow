# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division
import tensorflow.contrib.slim as slim
import tensorflow as tf

from utils.external import mobilenet_v2
from utils.external.mobilenet import training_scope
from utils.external.mobilenet_v2 import op
from utils.external.mobilenet_v2  import ops
expand_input = ops.expand_input_by_factor

V2_BASE_DEF = dict(
    defaults={
        # Note: these parameters of batch norm affect the architecture
        # that's why they are here and not in training_scope.
        (slim.batch_norm,): {'center': True, 'scale': True},
        (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
            'normalizer_fn': slim.batch_norm, 'activation_fn': tf.nn.relu6
        },
        (ops.expanded_conv,): {
            'expansion_size': expand_input(6),
            'split_expansion': 1,
            'normalizer_fn': slim.batch_norm,
            'residual': True
        },
        (slim.conv2d, slim.separable_conv2d): {'padding': 'SAME'}
    },
    spec=[
        op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
        op(ops.expanded_conv,
           expansion_size=expand_input(1, divisible_by=1),
           num_outputs=16, scope='expanded_conv'),
        op(ops.expanded_conv, stride=2, num_outputs=24, scope='expanded_conv_1'),
        op(ops.expanded_conv, stride=1, num_outputs=24, scope='expanded_conv_2'),
        op(ops.expanded_conv, stride=2, num_outputs=32, scope='expanded_conv_3'),
        op(ops.expanded_conv, stride=1, num_outputs=32, scope='expanded_conv_4'),
        op(ops.expanded_conv, stride=1, num_outputs=32, scope='expanded_conv_5'),
        op(ops.expanded_conv, stride=2, num_outputs=64, scope='expanded_conv_6'),
        op(ops.expanded_conv, stride=1, num_outputs=64, scope='expanded_conv_7'),
        op(ops.expanded_conv, stride=1, num_outputs=64, scope='expanded_conv_8'),
        op(ops.expanded_conv, stride=1, num_outputs=64, scope='expanded_conv_9'),
        op(ops.expanded_conv, stride=1, num_outputs=96, scope='expanded_conv_10'),
        op(ops.expanded_conv, stride=1, num_outputs=96, scope='expanded_conv_11'),
        op(ops.expanded_conv, stride=1, num_outputs=96, scope='expanded_conv_12')
    ],
)


V2_HEAD_DEF = dict(
    defaults={
        # Note: these parameters of batch norm affect the architecture
        # that's why they are here and not in training_scope.
        (slim.batch_norm,): {'center': True, 'scale': True},
        (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
            'normalizer_fn': slim.batch_norm, 'activation_fn': tf.nn.relu6
        },
        (ops.expanded_conv,): {
            'expansion_size': expand_input(6),
            'split_expansion': 1,
            'normalizer_fn': slim.batch_norm,
            'residual': True
        },
        (slim.conv2d, slim.separable_conv2d): {'padding': 'SAME'}
    },
    spec=[
        op(ops.expanded_conv, stride=2, num_outputs=160, scope='expanded_conv_13'),
        op(ops.expanded_conv, stride=1, num_outputs=160, scope='expanded_conv_14'),
        op(ops.expanded_conv, stride=1, num_outputs=160, scope='expanded_conv_15'),
        op(ops.expanded_conv, stride=1, num_outputs=320, scope='expanded_conv_16'),
        op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280, scope='Conv_1')
    ],
)

def mobilenetv2_scope(is_training=True,
                      trainable=True,
                      weight_decay=0.00004,
                      stddev=0.09,
                      dropout_keep_prob=0.8,
                      bn_decay=0.997):
  """Defines Mobilenet training scope.
  In default. We do not use BN

  ReWrite the scope.
  """
  batch_norm_params = {
      'is_training': False,
      'trainable': False,
      'decay': bn_decay,
  }
  with slim.arg_scope(training_scope(is_training=is_training, weight_decay=weight_decay)):
      with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.separable_conv2d],
                          trainable=trainable):
          with slim.arg_scope([slim.batch_norm], **batch_norm_params) as sc:
              return sc



def mobilenetv2_base(img_batch, is_training=True):

    with slim.arg_scope(mobilenetv2_scope(is_training=is_training, trainable=True)):

        feature_to_crop, endpoints = mobilenet_v2.mobilenet_base(input_tensor=img_batch,
                                                      num_classes=None,
                                                      is_training=False,
                                                      depth_multiplier=1.0,
                                                      scope='MobilenetV2',
                                                      conv_defs=V2_BASE_DEF,
                                                      finegrain_classification_mode=False)

        # feature_to_crop = tf.Print(feature_to_crop, [tf.shape(feature_to_crop)], summarize=10, message='rpn_shape')
        return feature_to_crop


def mobilenetv2_head(inputs, is_training=True):
    with slim.arg_scope(mobilenetv2_scope(is_training=is_training, trainable=True)):
        net, _ = mobilenet_v2.mobilenet(input_tensor=inputs,
                                        num_classes=None,
                                        is_training=False,
                                        depth_multiplier=1.0,
                                        scope='MobilenetV2',
                                        conv_defs=V2_HEAD_DEF,
                                        finegrain_classification_mode=False)

        net = tf.squeeze(net, [1, 2])

        return net