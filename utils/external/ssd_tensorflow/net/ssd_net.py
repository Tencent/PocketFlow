# Copyright 2018 Changan Wang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-5
_USE_FUSED_BN = True

# vgg_16/conv2/conv2_1/biases
# vgg_16/conv4/conv4_3/biases
# vgg_16/conv1/conv1_1/biases
# vgg_16/fc6/weights
# vgg_16/conv3/conv3_2/biases
# vgg_16/conv5/conv5_3/biases
# vgg_16/conv3/conv3_1/weights
# vgg_16/conv4/conv4_2/weights
# vgg_16/conv1/conv1_1/weights
# vgg_16/conv5/conv5_3/weights
# vgg_16/conv4/conv4_1/weights
# vgg_16/conv3/conv3_3/weights
# vgg_16/conv5/conv5_2/biases
# vgg_16/conv3/conv3_2/weights
# vgg_16/conv4/conv4_2/biases
# vgg_16/conv5/conv5_2/weights
# vgg_16/conv3/conv3_1/biases
# vgg_16/conv2/conv2_2/weights
# vgg_16/fc7/weights
# vgg_16/conv5/conv5_1/biases
# vgg_16/conv1/conv1_2/biases
# vgg_16/conv2/conv2_2/biases
# vgg_16/conv4/conv4_1/biases
# vgg_16/fc7/biases
# vgg_16/fc6/biases
# vgg_16/conv4/conv4_3/weights
# vgg_16/conv2/conv2_1/weights
# vgg_16/conv5/conv5_1/weights
# vgg_16/conv3/conv3_3/biases
# vgg_16/conv1/conv1_2/weights

class ReLuLayer(tf.layers.Layer):
    def __init__(self, name, **kwargs):
        super(ReLuLayer, self).__init__(name=name, trainable=trainable, **kwargs)
        self._name = name
    def build(self, input_shape):
        self._relu = lambda x : tf.nn.relu(x, name=self._name)
        self.built = True

    def call(self, inputs):
        return self._relu(inputs)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

def forward_module(m, inputs, training=False):
    if isinstance(m, tf.layers.BatchNormalization) or isinstance(m, tf.layers.Dropout):
        return m.apply(inputs, training=training)
    return m.apply(inputs)

class VGG16Backbone(object):
    def __init__(self, data_format='channels_first'):
        super(VGG16Backbone, self).__init__()
        self._data_format = data_format
        self._bn_axis = -1 if data_format == 'channels_last' else 1
        #initializer = tf.glorot_uniform_initializer  glorot_normal_initializer
        self._conv_initializer = tf.glorot_uniform_initializer
        self._conv_bn_initializer = tf.glorot_uniform_initializer#lambda : tf.truncated_normal_initializer(mean=0.0, stddev=0.005)
        # VGG layers
        self._conv1_block = self.conv_block(2, 64, 3, (1, 1), 'conv1')
        self._pool1 = tf.layers.MaxPooling2D(2, 2, padding='same', data_format=self._data_format, name='pool1')
        self._conv2_block = self.conv_block(2, 128, 3, (1, 1), 'conv2')
        self._pool2 = tf.layers.MaxPooling2D(2, 2, padding='same', data_format=self._data_format, name='pool2')
        self._conv3_block = self.conv_block(3, 256, 3, (1, 1), 'conv3')
        self._pool3 = tf.layers.MaxPooling2D(2, 2, padding='same', data_format=self._data_format, name='pool3')
        self._conv4_block = self.conv_block(3, 512, 3, (1, 1), 'conv4')
        self._pool4 = tf.layers.MaxPooling2D(2, 2, padding='same', data_format=self._data_format, name='pool4')
        self._conv5_block = self.conv_block(3, 512, 3, (1, 1), 'conv5')
        self._pool5 = tf.layers.MaxPooling2D(3, 1, padding='same', data_format=self._data_format, name='pool5')
        self._conv6 = tf.layers.Conv2D(filters=1024, kernel_size=3, strides=1, padding='same', dilation_rate=6,
                            data_format=self._data_format, activation=tf.nn.relu, use_bias=True,
                            kernel_initializer=self._conv_initializer(),
                            bias_initializer=tf.zeros_initializer(),
                            name='fc6', _scope='fc6', _reuse=None)
        self._conv7 = tf.layers.Conv2D(filters=1024, kernel_size=1, strides=1, padding='same',
                            data_format=self._data_format, activation=tf.nn.relu, use_bias=True,
                            kernel_initializer=self._conv_initializer(),
                            bias_initializer=tf.zeros_initializer(),
                            name='fc7', _scope='fc7', _reuse=None)
        # SSD layers
        with tf.variable_scope('additional_layers') as scope:
            self._conv8_block = self.ssd_conv_block(256, 2, 'conv8')
            self._conv9_block = self.ssd_conv_block(128, 2, 'conv9')
            self._conv10_block = self.ssd_conv_block(128, 1, 'conv10', padding='valid')
            self._conv11_block = self.ssd_conv_block(128, 1, 'conv11', padding='valid')

    def l2_normalize(self, x, name):
        with tf.name_scope(name, "l2_normalize", [x]) as name:
            axis = -1 if self._data_format == 'channels_last' else 1
            square_sum = tf.reduce_sum(tf.square(x), axis, keep_dims=True)
            x_inv_norm = tf.rsqrt(tf.maximum(square_sum, 1e-10))
            return tf.multiply(x, x_inv_norm, name=name)

    def forward(self, inputs, training=False):
        # inputs should in BGR
        feature_layers = []
        # forward vgg layers
        for conv in self._conv1_block:
            inputs = forward_module(conv, inputs, training=training)
        inputs = self._pool1.apply(inputs)
        for conv in self._conv2_block:
            inputs = forward_module(conv, inputs, training=training)
        inputs = self._pool2.apply(inputs)
        for conv in self._conv3_block:
            inputs = forward_module(conv, inputs, training=training)
        inputs = self._pool3.apply(inputs)
        for conv in self._conv4_block:
            inputs = forward_module(conv, inputs, training=training)
        # conv4_3
        with tf.variable_scope('conv4_3_scale') as scope:
            weight_scale = tf.Variable([20.] * 512, trainable=training, name='weights')
            if self._data_format == 'channels_last':
                weight_scale = tf.reshape(weight_scale, [1, 1, 1, -1], name='reshape')
            else:
                weight_scale = tf.reshape(weight_scale, [1, -1, 1, 1], name='reshape')

            feature_layers.append(tf.multiply(weight_scale, self.l2_normalize(inputs, name='norm'), name='rescale')
                                )
        inputs = self._pool4.apply(inputs)
        for conv in self._conv5_block:
            inputs = forward_module(conv, inputs, training=training)
        inputs = self._pool5.apply(inputs)
        # forward fc layers
        inputs = self._conv6.apply(inputs)
        inputs = self._conv7.apply(inputs)
        # fc7
        feature_layers.append(inputs)
        # forward ssd layers
        for layer in self._conv8_block:
            inputs = forward_module(layer, inputs, training=training)
        # conv8
        feature_layers.append(inputs)
        for layer in self._conv9_block:
            inputs = forward_module(layer, inputs, training=training)
        # conv9
        feature_layers.append(inputs)
        for layer in self._conv10_block:
            inputs = forward_module(layer, inputs, training=training)
        # conv10
        feature_layers.append(inputs)
        for layer in self._conv11_block:
            inputs = forward_module(layer, inputs, training=training)
        # conv11
        feature_layers.append(inputs)

        return feature_layers

    def conv_block(self, num_blocks, filters, kernel_size, strides, name, reuse=None):
        with tf.variable_scope(name):
            conv_blocks = []
            for ind in range(1, num_blocks + 1):
                conv_blocks.append(
                        tf.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                            data_format=self._data_format, activation=tf.nn.relu, use_bias=True,
                            kernel_initializer=self._conv_initializer(),
                            bias_initializer=tf.zeros_initializer(),
                            name='{}_{}'.format(name, ind), _scope='{}_{}'.format(name, ind), _reuse=None)
                    )
            return conv_blocks

    def ssd_conv_block(self, filters, strides, name, padding='same', reuse=None):
        with tf.variable_scope(name):
            conv_blocks = []
            conv_blocks.append(
                    tf.layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding=padding,
                        data_format=self._data_format, activation=tf.nn.relu, use_bias=True,
                        kernel_initializer=self._conv_initializer(),
                        bias_initializer=tf.zeros_initializer(),
                        name='{}_1'.format(name), _scope='{}_1'.format(name), _reuse=None)
                )
            conv_blocks.append(
                    tf.layers.Conv2D(filters=filters * 2, kernel_size=3, strides=strides, padding=padding,
                        data_format=self._data_format, activation=tf.nn.relu, use_bias=True,
                        kernel_initializer=self._conv_initializer(),
                        bias_initializer=tf.zeros_initializer(),
                        name='{}_2'.format(name), _scope='{}_2'.format(name), _reuse=None)
                )
            return conv_blocks

    def ssd_conv_bn_block(self, filters, strides, name, reuse=None):
        with tf.variable_scope(name):
            conv_bn_blocks = []
            conv_bn_blocks.append(
                    tf.layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding='same',
                        data_format=self._data_format, activation=None, use_bias=False,
                        kernel_initializer=self._conv_bn_initializer(),
                        bias_initializer=None,
                        name='{}_1'.format(name), _scope='{}_1'.format(name), _reuse=None)
                )
            conv_bn_blocks.append(
                    tf.layers.BatchNormalization(axis=self._bn_axis, momentum=BN_MOMENTUM, epsilon=BN_EPSILON, fused=USE_FUSED_BN,
                        name='{}_bn1'.format(name), _scope='{}_bn1'.format(name), _reuse=None)
                )
            conv_bn_blocks.append(
                    ReLuLayer('{}_relu1'.format(name), _scope='{}_relu1'.format(name), _reuse=None)
                )
            conv_bn_blocks.append(
                    tf.layers.Conv2D(filters=filters * 2, kernel_size=3, strides=strides, padding='same',
                        data_format=self._data_format, activation=None, use_bias=False,
                        kernel_initializer=self._conv_bn_initializer(),
                        bias_initializer=None,
                        name='{}_2'.format(name), _scope='{}_2'.format(name), _reuse=None)
                )
            conv_bn_blocks.append(
                    tf.layers.BatchNormalization(axis=self._bn_axis, momentum=BN_MOMENTUM, epsilon=BN_EPSILON, fused=USE_FUSED_BN,
                        name='{}_bn2'.format(name), _scope='{}_bn2'.format(name), _reuse=None)
                )
            conv_bn_blocks.append(
                    ReLuLayer('{}_relu2'.format(name), _scope='{}_relu2'.format(name), _reuse=None)
                )
            return conv_bn_blocks

def multibox_head(feature_layers, num_classes, num_anchors_depth_per_layer, data_format='channels_first'):
    with tf.variable_scope('multibox_head'):
        cls_preds = []
        loc_preds = []
        for ind, feat in enumerate(feature_layers):
            loc_preds.append(tf.layers.conv2d(feat, num_anchors_depth_per_layer[ind] * 4, (3, 3), use_bias=True,
                        name='loc_{}'.format(ind), strides=(1, 1),
                        padding='same', data_format=data_format, activation=None,
                        kernel_initializer=tf.glorot_uniform_initializer(),
                        bias_initializer=tf.zeros_initializer()))
            cls_preds.append(tf.layers.conv2d(feat, num_anchors_depth_per_layer[ind] * num_classes, (3, 3), use_bias=True,
                        name='cls_{}'.format(ind), strides=(1, 1),
                        padding='same', data_format=data_format, activation=None,
                        kernel_initializer=tf.glorot_uniform_initializer(),
                        bias_initializer=tf.zeros_initializer()))

        return loc_preds, cls_preds


