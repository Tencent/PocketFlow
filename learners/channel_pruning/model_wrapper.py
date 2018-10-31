# Tencent is pleased to support the open source community by making PocketFlow available.
#
# Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model warpper for easier graph manipulation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.framework import ops as tf_ops # pylint: disable=no-name-in-module

slim = tf.contrib.slim # pylint: disable=no-member
FLAGS = tf.app.flags.FLAGS


class Model:  # pylint: disable=too-many-instance-attributes
  """The model wraper make it easier to do some operation on a tensorflow model"""

  def __init__(self, sess):
    self.sess = sess
    self.g = self.sess.graph
    self._param_data = {}
    self._W1prunable = {}
    self.flops = {}
    self.fathers = {}  # the input op of an op
    self.children = {}  # the input op of an op
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True # pylint: disable=no-member

    def get_data_format():
      """ Get the data format

      Return:
          'NHWC' or 'NCHW'
      """
      with self.g.as_default():
        for operation in self.g.get_operations():
          if operation.type == 'Conv2D':
            return operation.get_attr('data_format').decode("utf-8")

      return 'NHWC'

    self.data_format = get_data_format()

  def get_operations_by_type(self, otype='Conv2D'):
    """Get all the operations of a certain type within the model graph
    Args:
        otype: The type of returned operations
    Return:
        operations of type `otype` within the model graph
    """
    return self.get_operations_by_types([otype])

  def get_operations_by_types(self, otypes=None):
    """Get all the operations of certain types in `otypes` within the model graph
    Args:
        otype: A list containing the type of returned operations
    Return:
        operations of type `otype` within the model graph
    """
    with self.g.as_default():
      operations = []
      for i in self.g.get_operations():
        if i.type in otypes:
          operations.append(i)
    return operations

  def get_outputs_by_type(self, otype='Conv2D'):
    """Get the ouputs of all the operations of a certain type within the model graph

    Args:
        otype: The type of returned operations
    Return:
        operations of type `otype` within the model graph
    """
    return self.get_outputs_by_types([otype])

  def get_outputs_by_types(self, otypes=None):
    """Get the ouputs of all the operations of certain types within the model graph

    Args:
        otypes: A list containing the types of returned operations
    Return:
        operations of type `otypes` within the model graph
    """
    with self.g.as_default():
      outputs = []
      operations = self.get_operations_by_types(otypes)
      for i in operations:
        outputs.append(self.get_output_by_op(i))
      return outputs

  def get_output_by_op(self, op):
    """Get outputs of the operations

    Args:
        op: An operation
    Return:
        The output of a operation
    """
    with self.g.as_default():
      output = op.outputs
      try:
        assert len(output) == 1, 'the length of output should be 1'
      except AssertionError as error:
        tf.logging.error(error)
      return output[0]

  def get_input_by_op(self, op):
    """Get the input of an operation

    Args:
        op: An operation
    Return:
        The input of a operation
    """
    with self.g.as_default():
      inputs = op.inputs
      real_inputs = []
      for inp in inputs:
        if not 'read' in inp.name and 'Const' not in inp.name:
          real_inputs.append(inp)
      if len(real_inputs) != 1:
        print(real_inputs)
      try:
        assert (len(real_inputs) == 1), \
          'the number of real inputs of {} should be 1'.format(op.name)
      except AssertionError as error:
        tf.logging.error(error)
      return real_inputs[0]

  def get_var_by_op(self, op):
    """Get the weights of an operation

    Args:
        op: An operation
    Return:
        The weights of a operation
    """
    with self.g.as_default():
      inptensor = op.inputs
      try:
        assert (len(inptensor) == 2), 'the number of inputs of {} should be 2'.format(op.name)
      except AssertionError as error:
        tf.logging.error(error)
      wtensor = None
      for wtensor in inptensor:
        if 'weight' in wtensor.name or 'bias' in wtensor.name:
          break
      wname = wtensor.name.split('/read:0')[0]
      wvar = slim.get_variables_by_name(wname)
      try:
        assert (len(wvar) == 1), 'the number of weights variable of {} should be 1'.format(op.name)
      except AssertionError as error:
        tf.logging.error(error)
      wvar = wvar[0]
      return wvar

  def param_shape(self, op):
    """ get the shape of an operation weights
    """
    wvar = self.get_var_by_op(op)
    s = []
    for i in wvar.shape:
      s.append(i.value)
    return s

  def param_data(self, op):
    """ get the weights of an operation (k_h,k_w,c,n)"""
    with self.g.as_default():
      opname = op.name
      if opname in self._param_data:
        return self._param_data[opname]
      wvar = self.get_var_by_op(op)
      ret = wvar.eval(session=self.sess)
      self._param_data[opname] = ret
      return ret

  @classmethod
  def get_names(cls, tensors):
    """ get the names of a list of tensors """
    names = []
    for i in tensors:
      names.append(i.name)
    return names

  def get_conv_def(self, op):
    """ get the definition of an operation which contains the following information:
      `ksizes`: kernel sizes
      `padding`: paddings
      `h`: height
      `w`: weight
      `c`: channels
      `n`: output channels
    """
    with self.g.as_default():
      definition = {}
      definition['padding'] = op.get_attr('padding')
      if self.data_format == 'NCHW':
        definition['strides'] = op.get_attr('strides')
        c = definition['strides'][1]
        definition['strides'][1] = definition['strides'][3]
        definition['strides'][3] = c
      else:
        definition['strides'] = op.get_attr('strides')

      s = self.param_shape(op)
      definition['ksizes'] = [1, s[0], s[1], 1]
      definition['h'] = s[0]
      definition['w'] = s[1]
      definition['c'] = s[2]
      definition['n'] = s[3]
      return definition

  @classmethod
  def get_outname_by_opname(cls, opname):
    """ get output name by operation name"""
    return opname + ':0'

  def output_height(self, name):
    """ get the height of a convolution output"""
    if self.data_format == 'NCHW':
      return self.g.get_tensor_by_name(name).shape[2].value

    return self.g.get_tensor_by_name(name).shape[1].value

  def output_width(self, name):
    """ get the width of a convolution output"""
    if self.data_format == 'NCHW':
      return self.g.get_tensor_by_name(name).shape[3].value

    return self.g.get_tensor_by_name(name).shape[2].value

  def output_channels(self, name):
    """ get the number of channels of a convolution output"""
    if self.data_format == 'NCHW':
      return self.g.get_tensor_by_name(name).shape[1].value

    return self.g.get_tensor_by_name(name).shape[3].value

  def compute_layer_flops(self, op):
    """ compute the flops of a certain convolution layer

    Args:
        operation: an convolution layer

    Return:
        The Flops
    """
    with self.g.as_default():
      opname = op.name
      if opname in self.flops:
        flops = self.flops[opname]
      else:
        flops = tf_ops.get_stats_for_node_def(self.g,
                                              op.node_def,
                                              'flops').value
        flops = flops / 2. / FLAGS.batch_size

        self.flops[opname] = flops
      return flops

  def get_Add_if_is_first_after_resblock(self, op):
    """ check whether the input operation is first layer after sum
    in a resual branch.

    Args: 'op' an operation

    Return: the name of the Add operation is the last of a residual block
    """
    curr_op = op
    is_first = True

    while True:
      curr_op = self.get_input_by_op(curr_op).op

      if curr_op.type == 'DepthwiseConv2dNative' or \
          curr_op.type == 'Conv2D':
        is_first = False
        break

      if curr_op.type == 'Add':
        break

    if is_first:
      return curr_op.outputs[0]

    return None

  @classmethod
  def get_Add_if_is_last_in_resblock(cls, op):
    """ check whether the input operation is last layer before sum
    in a resual branch.

    Args:
      'op': an operation

    Return:
      the name of the Add operation is the last of a residual block
    """
    curr_op = op
    is_last = True

    while True:
      next_ops = curr_op.outputs[0].consumers()

      go_on = False
      for curr_op in next_ops:
        if curr_op.type in [
            'Relu',
            'FusedBatchNorm',
            'DepthwiseConv2dNative',
            'MaxPool',
            'Relu6']:
          go_on = True
          break
      if go_on:
        continue

      if curr_op.type == 'Add':
        break
      is_last = False
      break

    if is_last:
      return curr_op.outputs[0]

    return None

  def is_W1_prunable(self, conv):
    """ if the op's input channels can be pruned"""
    conv_name = conv.name
    W1_prunable = True
    o = conv
    while True:
      # print(o.name, o.type)
      o = self.get_input_by_op(o).op
      if o.type in ['Relu',
                    'FusedBatchNorm',
                    'MaxPool',
                    'BiasAdd',
                    'Identity',
                    'Relu6']:
        continue
      if o.type == 'Conv2D' or o.type == 'DepthwiseConv2dNative':
        break
      W1_prunable = False
      break
    if W1_prunable:
      self.fathers[conv_name] = o.name
      self.children[o.name] = conv_name
    else:
      self.fathers[conv_name] = None
      self.children[o.name] = None
    self._W1prunable[conv_name] = W1_prunable
    return W1_prunable
