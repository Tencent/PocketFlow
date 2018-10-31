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
""" RL Helper Class for Non-Uniform Quantization """

import random
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

class RLHelper(object):
  # pylint: disable=too-many-instance-attributes
  """ The Helper Class for the DDPG algorithm. Making sure the states and actions
      satisfy the condition of constraints given the total number bits.
  """

  def __init__(self, sess, total_bits, num_weights, vars_list, random_layers=False):
    """
    Args:
    * sess: session to run the shape
    * total_bits: A scalar, total num of bits as budget
    * num_weights: A list of number of weights in each layers
    * var_list: List of Tensors. Weights to quantize
    * random_layers: True if shuffle the layers
   """

    self.nb_vars = len(num_weights)
    self.num_weights = num_weights
    self.total_num_weights = sum(num_weights)

    # TODO: the design of states is experimental
    self.s_dims = self.nb_vars + 6 # one hot encoding for layer ids
    self.total_bits = total_bits
    self.w_bits_used = 0
    self.random_layers = random_layers
    self.layer_idxs = list(range(self.nb_vars))
    self.num_weights_to_quantize = self.total_num_weights
    self.quantized_layers = 0

    self.var_shapes = []
    self.states = np.zeros((self.nb_vars, self.s_dims))

    for idx, var in enumerate(vars_list):
      var_shape = sess.run(tf.shape(var))
      shape_len = var.shape.__len__()
      assert shape_len in [2, 4], \
        "Unknown weight shape. Must be a 2 (fc) or 4 (conv) dimensional."
      if shape_len == 2:
        var_shape = np.hstack((np.ones(2), var_shape))
      self.var_shapes += [var_shape]

    for idx in range(self.nb_vars):
      state = np.zeros(self.s_dims)
      state[idx] = 1.0
      state[self.nb_vars  : self.nb_vars + 4] = self.var_shapes[idx]
      state[self.nb_vars + 4] = self.num_weights[idx] / np.max(self.num_weights)
      state[self.nb_vars + 5] = np.sum(self.num_weights[idx + 1 : ]) / self.total_num_weights
      self.states[idx] = state

  def calc_state(self, idx):
    """ return the rl state """
    state = np.copy(self.states[idx])
    return state[None, :]

  def calc_reward(self, accuracy):
    """ return the rl reward via reshaping """
    return accuracy * np.ones((1, 1))

  def reset(self):
    """ reset the helper params for each rollout """
    self.w_bits_used = 0
    self.quantized_layers = 0
    if self.random_layers:
      random.shuffle(self.layer_idxs)
    self.num_weights_to_quantize = self.total_num_weights

  def __calc_w_duty(self, idx):
    """ Compute the maximum bits used for layer idx """
    duty = self.total_bits - self.w_bits_used - self.num_weights_to_quantize*FLAGS.nuql_w_bit_min
    assert duty >= 0, "No enough budget for layer {}".format(idx)
    return duty

  def calc_w(self, action, idx):
    """
    Args:
    * action: An ndarray with shape (1,1), the output of actor network
    * idx: A scalar, the id of layer
    * num_weights: A list of scalars, the number of weights in each layer

    Return:
    * An ndarray with the same shape with 'action'
    """

    duty = self.__calc_w_duty(idx)
    if self.quantized_layers != self.nb_vars - 1:
      action = np.round(action) + FLAGS.nuql_w_bit_min
      action = np.minimum(action, FLAGS.nuql_w_bit_min + np.floor(duty*1.0/self.num_weights[idx]))
    else:
      action = np.floor((self.total_bits - self.w_bits_used)/self.num_weights[idx]) * np.ones((1, 1))
    action = np.minimum(action, FLAGS.nuql_w_bit_max)
    self.w_bits_used += action[0][0] * self.num_weights[idx]
    self.num_weights_to_quantize -= self.num_weights[idx]
    self.quantized_layers += 1
    return action
