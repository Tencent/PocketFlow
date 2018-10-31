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
"""Actor & critic networks' definitions."""

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('ddpg_actor_depth', 2, 'DDPG: actor network\'s depth')
tf.app.flags.DEFINE_integer('ddpg_actor_width', 64, 'DDPG: actor network\'s width')
tf.app.flags.DEFINE_integer('ddpg_critic_depth', 2, 'DDPG: critic network\'s depth')
tf.app.flags.DEFINE_integer('ddpg_critic_width', 64, 'DDPG: critic network\'s width')

ENBL_LAYER_NORM = True

def dense_block(inputs, units):
  """A block of densely connected layers.

  Args:
  * inputs: input tensor to the block
  * units: number of neurons in each layer

  Returns:
  * inputs: output tensor from the block
  """

  inputs = tf.layers.dense(inputs, units)
  if ENBL_LAYER_NORM:
    inputs = tf.contrib.layers.layer_norm(inputs)
  inputs = tf.nn.relu(inputs)

  return inputs

class Model(object):
  """Abstract model for actor & critic networks."""

  def __init__(self, scope):
    """Constructor function.

    Args:
    * scope: name scope in which the model is defined
    """

    self.scope = scope

  @property
  def vars(self):
    """List of all global variables."""

    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)

  @property
  def trainable_vars(self):
    """List of all trainable variables."""

    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

  @property
  def perturbable_vars(self):
    """List of all perturbable variables."""

    return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]

class Actor(Model):
  """Actor network."""

  def __init__(self, a_dims, a_min, a_max, scope='actor'):
    """Constructor function.

    Args:
    * a_dims: number of action vector's dimensions
    * a_min: minimal value in the action vector
    * a_max: maximal value in the action vector
    * scope: name scope in which the model is defined
    """

    super(Actor, self).__init__(scope)

    self.a_dims = a_dims
    self.a_min = a_min
    self.a_max = a_max

  def __call__(self, states, reuse=False):
    """Create the actor network.

    Args:
    * states: state vectors (inputs to the actor network)
    * reuse: whether to reuse previously defined variables

    Returns:
    * inputs: action vectors (outputs from the actor network)
    """

    with tf.variable_scope(self.scope) as scope:
      if reuse:
        scope.reuse_variables()

      inputs = states
      for __ in range(FLAGS.ddpg_actor_depth):
        inputs = dense_block(inputs, FLAGS.ddpg_actor_width)
      inputs = tf.layers.dense(inputs, self.a_dims)
      inputs = tf.sigmoid(inputs) * (self.a_max - self.a_min) + self.a_min

    return inputs

class Critic(Model):
  """Critic network."""

  def __init__(self, scope='critic'):
    """Constructor function.

    Args:
    * scope: name scope in which the model is defined
    """

    super(Critic, self).__init__(scope)

  def __call__(self, states, actions, reuse=False):
    """Create the critic network.

    Args:
    * states: state vectors (inputs to the critic network)
    * actions: action vectors (inputs to the critic network)
    * reuse: whether to reuse previously defined variables

    Returns:
    * inputs: reward scalars (outputs from the critic network)
    """

    with tf.variable_scope(self.scope) as scope:
      if reuse:
        scope.reuse_variables()

      inputs = dense_block(states, FLAGS.ddpg_critic_width)
      inputs = tf.concat([inputs, actions], axis=1)
      for __ in range(FLAGS.ddpg_critic_depth):
        inputs = dense_block(inputs, FLAGS.ddpg_critic_width)
      inputs = tf.layers.dense(inputs, 1)

    return inputs
