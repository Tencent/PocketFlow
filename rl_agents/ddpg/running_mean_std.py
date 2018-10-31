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
"""Running averages of mean value & standard deviation."""

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('ddpg_rms_eps', 1e-4, 'DDPG: running standard deviation\'s epsilon')

class RunningMeanStd(object):
  """Running averages of mean value & standard deviation."""

  def __init__(self, sess, nb_dims):
    """Constructor function.

    Args:
    * sess: TensorFlow session
    * nb_dims: number of sample's dimensions
    """

    self.sess = sess

    # statistics for computing running mean & standard deviation
    x_sum = tf.get_variable(
      'x_sum', shape=[nb_dims], initializer=tf.zeros_initializer(), trainable=False)
    x_sum_sq = tf.get_variable(
      'x_sum_sq', shape=[nb_dims], initializer=tf.zeros_initializer(), trainable=False)
    x_cnt = tf.get_variable(
      'x_cnt', shape=[], initializer=tf.zeros_initializer(), trainable=False)

    # update statistics with a mini-batch of samples
    self.x_new = tf.placeholder(tf.float32, shape=[None, nb_dims], name='x_new')
    self.updt_ops = [
      x_sum.assign_add(tf.reduce_sum(self.x_new, axis=0)),
      x_sum_sq.assign_add(tf.reduce_sum(tf.square(self.x_new), axis=0)),
      x_cnt.assign_add(tf.cast(tf.shape(self.x_new)[0], tf.float32))
    ]
    tf.summary.scalar('x_cnt', x_cnt)

    # compute running mean & standard deviation from statistics
    # Note: use default values if no samples have been added
    self.mean = tf.cond(x_cnt < 0.5, lambda: tf.zeros(shape=[nb_dims]), lambda: x_sum / x_cnt)
    self.std = tf.cond(
      x_cnt < 0.5, lambda: tf.ones(shape=[nb_dims]),
      lambda: tf.sqrt(tf.maximum(x_sum_sq / x_cnt - tf.square(self.mean), FLAGS.ddpg_rms_eps)))

  def updt(self, x_new):
    """Update running averages with a list of samples.

    Args:
    * x_new: np.array of list of samples (N x D)
    """

    self.sess.run(self.updt_ops, feed_dict={self.x_new: x_new})
