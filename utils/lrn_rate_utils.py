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
"""Utility functions for learning rates."""

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def setup_lrn_rate_piecewise_constant(global_step, batch_size, idxs_epoch, decay_rates):
  """Setup the learning rate with piecewise constant strategy.

  Args:
  * global_step: training iteration counter
  * batch_size: number of samples in each mini-batch
  * idxs_epoch: indices of epoches to decay the learning rate
  * decay_rates: list of decaying rates

  Returns:
  * lrn_rate: learning rate
  """

  # adjust interval endpoints w.r.t. FLAGS.nb_epochs_rat
  idxs_epoch = [idx_epoch * FLAGS.nb_epochs_rat for idx_epoch in idxs_epoch]

  # setup learning rate with the piecewise constant strategy
  lrn_rate_init = FLAGS.lrn_rate_init * batch_size / FLAGS.batch_size_norm
  nb_batches_per_epoch = float(FLAGS.nb_smpls_train) / batch_size
  bnds = [int(nb_batches_per_epoch * idx_epoch) for idx_epoch in idxs_epoch]
  vals = [lrn_rate_init * decay_rate for decay_rate in decay_rates]
  lrn_rate = tf.train.piecewise_constant(global_step, bnds, vals)

  return lrn_rate

def setup_lrn_rate_exponential_decay(global_step, batch_size, epoch_step, decay_rate):
  """Setup the learning rate with exponential decaying strategy.

  Args:
  * global_step: training iteration counter
  * batch_size: number of samples in each mini-batch
  * epoch_step: epoch step-size for applying the decaying step
  * decay_rate: decaying rate

  Returns:
  * lrn_rate: learning rate
  """

  # adjust the step size & decaying rate w.r.t. FLAGS.nb_epochs_rat
  epoch_step *= FLAGS.nb_epochs_rat

  # setup learning rate with the exponential decay strategy
  lrn_rate_init = FLAGS.lrn_rate_init * batch_size / FLAGS.batch_size_norm
  batch_step = int(FLAGS.nb_smpls_train * epoch_step / batch_size)
  lrn_rate = tf.train.exponential_decay(
    lrn_rate_init, tf.cast(global_step, tf.int32), batch_step, decay_rate, staircase=True)

  return lrn_rate
