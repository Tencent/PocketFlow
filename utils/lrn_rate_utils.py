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

from utils.multi_gpu_wrapper import MultiGpuWrapper as mgw

FLAGS = tf.app.flags.FLAGS

# set <nb_epochs_rat> to values smaller than 1.0 to use fewer epochs and speed up training
tf.app.flags.DEFINE_float('nb_epochs_rat', 1.0, '# of training epochs\'s ratio')

def calc_nb_batches(nb_epochs, batch_size):
  """Calculate the number of mini-batches.

  Args:
  * nb_epochs: number of epoches
  * batch_size: number of samples in each mini-batch

  Returns:
  * nb_batches: number of mini-batches
  """

  return int(FLAGS.nb_smpls_train * nb_epochs * FLAGS.nb_epochs_rat / batch_size)

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

  # adjust the interval endpoints
  idxs_epoch = [idx_epoch * FLAGS.nb_epochs_rat for idx_epoch in idxs_epoch]

  # setup learning rate with the piecewise constant strategy
  lrn_rate_init = FLAGS.lrn_rate_init * batch_size / FLAGS.batch_size_norm
  nb_batches_per_epoch = float(FLAGS.nb_smpls_train) / batch_size
  bnds = [int(nb_batches_per_epoch * idx_epoch) for idx_epoch in idxs_epoch]
  vals = [lrn_rate_init * decay_rate for decay_rate in decay_rates]

  return tf.train.piecewise_constant(global_step, bnds, vals)

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

  # adjust the step size & decaying rate
  epoch_step *= FLAGS.nb_epochs_rat

  # setup learning rate with the exponential decay strategy
  lrn_rate_init = FLAGS.lrn_rate_init * batch_size / FLAGS.batch_size_norm
  batch_step = int(FLAGS.nb_smpls_train * epoch_step / batch_size)

  return tf.train.exponential_decay(
    lrn_rate_init, global_step, batch_step, decay_rate, staircase=True)

def setup_lrn_rate_lenet_cifar10(global_step, batch_size):
  """Setup the learning rate for LeNet-like models on the CIFAR-10 dataset.

  Args:
  * global_step: training iteration counter
  * batch_size: number of samples in each mini-batch

  Returns:
  * lrn_rate: learning rate
  * nb_batches: number of mini-batches
  """

  nb_epochs = 250
  idxs_epoch = [100, 150, 200]
  decay_rates = [1.0, 0.1, 0.01, 0.001]
  lrn_rate = setup_lrn_rate_piecewise_constant(global_step, batch_size, idxs_epoch, decay_rates)
  nb_batches = calc_nb_batches(nb_epochs, batch_size)

  return lrn_rate, nb_batches

def setup_lrn_rate_resnet_cifar10(global_step, batch_size):
  """Setup the learning rate for ResNet models on the CIFAR-10 dataset.

  Args:
  * global_step: training iteration counter
  * batch_size: number of samples in each mini-batch

  Returns:
  * lrn_rate: learning rate
  * nb_batches: number of mini-batches
  """

  nb_epochs = 250
  idxs_epoch = [100, 150, 200]
  decay_rates = [1.0, 0.1, 0.01, 0.001]
  lrn_rate = setup_lrn_rate_piecewise_constant(global_step, batch_size, idxs_epoch, decay_rates)
  nb_batches = calc_nb_batches(nb_epochs, batch_size)

  return lrn_rate, nb_batches

def setup_lrn_rate_resnet_ilsvrc12(global_step, batch_size):
  """Setup the learning rate for ResNet models on the ILSVRC-12 dataset.

  Args:
  * global_step: training iteration counter
  * batch_size: number of samples in each mini-batch

  Returns:
  * lrn_rate: learning rate
  * nb_batches: number of mini-batches
  """

  nb_epochs = 100
  idxs_epoch = [30, 60, 80, 90]
  decay_rates = [1.0, 0.1, 0.01, 0.001, 0.0001]
  lrn_rate = setup_lrn_rate_piecewise_constant(global_step, batch_size, idxs_epoch, decay_rates)
  nb_batches = calc_nb_batches(nb_epochs, batch_size)

  return lrn_rate, nb_batches

def setup_lrn_rate_mobilenet_v1_ilsvrc12(global_step, batch_size):
  """Setup the learning rate for MobileNet-v1 models on the ILSVRC-12 dataset.

  Args:
  * global_step: training iteration counter
  * batch_size: number of samples in each mini-batch

  Returns:
  * lrn_rate: learning rate
  * nb_batches: number of mini-batches
  """

  nb_epochs = 100
  idxs_epoch = [30, 60, 80, 90]
  decay_rates = [1.0, 0.1, 0.01, 0.001, 0.0001]
  lrn_rate = setup_lrn_rate_piecewise_constant(global_step, batch_size, idxs_epoch, decay_rates)
  nb_batches = calc_nb_batches(nb_epochs, batch_size)

  return lrn_rate, nb_batches

def setup_lrn_rate_mobilenet_v2_ilsvrc12(global_step, batch_size):
  """Setup the learning rate for MobileNet-v2 models on the ILSVRC-12 dataset.

  Args:
  * global_step: training iteration counter
  * batch_size: number of samples in each mini-batch

  Returns:
  * lrn_rate: learning rate
  * nb_batches: number of mini-batches
  """

  nb_epochs = 412
  epoch_step = 2.5
  decay_rate = 0.98 ** epoch_step
  lrn_rate = setup_lrn_rate_exponential_decay(global_step, batch_size, epoch_step, decay_rate)
  nb_batches = calc_nb_batches(nb_epochs, batch_size)

  return lrn_rate, nb_batches

def setup_lrn_rate(global_step, model_name, dataset_name):
  """Setup the learning rate for the given dataset.

  Args:
  * global_step: training iteration counter
  * model_name: model's name; must be one of ['lenet', 'resnet_*', 'mobilenet_v1', 'mobilenet_v2']
  * dataset_name: dataset's name; must be one of ['cifar_10', 'ilsvrc_12']

  Returns:
  * lrn_rate: learning rate
  * nb_batches: number of training mini-batches
  """

  # obtain the overall batch size across all GPUs
  if not FLAGS.enbl_multi_gpu:
    batch_size = FLAGS.batch_size
  else:
    batch_size = FLAGS.batch_size * mgw.size()

  # choose a learning rate protocol according to the model & dataset combination
  global_step = tf.cast(global_step, tf.int32)
  if dataset_name == 'cifar_10':
    if model_name == 'lenet':
      lrn_rate, nb_batches = setup_lrn_rate_lenet_cifar10(global_step, batch_size)
    elif model_name.startswith('resnet'):
      lrn_rate, nb_batches = setup_lrn_rate_resnet_cifar10(global_step, batch_size)
    else:
      raise NotImplementedError('model: {} / dataset: {}'.format(model_name, dataset_name))
  elif dataset_name == 'ilsvrc_12':
    if model_name.startswith('resnet'):
      lrn_rate, nb_batches = setup_lrn_rate_resnet_ilsvrc12(global_step, batch_size)
    elif model_name.startswith('mobilenet_v1'):
      lrn_rate, nb_batches = setup_lrn_rate_mobilenet_v1_ilsvrc12(global_step, batch_size)
    elif model_name.startswith('mobilenet_v2'):
      lrn_rate, nb_batches = setup_lrn_rate_mobilenet_v2_ilsvrc12(global_step, batch_size)
    else:
      raise NotImplementedError('model: {} / dataset: {}'.format(model_name, dataset_name))
  else:
    raise NotImplementedError('dataset: ' + dataset_name)

  return lrn_rate, nb_batches
