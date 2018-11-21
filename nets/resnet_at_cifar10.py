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
"""Model helper for creating a ResNet model for the CIFAR-10 dataset."""

import tensorflow as tf

from nets.abstract_model_helper import AbstractModelHelper
from datasets.cifar10_dataset import Cifar10Dataset
from utils.external import resnet_model as ResNet
from utils.lrn_rate_utils import setup_lrn_rate_piecewise_constant
from utils.multi_gpu_wrapper import MultiGpuWrapper as mgw

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('resnet_size', 20, '# of layers in the ResNet model')
tf.app.flags.DEFINE_float('nb_epochs_rat', 1.0, '# of training epochs\'s ratio')
tf.app.flags.DEFINE_float('lrn_rate_init', 1e-1, 'initial learning rate')
tf.app.flags.DEFINE_float('batch_size_norm', 128, 'normalization factor of batch size')
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum coefficient')
tf.app.flags.DEFINE_float('loss_w_dcy', 2e-4, 'weight decaying loss\'s coefficient')

def forward_fn(inputs, is_train, data_format):
  """Forward pass function.

  Args:
  * inputs: inputs to the network's forward pass
  * is_train: whether to use the forward pass with training operations inserted
  * data_format: data format ('channels_last' OR 'channels_first')

  Returns:
  * inputs: outputs from the network's forward pass
  """

  # setup hyper-parameters
  nb_blocks = (FLAGS.resnet_size - 2) // 6
  bottleneck = False
  nb_classes = FLAGS.nb_classes
  nb_filters = 16
  kernel_size = 3
  conv_stride = 1
  first_pool_size = None
  first_pool_stride = None
  block_sizes = [nb_blocks] * 3
  block_strides = [1, 2, 2]

  # model definition
  model = ResNet.Model(
    FLAGS.resnet_size, bottleneck, nb_classes, nb_filters, kernel_size, conv_stride,
    first_pool_size, first_pool_stride, block_sizes, block_strides, data_format=data_format)
  inputs = model(inputs, is_train)

  return inputs

class ModelHelper(AbstractModelHelper):
  """Model helper for creating a ResNet model for the CIFAR-10 dataset."""

  def __init__(self):
    """Constructor function."""

    # class-independent initialization
    super(ModelHelper, self).__init__()

    # initialize training & evaluation subsets
    self.dataset_train = Cifar10Dataset(is_train=True)
    self.dataset_eval = Cifar10Dataset(is_train=False)

  def build_dataset_train(self, enbl_trn_val_split=False):
    """Build the data subset for training, usually with data augmentation."""

    return self.dataset_train.build(enbl_trn_val_split)

  def build_dataset_eval(self):
    """Build the data subset for evaluation, usually without data augmentation."""

    return self.dataset_eval.build()

  def forward_train(self, inputs, data_format='channels_last'):
    """Forward computation at training."""

    return forward_fn(inputs, is_train=True, data_format=data_format)

  def forward_eval(self, inputs, data_format='channels_last'):
    """Forward computation at evaluation."""

    return forward_fn(inputs, is_train=False, data_format=data_format)

  def calc_loss(self, labels, outputs, trainable_vars):
    """Calculate loss (and some extra evaluation metrics)."""

    loss = tf.losses.softmax_cross_entropy(labels, outputs)
    loss_filter = lambda var: 'batch_normalization' not in var.name
    loss += FLAGS.loss_w_dcy \
      * tf.add_n([tf.nn.l2_loss(var) for var in trainable_vars if loss_filter(var)])
    accuracy = tf.reduce_mean(
      tf.cast(tf.equal(tf.argmax(labels, axis=1), tf.argmax(outputs, axis=1)), tf.float32))
    metrics = {'accuracy': accuracy}

    return loss, metrics

  def setup_lrn_rate(self, global_step):
    """Setup the learning rate (and number of training iterations)."""

    nb_epochs = 250
    idxs_epoch = [100, 150, 200]
    decay_rates = [1.0, 0.1, 0.01, 0.001]
    batch_size = FLAGS.batch_size * (1 if not FLAGS.enbl_multi_gpu else mgw.size())
    lrn_rate = setup_lrn_rate_piecewise_constant(global_step, batch_size, idxs_epoch, decay_rates)
    nb_iters = int(FLAGS.nb_smpls_train * nb_epochs * FLAGS.nb_epochs_rat / batch_size)

    return lrn_rate, nb_iters

  @property
  def model_name(self):
    """Model's name."""

    return 'resnet_%d' % FLAGS.resnet_size

  @property
  def dataset_name(self):
    """Dataset's name."""

    return 'cifar_10'
