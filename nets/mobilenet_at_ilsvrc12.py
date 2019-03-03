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
"""Model helper for creating a MobileNet model for the ILSVRC-12 dataset."""

import tensorflow as tf
from tensorflow.contrib import slim

from nets.abstract_model_helper import AbstractModelHelper
from datasets.ilsvrc12_dataset import Ilsvrc12Dataset
from utils.external import mobilenet_v1 as MobileNetV1
from utils.external import mobilenet_v2 as MobileNetV2
from utils.lrn_rate_utils import setup_lrn_rate_piecewise_constant
from utils.lrn_rate_utils import setup_lrn_rate_exponential_decay
from utils.multi_gpu_wrapper import MultiGpuWrapper as mgw

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('mobilenet_version', 1, 'MobileNet\'s version (1 or 2)')
tf.app.flags.DEFINE_float('mobilenet_depth_mult', 1.0, 'MobileNet\'s depth multiplier')
tf.app.flags.DEFINE_float('nb_epochs_rat', 1.0, '# of training epochs\'s ratio')
tf.app.flags.DEFINE_float('lrn_rate_init', 0.045, 'initial learning rate')
tf.app.flags.DEFINE_float('batch_size_norm', 96, 'normalization factor of batch size')
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum coefficient')
tf.app.flags.DEFINE_float('loss_w_dcy', 4e-5, 'weight decaying loss\'s coefficient')

def forward_fn(inputs, is_train):
  """Forward pass function.

  Args:
  * inputs: inputs to the network's forward pass
  * is_train: whether to use the forward pass with training operations inserted

  Returns:
  * outputs: outputs from the network's forward pass
  """

  nb_classes = FLAGS.nb_classes
  depth_mult = FLAGS.mobilenet_depth_mult

  if FLAGS.mobilenet_version == 1:
    scope_fn = MobileNetV1.mobilenet_v1_arg_scope
    with slim.arg_scope(scope_fn(is_training=is_train)): # pylint: disable=not-context-manager
      outputs, __ = MobileNetV1.mobilenet_v1(
        inputs, is_training=is_train, num_classes=nb_classes, depth_multiplier=depth_mult)
  elif FLAGS.mobilenet_version == 2:
    scope_fn = MobileNetV2.training_scope
    with slim.arg_scope(scope_fn(is_training=is_train)): # pylint: disable=not-context-manager
      outputs, __ = MobileNetV2.mobilenet(
        inputs, num_classes=nb_classes, depth_multiplier=depth_mult)
  else:
    raise ValueError('invalid MobileNet version: {}'.format(FLAGS.mobilenet_version))

  return outputs

class ModelHelper(AbstractModelHelper):
  """Model helper for creating a MobileNet model for the ILSVRC-12 dataset."""

  def __init__(self, data_format='channels_last'):
    """Constructor function."""

    # class-independent initialization
    assert data_format == 'channels_last', 'MobileNet only supports \'channels_last\' data format'
    super(ModelHelper, self).__init__(data_format)

    # initialize training & evaluation subsets
    self.dataset_train = Ilsvrc12Dataset(is_train=True)
    self.dataset_eval = Ilsvrc12Dataset(is_train=False)

  def build_dataset_train(self, enbl_trn_val_split=False):
    """Build the data subset for training, usually with data augmentation."""

    return self.dataset_train.build(enbl_trn_val_split)

  def build_dataset_eval(self):
    """Build the data subset for evaluation, usually without data augmentation."""

    return self.dataset_eval.build()

  def forward_train(self, inputs):
    """Forward computation at training."""

    return forward_fn(inputs, is_train=True)

  def forward_eval(self, inputs):
    """Forward computation at evaluation."""

    return forward_fn(inputs, is_train=False)

  def calc_loss(self, labels, outputs, trainable_vars):
    """Calculate loss (and some extra evaluation metrics)."""

    loss = tf.losses.softmax_cross_entropy(labels, outputs)
    loss_filter = lambda var: 'batch_normalization' not in var.name
    loss += FLAGS.loss_w_dcy \
        * tf.add_n([tf.nn.l2_loss(var) for var in trainable_vars if loss_filter(var)])
    targets = tf.argmax(labels, axis=1)
    acc_top1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(outputs, targets, 1), tf.float32))
    acc_top5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(outputs, targets, 5), tf.float32))
    metrics = {'accuracy': acc_top5, 'acc_top1': acc_top1, 'acc_top5': acc_top5}

    return loss, metrics

  def setup_lrn_rate(self, global_step):
    """Setup the learning rate (and number of training iterations)."""

    batch_size = FLAGS.batch_size * (1 if not FLAGS.enbl_multi_gpu else mgw.size())
    if FLAGS.mobilenet_version == 1:
      nb_epochs = 100
      idxs_epoch = [30, 60, 80, 90]
      decay_rates = [1.0, 0.1, 0.01, 0.001, 0.0001]
      lrn_rate = setup_lrn_rate_piecewise_constant(global_step, batch_size, idxs_epoch, decay_rates)
      nb_iters = int(FLAGS.nb_smpls_train * nb_epochs * FLAGS.nb_epochs_rat / batch_size)
    elif FLAGS.mobilenet_version == 2:
      nb_epochs = 412
      epoch_step = 2.5
      decay_rate = 0.98 ** epoch_step  # which is better, 0.98 OR (0.98 ** epoch_step)?
      lrn_rate = setup_lrn_rate_exponential_decay(global_step, batch_size, epoch_step, decay_rate)
      nb_iters = int(FLAGS.nb_smpls_train * nb_epochs * FLAGS.nb_epochs_rat / batch_size)
    else:
      raise ValueError('invalid MobileNet version: {}'.format(FLAGS.mobilenet_version))

    return lrn_rate, nb_iters

  @property
  def model_name(self):
    """Model's name."""

    return 'mobilenet_v%d' % FLAGS.mobilenet_version

  @property
  def dataset_name(self):
    """Dataset's name."""

    return 'ilsvrc_12'
