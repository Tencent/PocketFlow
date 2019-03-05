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
"""Abstract class for model helpers."""

from abc import ABC
from abc import abstractmethod

class AbstractModelHelper(ABC):
  """Abstract class for model helpers.

  A model helper should define the following function interface:
    1. Data input pipeline for training and evaluation subsets.
    2. Network's forward pass during training and evaluation.
    3. Loss function (and some extra evaluation metrics).

  All functions marked with "@abstractmethod" must be explicitly implemented in the sub-class.
  """

  def __init__(self, data_format, forward_w_labels=False):
    """Constructor function.

    Note: DO NOT create any TF operations here!!!

    Args:
    * data_format: data format ('channels_last' OR 'channels_first')
    """

    self.data_format = data_format
    self.forward_w_labels = forward_w_labels

  @abstractmethod
  def build_dataset_train(self, enbl_trn_val_split):
    """Build the data subset for training, usually with data augmentation.

    Args:
    * enbl_trn_val_split: enable the training & validation splitting

    Returns:
    * iterator_trn: iterator for the training subset
    * iterator_val: iterator for the validation subset
      OR
    * iterator: iterator for the training subset
    """
    pass

  @abstractmethod
  def build_dataset_eval(self):
    """Build the data subset for evaluation, usually without data augmentation.

    Returns:
    * iterator: iterator over the evaluation subset
    """
    pass

  @abstractmethod
  def forward_train(self, inputs, labels=None):
    """Forward computation at training.

    Args:
    * inputs: inputs to the network's forward pass
    * labels: ground-truth labels

    Returns:
    * outputs: outputs from the network's forward pass
    """
    pass

  @abstractmethod
  def forward_eval(self, inputs):
    """Forward computation at evaluation.

    Args:
    * inputs: inputs to the network's forward pass

    Returns:
    * outputs: outputs from the network's forward pass
    """
    pass

  @abstractmethod
  def calc_loss(self, labels, outputs, trainable_vars):
    """Calculate loss (and some extra evaluation metrics).

    Args:
    * labels: ground-truth labels
    * outputs: outputs from the network's forward pass
    * trainable_vars: list of trainable variables

    Returns:
    * loss: loss function's value
    * metrics: dictionary of extra evaluation metrics
    """
    pass

  @abstractmethod
  def setup_lrn_rate(self, global_step):
    """Setup the learning rate (and number of training iterations).

    Args:
    * global_step: training iteration counter

    Returns:
    * lrn_rate: learning rate
    * nb_iters: number of training iterations
    """
    pass

  def warm_start(self, sess):
    """Initialize the model for warm-start.

    Args:
    * sess: TensorFlow session
    """
    pass

  def dump_n_eval(self, outputs, action):
    """Dump the model's outputs to files and evaluate.

    Args:
    * outputs: outputs from the network's forward pass
    * action: 'init' | 'dump' | 'eval'
    """
    pass

  @property
  @abstractmethod
  def model_name(self):
    """Model's name."""
    pass

  @property
  @abstractmethod
  def dataset_name(self):
    """Dataset's name."""
    pass
