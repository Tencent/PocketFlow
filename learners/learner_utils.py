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
"""Utility function for creating the specified learner."""

import tensorflow as tf

from learners.full_precision.learner import FullPrecLearner
from learners.weight_sparsification.learner import WeightSparseLearner
from learners.channel_pruning.learner import ChannelPrunedLearner
from learners.channel_pruning_gpu.learner import ChannelPrunedGpuLearner
from learners.channel_pruning_rmt.learner import ChannelPrunedRmtLearner
from learners.discr_channel_pruning.learner import DisChnPrunedLearner
from learners.uniform_quantization.learner import UniformQuantLearner
from learners.uniform_quantization_tf.learner import UniformQuantTFLearner
from learners.nonuniform_quantization.learner import NonUniformQuantLearner

FLAGS = tf.app.flags.FLAGS

def create_learner(sm_writer, model_helper):
  """Create the learner as specified by FLAGS.learner.

  Args:
  * sm_writer: TensorFlow's summary writer
  * model_helper: model helper with definitions of model & dataset

  Returns:
  * learner: the specified learner
  """

  learner = None
  if FLAGS.learner == 'full-prec':
    learner = FullPrecLearner(sm_writer, model_helper)
  elif FLAGS.learner == 'weight-sparse':
    learner = WeightSparseLearner(sm_writer, model_helper)
  elif FLAGS.learner == 'channel':
    learner = ChannelPrunedLearner(sm_writer, model_helper)
  elif FLAGS.learner == 'chn-pruned-gpu':
    learner = ChannelPrunedGpuLearner(sm_writer, model_helper)
  elif FLAGS.learner == 'chn-pruned-rmt':
    learner = ChannelPrunedRmtLearner(sm_writer, model_helper)
  elif FLAGS.learner == 'dis-chn-pruned':
    learner = DisChnPrunedLearner(sm_writer, model_helper)
  elif FLAGS.learner == 'uniform':
    learner = UniformQuantLearner(sm_writer, model_helper)
  elif FLAGS.learner == 'uniform-tf':
    learner = UniformQuantTFLearner(sm_writer, model_helper)
  elif FLAGS.learner == 'non-uniform':
    learner = NonUniformQuantLearner(sm_writer, model_helper)
  else:
    raise ValueError('unrecognized learner\'s name: ' + FLAGS.learner)

  return learner
