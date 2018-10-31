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
"""Parameter & action noise's specifications."""

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('ddpg_noise_type', 'param',
                           'DDPG: noise type (\'action\' OR \'param\')')
tf.app.flags.DEFINE_string('ddpg_noise_prtl', 'tdecy',
                           'DDPG: noise adjustment protocol (\'adapt\' OR \'tdecy\')')
tf.app.flags.DEFINE_float('ddpg_noise_std_init', 1e+0,
                          'DDPG: parameter / action noise\'s initial stdev.')

# for <AdaptiveNoiseSpec> only
tf.app.flags.DEFINE_float('ddpg_noise_dst_finl', 1e-2, 'DDPG: action noise\'s final distance')
tf.app.flags.DEFINE_float('ddpg_noise_adpt_rat', 1.03, 'DDPG: parameter noise\'s adaption rate')

# for <TimeDecayNoiseSpec> only
tf.app.flags.DEFINE_float('ddpg_noise_std_finl', 1e-5,
                          'DDPG: parameter / action noise\'s final stdev.')

class AdaptiveNoiseSpec(object):
  """Adaptive parameter noise's specifications.

  To enable, set <ddpg_noise_type> to 'param' and <ddpg_noise_prtl> to 'adapt'.
  """

  def __init__(self):
    """Constructor function."""

    self.stdev_curr = FLAGS.ddpg_noise_std_init

  def reset(self):
    """Reset the standard deviation."""

    self.stdev_curr = FLAGS.ddpg_noise_std_init

  def adapt(self, dst_curr):
    """Adjust the standard deviation to meet the distance requirement between actions.

    Args:
    * dst_curr: current distance between clean & distorted actions
    """

    if dst_curr > FLAGS.ddpg_noise_dst_finl:
      self.stdev_curr /= FLAGS.ddpg_noise_adpt_rat
    else:
      self.stdev_curr *= FLAGS.ddpg_noise_adpt_rat

class TimeDecayNoiseSpec(object):
  """Time-decaying action / parameter noise's specifications.

  To enable, set <ddpg_noise_type> to 'action' / 'param' and <ddpg_noise_prtl> to 'tdecy'.
  """

  def __init__(self, nb_rlouts):
    """Constructor function."""

    self.stdev_curr = FLAGS.ddpg_noise_std_init
    self.decy_rat = (FLAGS.ddpg_noise_std_finl / FLAGS.ddpg_noise_std_init) ** (1.0 / nb_rlouts)

  def reset(self):
    """Reset the standard deviation."""

    self.stdev_curr = FLAGS.ddpg_noise_std_init

  def adapt(self):
    """Adjust the standard deviation by multiplying with the decaying ratio."""

    self.stdev_curr *= self.decy_rat
