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
"""Reinforcement learning helper for the weight sparsification learner."""

import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

class RLHelper(object):
  """Reinforcement learning helper for the weight sparsification learner."""

  def __init__(self, sess, maskable_vars, skip_head_n_tail):
    """Constructor function.

    Args:
    * sess: TensorFlow session
    * maskable_vars: list of maskable variables
    * skip_head_n_tail: whether to skip the head & tail layers
    """

    # obtain the shape & # of parameters of each maskable variable
    nb_vars = len(maskable_vars)
    var_shapes = []
    self.prune_ratios = np.zeros(nb_vars)
    self.nb_params_full = np.zeros(nb_vars)
    for idx, var in enumerate(maskable_vars):
      var_shape = sess.run(tf.shape(var))
      assert var_shape.size in [2, 4], '# of variable dimensions is %d (invalid)' % var_shape.size
      if var_shape.size == 2:
        var_shape = np.hstack((np.ones(2), var_shape))
      var_shapes += [var_shape]
      self.nb_params_full[idx] = np.prod(var_shape)

    # construct the state vector of each maskable variable
    self.s_dims = nb_vars + 4 + 3  # 4 for shape & 3 for nb_params_curr/pre/post
    self.states = np.zeros((nb_vars, self.s_dims))
    for idx, var in enumerate(maskable_vars):
      state = np.zeros(self.s_dims)
      state[idx] = 1.0
      state[nb_vars:nb_vars + 4] = var_shapes[idx]
      state[nb_vars + 4] = self.nb_params_full[idx]
      state[nb_vars + 6] = np.sum(self.nb_params_full[idx + 1:])
      self.states[idx] = state  # second to the last column will be set dynamically in calc_state()
    self.state_normalizer = np.max(self.states, axis=0)
    self.state_normalizer[-2] = self.state_normalizer[-1]

    # obtain the minimal & maximal pruning ratios of each maskable variable
    prune_ratio_min = max(0.0, 1.0 - (1.0 - FLAGS.ws_prune_ratio) * 3.0)
    prune_ratio_max = 1.0 - (1.0 - FLAGS.ws_prune_ratio) / 3.0
    self.prune_ratios_min = prune_ratio_min * np.ones(nb_vars)
    self.prune_ratios_max = prune_ratio_max * np.ones(nb_vars)
    if skip_head_n_tail:
      self.prune_ratios_min[0] = 0.0
      self.prune_ratios_max[0] = 0.0
      self.prune_ratios_min[-1] = 0.0
      self.prune_ratios_max[-1] = 0.0

  def calc_state(self, idx):
    """Calculate the state vector for the chosen maskable variable.

    Args:
    * idx: index to the chosen maskable variable

    Returns:
    * state: state vector of the chosen maskable variable
    """

    state = np.copy(self.states[idx])
    state[-2] = np.sum(self.nb_params_full[:idx] * (1.0 - self.prune_ratios[:idx]))
    state /= self.state_normalizer

    return state[None, :]

  def calc_reward(self, accuracy):
    """Calculate the reward.

    Args:
    * accuracy: classification accuracy after applying masks

    Returns:
    * reward: reward function's value
    """

    if FLAGS.ws_reward_type == 'single-obj':
      reward = accuracy
    elif FLAGS.ws_reward_type == 'multi-obj':
      prune_ratio = self.calc_overall_prune_ratio()
      reward = accuracy * np.log(1.0 + prune_ratio)
    else:
      raise ValueError('unrecognized reward type: ' + FLAGS.ws_reward_type)

    return reward

  def cvt_action_to_prune_ratio(self, idx, action):
    """Convert action to pruning ratio for the chosen maskable variable.

    Args:
    * idx: index to the chosen maskable variable
    * action: action's value

    Returns:
    * prune_ratio: pruning ratio of the chosen maskable variable
    """

    # piecewise-linear conversion from RL action to pruning ratio
    pr_min, pr_max = self.__calc_prune_ratio_min_max(idx)
    if action > 0.5:
      prune_ratio = pr_max - (1.0 - action) / 0.5 * (pr_max - FLAGS.ws_prune_ratio)
    else:
      prune_ratio = pr_min + (action - 0.0) / 0.5 * (FLAGS.ws_prune_ratio - pr_min)
    self.prune_ratios[idx] = max(pr_min, min(pr_max, prune_ratio))

    return self.prune_ratios[idx]

  def calc_overall_prune_ratio(self):
    """Calculate the overall pruning ratio.

    Returns:
    * prune_ratio: overall pruning ratio
    """

    return np.sum(self.nb_params_full * self.prune_ratios) / np.sum(self.nb_params_full)

  def __calc_prune_ratio_min_max(self, idx):
    """Calculate the minimal & maximal pruning ratio for the chosen maskable variable.

    Args:
    * idx: index to the chosen maskable variable

    Returns:
    * prune_ratio_min: minimal pruning ratio of the chosen maskable variable
    * prune_ratio_max: maximal pruning ratio of the chosen maskable variable
    """

    prune_ratio_min = self.prune_ratios_min[idx]
    prune_ratio_max = self.prune_ratios_max[idx]
    if FLAGS.ws_reward_type == 'single-obj':
      nb_params_pruned_max = np.sum(self.nb_params_full[:idx] * self.prune_ratios[:idx]) \
        + np.sum(self.nb_params_full[idx + 1:] * self.prune_ratios_max[idx + 1:])
      nb_params_pruned_min = np.sum(self.nb_params_full) * FLAGS.ws_prune_ratio
      prune_ratio_req = (nb_params_pruned_min - nb_params_pruned_max) / self.nb_params_full[idx]
      assert prune_ratio_req < prune_ratio_max + 1e-4, \
        'cannot reach the required pruning ratio: %f vs. %f' % (prune_ratio_req, prune_ratio_max)
      prune_ratio_min = max(prune_ratio_min, prune_ratio_req)

    return prune_ratio_min, prune_ratio_max
