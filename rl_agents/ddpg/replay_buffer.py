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
"""Replay buffer for storing state-action-reward transitions."""

import numpy as np

class ReplayBuffer(object):
  """Replay buffer for storing state-action-reward transitions.

  Each transition consists of five components:
  1. state: current state
  2. action: action under the current state
  3. reward: reward for the chosen action under the current state
  4. terminal: whether the terminal state is reached after the chosen action
  5. state_next: next state
  """

  def __init__(self, s_dims, a_dims, buf_size):
    """Constructor function.

    Args:
    * s_dims: number of state vector's dimensions
    * a_dims: number of action vector's dimensions
    * buf_size: maximal number of transitions to be stored
    """

    self.s_dims = s_dims
    self.a_dims = a_dims
    self.buf_size = buf_size

    # initialize the buffer & counters
    self.idx_smpl = 0
    self.nb_smpls = 0  # number of valid samples in the buffer
    self.buffers = {
      'states': np.zeros((self.buf_size, self.s_dims), dtype=np.float32),
      'actions': np.zeros((self.buf_size, self.a_dims), dtype=np.float32),
      'rewards': np.zeros((self.buf_size, 1), dtype=np.float32),
      'terminals': np.zeros((self.buf_size, 1), dtype=np.float32),
      'states_next': np.zeros((self.buf_size, self.s_dims), dtype=np.float32),
    }

  def reset(self):
    """Reset the replay buffer (all transitions are removed)."""

    self.idx_smpl = 0
    self.nb_smpls = 0

  def is_ready(self):
    """Check whether the replay buffer is ready for sampling transitions.

    Returns:
    * is_ready: True if the replay buffer is ready
    """

    return self.nb_smpls == self.buf_size

  def append(self, states, actions, rewards, terminals, states_next):
    """Append multiple transitions into the replay buffer.

    Args:
    * states: np.array of current state vectors
    * actions: np.array of action vectors
    * rewards: np.array of reward scalars
    * terminals: np.array of terminal scalars (1: terminal; 0: non-terminal)
    * states_next: np.array of next state vectors
    """

    # pack transactions into a mini-batch
    nb_smpls = states.shape[0]
    mbatch = {
      'states': states,
      'actions': actions,
      'rewards': rewards,
      'terminals': terminals,
      'states_next': states_next,
    }

    # insert samples into the buffer
    if self.idx_smpl + nb_smpls <= self.buf_size:
      idxs = np.arange(self.idx_smpl, self.idx_smpl + nb_smpls)
      for key in self.buffers:
        self.buffers[key][idxs] = mbatch[key]
    else:
      nb_smpls_tail = self.buf_size - self.idx_smpl  # samples to be added to the buffer tail
      nb_smpls_head = nb_smpls - nb_smpls_tail  # samples to be added to the buffer head
      for key in self.buffers:
        self.buffers[key][self.idx_smpl:] = mbatch[key][:nb_smpls_tail]
        self.buffers[key][:nb_smpls_head] = mbatch[key][nb_smpls_tail:]

    # update counters
    self.idx_smpl = (self.idx_smpl + nb_smpls) % self.buf_size
    self.nb_smpls = min(self.nb_smpls + nb_smpls, self.buf_size)

  def sample(self, batch_size):
    """Sample a mini-batch of trasitions.

    Args:
    * batch_size: number of transitions in the mini-batch

    Returns:
    * mbatch: a mini-batch of transitions
    """

    idxs_smpl = np.random.randint(0, self.nb_smpls, batch_size)
    mbatch = {key: self.buffers[key][idxs_smpl] for key in self.buffers}

    return mbatch
