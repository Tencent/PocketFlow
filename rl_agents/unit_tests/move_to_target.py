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
"""Unit-test for the Move-to-target problem."""

import traceback
import numpy as np
from numpy.linalg import norm
import tensorflow as tf

from rl_agents.ddpg.agent import Agent as DdpgAgent

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', './logs', 'logging directory')
tf.app.flags.DEFINE_integer('nb_dims', 4, '# of state & action dimensions')
tf.app.flags.DEFINE_integer('nb_rlouts', 200, '# of roll-outs')
tf.app.flags.DEFINE_integer('nb_rlouts_eval', 100, '# of roll-outs for evaluation')
tf.app.flags.DEFINE_integer('rlout_len', 200, 'roll-out\'s length')

class Env(object):
  """Environment for the Move-to-target problem.

  The reward for moving from <x_curr> to <x_next> is defined as:
    reward := Dist(x_curr, target) - Dist(x_next, target) - Dist(x_curr, x_next)
  The optimal sum of reward is zero.
  """

  def __init__(self):
    """Constructor function."""

    self.x_lbnd = -10.0
    self.x_ubnd = 10.0
    self.x_curr = None
    self.target = np.zeros((1, FLAGS.nb_dims))

  def reset(self):
    """Reset the current position."""

    self.x_curr = np.random.uniform(self.x_lbnd, self.x_ubnd, (1, FLAGS.nb_dims))

    return self.x_curr

  def step(self, action):
    """Move to the next position."""

    x_next = self.x_curr + action
    reward = norm(self.x_curr - self.target) \
      - norm(x_next - self.target) - norm(self.x_curr - x_next)
    self.x_curr = x_next

    return self.x_curr, reward * np.ones((1, 1))

def build_env_n_agent(sess):
  """Build the environment and an RL agent to solve it.

  Args:
  * sess: TensorFlow session

  Returns:
  * env: environment
  * agent: RL agent
  """

  env = Env()
  s_dims = FLAGS.nb_dims
  a_dims = FLAGS.nb_dims
  nb_rlouts = FLAGS.nb_rlouts
  buf_size = int(FLAGS.rlout_len * nb_rlouts * 0.25)
  a_lbnd = -1.0
  a_ubnd = 1.0
  agent = DdpgAgent(sess, s_dims, a_dims, nb_rlouts, buf_size, a_lbnd, a_ubnd)

  return env, agent

def train_agent(sess, env, agent):
  """Train the RL agent through multiple roll-outs.

  Args:
  * sess: TensorFlow session
  * env: environment
  * agent: RL agent
  """

  agent.init()
  for idx_rlout in range(FLAGS.nb_rlouts):
    agent.init_rlout()
    state = env.reset()
    rewards = np.zeros(FLAGS.rlout_len)
    tf.logging.info('initial state: {}'.format(state))
    for idx_iter in range(FLAGS.rlout_len):
      action = sess.run(agent.actions_noisy, feed_dict={agent.states: state})
      state_next, reward = env.step(action)
      terminal = np.ones((1, 1)) if (idx_iter == FLAGS.rlout_len - 1) else np.zeros((1, 1))
      agent.record(state, action, reward, terminal, state_next)
      actor_loss, critic_loss, noise_std = agent.train()
      state = state_next
      rewards[idx_iter] = reward
    agent.finalize_rlout(rewards)
    tf.logging.info('terminal state: {}'.format(state))
    tf.logging.info('roll-out #%d: reward (ave.): %.2e' % (idx_rlout, np.mean(rewards)))
    tf.logging.info('roll-out #%d: a-loss = %.2e | c-loss = %.2e | noise std. = %.2e'
                    % (idx_rlout, actor_loss, critic_loss, noise_std))

def eval_agent(sess, env, agent):
  """Evaluate the RL agent through multiple roll-outs.

  Args:
  * sess: TensorFlow session
  * env: environment
  * agent: RL agent
  """

  reward_ave_list = []
  for idx_rlout in range(FLAGS.nb_rlouts_eval):
    state = env.reset()
    rewards = np.zeros(FLAGS.rlout_len)
    tf.logging.info('initial state: {}'.format(state))
    for idx_iter in range(FLAGS.rlout_len):
      action = sess.run(agent.actions_clean, feed_dict={agent.states: state})
      state, reward = env.step(action)
      rewards[idx_iter] = reward
    tf.logging.info('terminal state: {}'.format(state))
    tf.logging.info('roll-out #%d: reward (ave.): %.2e' % (idx_rlout, np.mean(rewards)))
    reward_ave_list += [np.mean(rewards)]
  tf.logging.info('[EVAL] reward (ave.): %.4e' % np.mean(np.array(reward_ave_list)))

def main(unused_argv):
  """Main entry."""

  try:
    # setup the TF logging routine
    tf.logging.set_verbosity(tf.logging.INFO)

    # display FLAGS's values
    tf.logging.info('FLAGS:')
    for key, value in FLAGS.flag_values_dict().items():
      tf.logging.info('{}: {}'.format(key, value))

    # build the environment & agent
    sess = tf.Session()
    env, agent = build_env_n_agent(sess)

    # train & evaluate the RL agent
    train_agent(sess, env, agent)
    eval_agent(sess, env, agent)

    # exit normally
    return 0
  except ValueError:
    traceback.print_exc()
    return 1  # exit with errors

if __name__ == '__main__':
  tf.app.run()
