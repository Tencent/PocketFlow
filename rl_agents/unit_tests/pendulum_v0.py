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
"""Unit-test for the Pendulum-v0 problem."""

import traceback
import numpy as np
import tensorflow as tf
import gym

from rl_agents.ddpg.agent import Agent as DdpgAgent

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', './logs', 'logging directory')
tf.app.flags.DEFINE_integer('nb_rlouts', 200, '# of roll-outs')
tf.app.flags.DEFINE_integer('nb_rlouts_eval', 100, '# of roll-outs for evaluation')
tf.app.flags.DEFINE_integer('rlout_len', 200, 'roll-out\'s length')

def build_env_n_agent(sess):
  """Build the environment and an RL agent to solve it.

  Args:
  * sess: TensorFlow session

  Returns:
  * env: environment
  * agent: RL agent
  """

  env = gym.make('Pendulum-v0')
  s_dims = env.observation_space.shape[-1]
  a_dims = env.action_space.shape[-1]
  buf_size = int(FLAGS.rlout_len * FLAGS.nb_rlouts * 0.25)
  a_lbnd = env.action_space.low[0]
  a_ubnd = env.action_space.high[0]
  agent = DdpgAgent(sess, s_dims, a_dims, FLAGS.nb_rlouts, buf_size, a_lbnd, a_ubnd)
  tf.logging.info('s_dims = %d, a_dims = %d' % (s_dims, a_dims))
  tf.logging.info('a_lbnd = %f, a_ubnd = %f' % (a_lbnd, a_ubnd))

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
    for idx_iter in range(FLAGS.rlout_len):
      action = sess.run(agent.actions_noisy, feed_dict={agent.states: state[None, :]})
      state_next, reward, __, __ = env.step(action.ravel())
      terminal = np.ones((1, 1)) if idx_iter == FLAGS.rlout_len - 1 else np.zeros((1, 1))
      agent.record(state[None, :], action,
                   reward * np.ones((1, 1)), terminal, state_next[None, :])
      actor_loss, critic_loss, noise_std = agent.train()
      state = state_next
      rewards[idx_iter] = reward
    agent.finalize_rlout(rewards)
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
    for idx_iter in range(FLAGS.rlout_len):
      action = sess.run(agent.actions_clean, feed_dict={agent.states: state[None, :]})
      state, reward, __, __ = env.step(action.ravel())
      rewards[idx_iter] = reward
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

    # train and evaluate the RL agent
    train_agent(sess, env, agent)
    eval_agent(sess, env, agent)

    # exit normally
    return 0
  except ValueError:
    traceback.print_exc()
    return 1  # exit with errors

if __name__ == '__main__':
  tf.app.run()
