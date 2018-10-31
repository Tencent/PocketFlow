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
"""DDPG (Deep Deterministic Policy Gradient) agent."""

import numpy as np
import tensorflow as tf

from rl_agents.ddpg.actor_critic import Actor
from rl_agents.ddpg.actor_critic import Critic
from rl_agents.ddpg.noise import AdaptiveNoiseSpec
from rl_agents.ddpg.noise import TimeDecayNoiseSpec
from rl_agents.ddpg.replay_buffer import ReplayBuffer
from rl_agents.ddpg.running_mean_std import RunningMeanStd

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('ddpg_tau', 0.01, 'DDPG: target networks\' update coefficient')
tf.app.flags.DEFINE_float('ddpg_gamma', 0.9, 'DDPG: reward discounting factor')
tf.app.flags.DEFINE_float('ddpg_lrn_rate', 1e-3, 'DDPG: actor & critic networks\' learning rate')
tf.app.flags.DEFINE_float('ddpg_loss_w_dcy', 0.0, 'DDPG: weight decaying coefficient')
tf.app.flags.DEFINE_integer('ddpg_record_step', 1, 'DDPG: recording step size')
tf.app.flags.DEFINE_integer('ddpg_batch_size', 64, 'DDPG: batch size')
tf.app.flags.DEFINE_boolean('ddpg_enbl_bsln_func', True, 'DDPG: enable baseline function')
tf.app.flags.DEFINE_float('ddpg_bsln_decy_rate', 0.95, 'DDPG: baseline function\'s decaying rate')

def normalize(smpl_mat, rms):
  """Normalize with the given running averages of mean value & standard deviation.

  Args:
  * smpl_mat: unnormalized sample matrix
  * rms: running averages of mean value & standard deviation

  Returns:
  * samp_mat: normalized sample matrix
  """

  return smpl_mat if rms is None else (smpl_mat - rms.mean) / rms.std

def denormalize(smpl_mat, rms):
  """De-normalize with the given running averages of mean value & standard deviation.

  Args:
  * smpl_mat: normalized sample matrix
  * rms: running averages of mean value & standard deviation

  Returns:
  * samp_mat: de-normalized sample matrix
  """

  return smpl_mat if rms is None else (smpl_mat * rms.std + rms.mean)

def calc_loss_dcy(trainable_vars):
  """Calculate the weight-decaying loss.

  Args:
  * trainable_vars: list of trainable variables
  """

  return tf.add_n([tf.nn.l2_loss(var) for var in trainable_vars])

def get_target_model_ops(model, model_tr):
  """Get operations related to the target model.

  Args:
  * model: original model
  * model_tr: target model

  Returns:
  * init_op: initialization operation for the target model
  * updt_op: update operation for the target model
  """

  init_ops, updt_ops = [], []
  for var, var_tr in zip(model.vars, model_tr.vars):
    init_ops.append(tf.assign(var_tr, var))
    if var not in model.trainable_vars:
      updt_ops.append(tf.assign(var_tr, var))  # direct update for non-trainable variables
    else:
      updt_ops.append(tf.assign(var_tr, (1. - FLAGS.ddpg_tau) * var_tr + FLAGS.ddpg_tau * var))

  return tf.group(*init_ops), tf.group(*updt_ops)

def get_perturb_op(model, model_noisy, param_noise_std):
  """Get operations for pertubing the model's parameters.

  Args:
  * model: original model
  * model_noisy: perturbed model
  * param_noise_std: standard deviation of the parameter noise

  Returns:
  * perturb_op: perturbation operation for the noisy model
  """

  perturb_ops = []
  for var_clean, var_noisy in zip(model.vars, model_noisy.vars):
    if var_clean not in model.perturbable_vars:
      perturb_ops.append(tf.assign(var_noisy, var_clean))
    else:
      var_noise = tf.random_normal(tf.shape(var_clean), mean=0., stddev=param_noise_std)
      perturb_ops.append(tf.assign(var_noisy, var_clean + var_noise))

  return tf.group(*perturb_ops)

class Agent(object):  # pylint: disable=too-many-instance-attributes
  """DDPG (Deep Deterministic Policy Gradient) agent."""

  def __init__(self, sess, s_dims, a_dims, nb_rlouts, buf_size, a_min=0.0, a_max=1.0):
    """Constructor function.

    Args:
    * sess: TensorFlow session
    * s_dims: number of state vector's dimensions
    * a_dims: number of action vector's dimensions
    * nb_rlouts: number of roll-outs
    * buf_size: maximal number of transitions to be stored in the replay buffer
    * a_min: minimal value in the action vector
    * a_max: maximal value in the action vector
    """

    self.sess = sess
    self.scope = 'agent'
    self.ops = {}
    self.reward_ema = None  # exponential moving average of rewards
    self.in_explore = True
    self.__build(s_dims, a_dims, nb_rlouts, buf_size, a_min, a_max)

  def init(self):
    """Initialize the agent before all roll-outs.

    Actor & critic networks will be initialized and the replay buffer will be reset.
    """

    # initialize actor & critic networks
    self.sess.run([self.ops['init'], self.ops['actor_init'], self.ops['critic_init']])
    self.sess.run(self.ops['target_init'])
    if FLAGS.ddpg_noise_type == 'param':
      self.sess.run([self.ops['actor_np_init'], self.ops['actor_ns_init']])

    # reset non-TF variables and perturbed actor networks
    self.memory.reset()
    self.noise_spec.reset()
    self.in_explore = True

  def init_rlout(self):
    """Initialize the agent before each roll-out.

    Parameter / action noise will be re-initialized for the upcoming roll-out.
    """

    # adjust the noise scale for <TimeDecayNoiseSpec>
    if FLAGS.ddpg_noise_prtl == 'tdecy' and not self.in_explore:
      self.noise_spec.adapt()

    # initialize the parameter / action noise for the current roll-out
    if FLAGS.ddpg_noise_type == 'action':
      self.sess.run(self.action_noise_std.assign(self.action_noise_std_ph),
                    feed_dict={self.action_noise_std_ph: self.noise_spec.stdev_curr})
    elif FLAGS.ddpg_noise_type == 'param':
      self.sess.run(self.ops['actor_np_updt'],
                    feed_dict={self.param_noise_std: self.noise_spec.stdev_curr})
    else:
      raise ValueError('unrecognized noise type: ' + FLAGS.ddpg_noise_type)

  def finalize_rlout(self, rewards):
    """Finalize the current roll-out (to update the baseline function).

    Args:
    * rewards: reward scalars (one per roll-out tick)
    """

    # early return if baseline function is disabled
    if not FLAGS.ddpg_enbl_bsln_func:
      return

    # update the baseline function
    if self.reward_ema is None:
      self.reward_ema = np.mean(rewards)
    else:
      self.reward_ema = FLAGS.ddpg_bsln_decy_rate * self.reward_ema \
          + (1.0 - FLAGS.ddpg_bsln_decy_rate) * np.mean(rewards)

  def record(self, states, actions, rewards, terminals, states_next):
    """Append multiple transitions into the replay buffer.

    Args:
    * states: np.array of current state vectors
    * actions: np.array of action vectors
    * rewards: np.array of reward scalars
    * terminals: np.array of terminal scalars (1: terminal; 0: non-terminal)
    * states_next: np.array of next state vectors
    """

    self.memory.append(states[::FLAGS.ddpg_record_step],
                       actions[::FLAGS.ddpg_record_step],
                       rewards[::FLAGS.ddpg_record_step, None],
                       terminals[::FLAGS.ddpg_record_step, None],
                       states_next[::FLAGS.ddpg_record_step])
    if self.state_rms is not None:
      self.state_rms.updt(states[::FLAGS.ddpg_record_step])

  def train(self):
    """Train the agent's actor & critic networks."""

    # early break if there is no sufficient replays in the buffer
    if not self.memory.is_ready():
      return 0.0, 0.0, self.noise_spec.stdev_curr

    # adjust the noise scale for <AdaptiveNoiseSpec>
    self.in_explore = False
    if FLAGS.ddpg_noise_prtl == 'adapt':
      mbatch = self.memory.sample(FLAGS.ddpg_batch_size)
      self.sess.run(self.ops['actor_ns_updt'],
                    feed_dict={self.param_noise_std: self.noise_spec.stdev_curr})
      action_dist = self.sess.run(self.action_dist, feed_dict={self.states: mbatch['states']})
      self.noise_spec.adapt(action_dist)

    # update actor & critic networks
    mbatch = self.memory.sample(FLAGS.ddpg_batch_size)
    if FLAGS.ddpg_enbl_bsln_func:
      mbatch['rewards'] -= self.reward_ema
    target_q, actor_loss, critic_loss, __, __ = self.sess.run(
      self.ops['monitor'] + [self.ops['actor_updt'], self.ops['critic_updt']],
      feed_dict={self.states: mbatch['states'],
                 self.actions: mbatch['actions'],
                 self.rewards: mbatch['rewards'],
                 self.terminals: mbatch['terminals'],
                 self.states_next: mbatch['states_next']})
    self.sess.run(self.ops['target_updt'])
    if self.return_rms is not None:
      self.return_rms.updt(target_q)

    return actor_loss, critic_loss, self.noise_spec.stdev_curr

  def __build(self, s_dims, a_dims, nb_rlouts, buf_size, a_min, a_max):
    """Build actor & critic networks, replay buffer, and noise generators.

    Args:
    * s_dims: number of state vector's dimensions
    * a_dims: number of action vector's dimensions
    * nb_rlouts: number of roll-outs
    * buf_size: maximal number of transitions to be stored in the replay buffer
    * a_min: minimal value in the action vector
    * a_max: maximal value in the action vector
    """

    # inputs
    with tf.variable_scope(self.scope):
      self.states = tf.placeholder(tf.float32, shape=[None, s_dims], name='states')
      self.actions = tf.placeholder(tf.float32, shape=[None, a_dims], name='actions')
      self.rewards = tf.placeholder(tf.float32, shape=[None, 1], name='rewards')
      self.terminals = tf.placeholder(tf.float32, shape=[None, 1], name='terminals')
      self.states_next = tf.placeholder(tf.float32, shape=[None, s_dims], name='states_next')
      self.param_noise_std = tf.placeholder(tf.float32, shape=[], name='param_noise_std')
      self.action_noise_std_ph = tf.placeholder(tf.float32, shape=[], name='action_noise_std_ph')
      self.action_noise_std = tf.get_variable('action_noise_std', shape=[], trainable=False)

    # normalize states & returns to zero-mean & unit-norm
    normalize_state = False
    normalize_return = False
    self.state_rms, self.return_rms = None, None
    if normalize_state:
      with tf.variable_scope(self.scope + '/state_rms'):
        self.state_rms = RunningMeanStd(self.sess, s_dims)
    if normalize_return:
      with tf.variable_scope(self.scope + '/return_rms'):
        self.return_rms = RunningMeanStd(self.sess, 1)
    normalized_states = normalize(self.states, self.state_rms)
    normalized_states_next = normalize(self.states_next, self.state_rms)

    # actor & critic networks (mn: main / tr: target)
    self.actor = Actor(a_dims, a_min, a_max, scope=self.scope + '/actor_mn')
    self.actor_tr = Actor(a_dims, a_min, a_max, scope=self.scope + '/actor_tr')
    self.critic = Critic(scope=self.scope + '/critic_mn')
    self.critic_tr = Critic(scope=self.scope + '/critic_tr')

    # experience replay buffer
    self.memory = ReplayBuffer(s_dims, a_dims, buf_size)

    # parameter noise
    if FLAGS.ddpg_noise_prtl == 'adapt':
      self.noise_spec = AdaptiveNoiseSpec()
    elif FLAGS.ddpg_noise_prtl == 'tdecy':
      self.noise_spec = TimeDecayNoiseSpec(nb_rlouts)
    else:
      raise ValueError('unrecognized noise adjustment protocol: ' + FLAGS.ddpg_noise_prtl)

    # create core TF parts shared across components
    self.actor_tf = self.actor(normalized_states)
    self.normalized_critic_tf = self.critic(normalized_states, self.actions)
    self.critic_tf = denormalize(self.normalized_critic_tf, self.return_rms)
    self.critic_tf_w_actor = denormalize(self.critic(
      normalized_states, self.actor_tf, reuse=True), self.return_rms)
    self.critic_tr_tf = denormalize(self.critic_tr(
      normalized_states_next, self.actor_tr(normalized_states_next)), self.return_rms)
    self.target_q = self.rewards + (1.0 - self.terminals) * FLAGS.ddpg_gamma * self.critic_tr_tf
    self.normalized_target_q = normalize(self.target_q, self.return_rms)

    # setup components' TF operations
    if FLAGS.ddpg_noise_type == 'action':
      self.actor_tf_np = self.__setup_action_noise(a_min, a_max)
    elif FLAGS.ddpg_noise_type == 'param':
      self.actor_tf_np, self.action_dist = \
        self.__setup_param_noise(normalized_states, a_dims, a_min, a_max)
    self.__setup_target_init_updt_ops()
    actor_loss = self.__setup_actor_init_updt_ops()
    critic_loss = self.__setup_critic_init_updt_ops()

    # setup TF tensors & operations
    self.actions_clean = self.actor_tf  # for deployment
    self.actions_noisy = self.actor_tf_np  # for training
    self.ops['init'] = tf.variables_initializer(self.vars)
    self.ops['monitor'] = [self.target_q, actor_loss, critic_loss]

  def __setup_action_noise(self, a_min, a_max):
    """Setup the action noise.

    Args:
    * a_min: minimal value in the action vector
    * a_max: maximal value in the action vector

    Returns:
    * actor_tf_np: primary noisy actor network's outputs
    """

    actions_noise = tf.random_normal(tf.shape(self.actor_tf), stddev=self.action_noise_std)
    actor_tf_np = tf.clip_by_value(self.actor_tf + actions_noise, a_min, a_max)

    return actor_tf_np

  def __setup_param_noise(self, normalized_states, a_dims, a_min, a_max):
    """Setup the parameter noise.

    Args:
    * normalized_states: normalized state vectors
    * a_dims: number of action vector's dimensions
    * a_min: minimal value in the action vector
    * a_max: maximal value in the action vector

    Returns:
    * actor_tf_np: primary noisy actor network's outputs
    * action_dist: distance between clean outputs and secondary noisy actor network's outputs
    """

    # noisy actor network - primary
    actor_np = Actor(a_dims, a_min, a_max, scope=self.scope + '/actor_np')
    actor_tf_np = actor_np(normalized_states)
    self.ops['actor_np_init'] = tf.variables_initializer(actor_np.vars)
    self.ops['actor_np_updt'] = get_perturb_op(self.actor, actor_np, self.param_noise_std)

    # noisy actor network - secondary
    actor_ns = Actor(a_dims, a_min, a_max, scope=self.scope + '/actor_ns')
    actor_tf_ns = actor_ns(normalized_states)
    self.ops['actor_ns_init'] = tf.variables_initializer(actor_ns.vars)
    self.ops['actor_ns_updt'] = get_perturb_op(self.actor, actor_ns, self.param_noise_std)

    # distance between clean & noisy action networks' outputs
    action_dist = tf.reduce_mean(tf.abs(self.actor_tf - actor_tf_ns))

    return actor_tf_np, action_dist

  def __setup_target_init_updt_ops(self):
    """Setup target network's initialization & update operations."""

    actor_init_op, actor_updt_op = get_target_model_ops(self.actor, self.actor_tr)
    critic_init_op, critic_updt_op = get_target_model_ops(self.critic, self.critic_tr)
    self.ops['target_init'] = tf.group(actor_init_op, critic_init_op)
    self.ops['target_updt'] = tf.group(actor_updt_op, critic_updt_op)

  def __setup_actor_init_updt_ops(self):
    """Setup actor network's initialization & update operations.

    Returns:
    * actor_loss: actor network's loss
    """

    actor_loss = -tf.reduce_mean(self.critic_tf_w_actor)
    actor_loss += FLAGS.ddpg_loss_w_dcy * calc_loss_dcy(self.actor.trainable_vars)
    optimizer = tf.train.AdamOptimizer(FLAGS.ddpg_lrn_rate)
    self.ops['actor_updt'] = optimizer.minimize(actor_loss, var_list=self.actor.trainable_vars)
    self.ops['actor_init'] = tf.variables_initializer(self.actor.vars + optimizer.variables())

    return actor_loss

  def __setup_critic_init_updt_ops(self):
    """Setup critic network's initialization & update operations.

    Returns:
    * critic_loss: critic network's loss
    """

    critic_loss = tf.nn.l2_loss(self.normalized_critic_tf - self.normalized_target_q)
    critic_loss += FLAGS.ddpg_loss_w_dcy * calc_loss_dcy(self.critic.trainable_vars)
    optimizer = tf.train.AdamOptimizer(FLAGS.ddpg_lrn_rate)
    self.ops['critic_updt'] = optimizer.minimize(critic_loss, var_list=self.critic.trainable_vars)
    self.ops['critic_init'] = tf.variables_initializer(self.critic.vars + optimizer.variables())

    return critic_loss

  @property
  def vars(self):
    """List of all global variables."""

    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
