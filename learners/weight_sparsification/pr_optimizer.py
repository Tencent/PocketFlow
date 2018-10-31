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
"""Pruning ratio optimizer for the weight sparsification learner."""

import os
import re
import math
from timeit import default_timer as timer
import numpy as np
import tensorflow as tf

from learners.weight_sparsification.rl_helper import RLHelper
from learners.weight_sparsification.utils import get_maskable_vars
from rl_agents.ddpg.agent import Agent as DdpgAgent
from utils.multi_gpu_wrapper import MultiGpuWrapper as mgw
from utils.misc_utils import is_primary_worker

FLAGS = tf.app.flags.FLAGS

def get_vars_by_scope(scope):
  """Get list of variables within certain name scope.

  Args:
  * scope: name scope

  Returns:
  * vars_dict: dictionary of list of all, trainable, and maskable variables
  """

  vars_dict = {}
  vars_dict['all'] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
  vars_dict['trainable'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
  vars_dict['maskable'] = get_maskable_vars(vars_dict['trainable'])

  return vars_dict

def get_ops_by_scope_n_patterns(scope, patterns):
  """Get list of operations within certain name scope and also matches the pattern.

  Args:
  * scope: name scope
  * patterns: list of name patterns to be matched

  Returns:
  * ops: list of operations
  """

  ops = []
  for op in tf.get_default_graph().get_operations():
    if not op.name.startswith(scope):
      continue
    for pattern in patterns:
      if re.search(pattern, op.name) is not None:
        ops += [op]

  return ops

def save_vals_to_file(vals, file_path):
  """Save a list of values to a plain text file.

  Args:
  * vals: list of values
  * file_path: file path
  """

  with open(file_path, 'w') as o_file:
    o_file.write('\n'.join(['%f' % val for val in vals]))

def restore_vals_from_file(file_path):
  """Restore a list of values from a plain text file.

  Args:
  * file_path: file path

  Returns:
  * vals: list of values
  """

  with open(file_path, 'r') as i_file:
    return np.array([float(i_line) for i_line in i_file])

class PROptimizer(object):  # pylint: disable=too-many-instance-attributes
  """Pruning ratio optimizer for the weight sparsification learner."""

  def __init__(self, model_helper, mpi_comm):
    """Constructor function.

    Args:
    * model_helper: model helper with definitions of model & dataset
    * mpi_comm: MPI communication object
    """

    # initialize attributes
    self.model_name = model_helper.model_name
    self.dataset_name = model_helper.dataset_name
    self.mpi_comm = mpi_comm
    self.data_scope = 'data'
    self.model_scope_full = 'model'
    self.model_scope_prnd = 'pruned_model'

    # build graphs for training & evaluation
    if FLAGS.ws_prune_ratio_prtl in ['uniform', 'heurist']:
      self.__build_minimal(model_helper)  # no RL-related tensors & operations
    elif FLAGS.ws_prune_ratio_prtl == 'optimal':
      self.__build_train(model_helper)
      self.__build_eval(model_helper)
    else:
      raise ValueError('unrecognzed WS pruning ratio protocol: ' + FLAGS.ws_prune_ratio_prtl)

  def run(self):
    """Run the optimizer to obtain pruning ratios for all maskable variables.

    Returns:
    * var_names_n_prune_ratios: list of variable name & pruning ratio pairs
    """

    # obtain a list of (variable name, pruning ratio) tuples
    if FLAGS.ws_prune_ratio_prtl == 'uniform':
      var_names_n_prune_ratios = self.__calc_uniform_prune_ratios()
    elif FLAGS.ws_prune_ratio_prtl == 'heurist':
      var_names_n_prune_ratios = self.__calc_heurist_prune_ratios()
    elif FLAGS.ws_prune_ratio_prtl == 'optimal':
      var_names_n_prune_ratios = self.__calc_optimal_prune_ratios()

    # display the pruning ratio for each maskable variable
    if is_primary_worker('global'):
      for var_name, prune_ratio in var_names_n_prune_ratios:
        tf.logging.info('%s: %f' % (var_name, prune_ratio))

    return var_names_n_prune_ratios

  def __build_minimal(self, model_helper):
    """Build the minimal graph for 'uniform' & 'heurist' protocols.

    Args:
    * model_helper: model helper with definitions of model & dataset
    """

    with tf.Graph().as_default():
      # create a TF session for the current graph
      config = tf.ConfigProto()
      config.gpu_options.visible_device_list = str(mgw.local_rank() if FLAGS.enbl_multi_gpu else 0)  # pylint: disable=no-member
      self.sess = tf.Session(config=config)

      # data input pipeline
      with tf.variable_scope(self.data_scope):
        iterator = model_helper.build_dataset_train()
        images, __ = iterator.get_next()

      # model definition - full-precision network
      with tf.variable_scope(self.model_scope_full):
        __ = model_helper.forward_eval(images)  # DO NOT USE forward_train() HERE!!!
        self.vars_full = get_vars_by_scope(self.model_scope_full)

  def __build_train(self, model_helper):  # pylint: disable=too-many-locals
    """Build the training graph for the 'optimal' protocol.

    Args:
    * model_helper: model helper with definitions of model & dataset
    """

    with tf.Graph().as_default():
      # create a TF session for the current graph
      config = tf.ConfigProto()
      config.gpu_options.visible_device_list = str(mgw.local_rank() if FLAGS.enbl_multi_gpu else 0)  # pylint: disable=no-member
      sess = tf.Session(config=config)

      # data input pipeline
      with tf.variable_scope(self.data_scope):
        iterator, __ = model_helper.build_dataset_train(enbl_trn_val_split=True)
        images, labels = iterator.get_next()

      # model definition - full-precision network
      with tf.variable_scope(self.model_scope_full):
        logits = model_helper.forward_eval(images)  # DO NOT USE forward_train() HERE!!!
        self.vars_full = get_vars_by_scope(self.model_scope_full)
        self.saver_full = tf.train.Saver(self.vars_full['all'])
        self.save_path_full = FLAGS.save_path

      # model definition - weight sparsified network
      with tf.variable_scope(self.model_scope_prnd):
        # forward pass & variables' saver
        logits = model_helper.forward_eval(images)  # DO NOT USE forward_train() HERE!!!
        self.vars_prnd = get_vars_by_scope(self.model_scope_prnd)
        self.maskable_var_names = [var.name for var in self.vars_prnd['maskable']]
        loss, __ = model_helper.calc_loss(labels, logits, self.vars_prnd['trainable'])
        self.saver_prnd_train = tf.train.Saver(self.vars_prnd['all'])
        self.save_path_prnd = FLAGS.save_path.replace('models', 'models_pruned')

        # build masks for variable pruning
        self.masks, self.pr_all, self.pr_assign_op = self.__build_masks()

      # create operations for initializing the weight sparsified network
      init_ops = []
      for var_full, var_prnd in zip(self.vars_full['all'], self.vars_prnd['all']):
        if var_full not in self.vars_full['maskable']:
          init_ops += [var_prnd.assign(var_full)]
        else:
          idx = self.vars_full['maskable'].index(var_full)
          init_ops += [var_prnd.assign(var_full * self.masks[idx])]
      self.init_op = tf.group(init_ops)

      # build operations for layerwise regression & network fine-tuning
      self.rg_init_op, self.rg_train_ops = self.__build_layer_rg_ops()
      self.ft_init_op, self.ft_train_op = self.__build_network_ft_ops(loss)
      if FLAGS.enbl_multi_gpu:
        self.bcast_op = mgw.broadcast_global_variables(0)

      # create RL helper & agent on the primary worker
      if is_primary_worker('global'):
        self.rl_helper, self.agent = self.__build_rl_helper_n_agent(sess)

      # TF operations
      self.sess_train = sess

  def __build_eval(self, model_helper):
    """Build the evaluation graph for the 'optimal' protocol.

    Args:
    * model_helper: model helper with definitions of model & dataset
    """

    with tf.Graph().as_default():
      # create a TF session for the current graph
      config = tf.ConfigProto()
      config.gpu_options.visible_device_list = str(mgw.local_rank() if FLAGS.enbl_multi_gpu else 0)  # pylint: disable=no-member
      self.sess_eval = tf.Session(config=config)

      # data input pipeline
      with tf.variable_scope(self.data_scope):
        __, iterator = model_helper.build_dataset_train(enbl_trn_val_split=True)
        images, labels = iterator.get_next()

      # model definition - weight sparsified network
      with tf.variable_scope(self.model_scope_prnd):
        logits = model_helper.forward_eval(images)
        vars_prnd = get_vars_by_scope(self.model_scope_prnd)
        self.loss_eval, self.metrics_eval = \
            model_helper.calc_loss(labels, logits, vars_prnd['trainable'])
        self.saver_prnd_eval = tf.train.Saver(vars_prnd['all'])

  def __build_masks(self):
    """Build pruning masks for all the maskable variables.

    Returns:
    * masks: list of pruning masks
    * pr_all: placeholder of all the pruning ratios
    * pr_assign_op: pruning ratios' assign operation
    """

    # create masks & pruning ratios
    masks, prune_ratios = [], []
    for var_full, var_prnd in zip(self.vars_full['maskable'], self.vars_prnd['maskable']):
      name = var_prnd.name.replace(':0', '_prune_ratio')
      prune_ratio = tf.get_variable(name, shape=[], trainable=False)
      mask_thres = tf.contrib.distributions.percentile(tf.abs(var_full), prune_ratio * 100)
      mask = tf.cast(tf.abs(var_full) > mask_thres, tf.float32)
      masks += [mask]
      prune_ratios += [prune_ratio]

    # create a placeholder to pass pruning ratios to the TF tensor
    pr_all = tf.placeholder(tf.float32, shape=(len(prune_ratios)), name='pr_all')
    pr_all_split = tf.split(pr_all, len(prune_ratios))
    pr_assign_op = tf.group([prune_ratio.assign(tf.reshape(pr, []))
                             for prune_ratio, pr in zip(prune_ratios, pr_all_split)])

    return masks, pr_all, pr_assign_op

  def __build_layer_rg_ops(self):
    """Build operations for layerwise regression.

    Returns:
    * init_op: initialization operation
    * train_ops: list of training operations, one per layer
    """


    # obtain lists of core operations in both networks
    if self.model_name.startswith('mobilenet'):
      patterns = ['pointwise/Conv2D', 'Conv2d_1c_1x1/Conv2D']
    else:
      patterns = ['Conv2D', 'MatMul']
    core_ops_full = get_ops_by_scope_n_patterns(self.model_scope_full, patterns)
    core_ops_prnd = get_ops_by_scope_n_patterns(self.model_scope_prnd, patterns)

    # construct initialization & training operations
    init_ops, train_ops = [], []
    for idx, (core_op_full, core_op_prnd) in enumerate(zip(core_ops_full, core_ops_prnd)):
      loss = tf.nn.l2_loss(core_op_prnd.outputs[0] - core_op_full.outputs[0])
      optimizer_base = tf.train.AdamOptimizer(FLAGS.ws_lrn_rate_rg)
      if FLAGS.enbl_multi_gpu:
        optimizer = mgw.DistributedOptimizer(optimizer_base)
      else:
        optimizer = optimizer_base
      grads_origin = optimizer.compute_gradients(loss, [self.vars_prnd['maskable'][idx]])
      grads_pruned = self.__calc_grads_pruned(grads_origin)
      train_ops += [optimizer.apply_gradients(grads_pruned)]
      init_ops += [tf.variables_initializer(optimizer_base.variables())]

    return tf.group(init_ops), train_ops

  def __build_network_ft_ops(self, loss):
    """Build operations for network fine-tuning.

    Args:
    * loss: loss function's value

    Returns:
    * init_op: initialization operation
    * train_op: training operation
    """

    optimizer_base = tf.train.AdamOptimizer(FLAGS.ws_lrn_rate_ft)
    if FLAGS.enbl_multi_gpu:
      optimizer = mgw.DistributedOptimizer(optimizer_base)
    else:
      optimizer = optimizer_base
    grads_origin = optimizer.compute_gradients(loss, self.vars_prnd['trainable'])
    grads_pruned = self.__calc_grads_pruned(grads_origin)
    train_op = optimizer.apply_gradients(grads_pruned)
    init_op = tf.variables_initializer(optimizer_base.variables())

    return init_op, train_op

  def __calc_grads_pruned(self, grads_origin):
    """Calculate the mask-pruned gradients.

    Args:
    * grads_origin: list of original gradients

    Returns:
    * grads_pruned: list of mask-pruned gradients
    """

    grads_pruned = []
    for grad in grads_origin:
      if grad[1].name not in self.maskable_var_names:
        grads_pruned += [grad]
      else:
        idx_mask = self.maskable_var_names.index(grad[1].name)
        grads_pruned += [(grad[0] * self.masks[idx_mask], grad[1])]

    return grads_pruned

  def __build_rl_helper_n_agent(self, sess):
    """Build the RL helper and controller / agent.

    Args:
    * sess: TensorFlow session

    Returns:
    * rl_helper: RL helper
    * agent: RL controller / agent
    """

    # build an RL helper
    skip_head_n_tail = (self.dataset_name == 'cifar_10')  # skip head & tail layers on CIFAR-10
    rl_helper = RLHelper(sess, self.vars_full['maskable'], skip_head_n_tail)

    # build an RL controller / agent
    s_dims = rl_helper.s_dims
    a_dims = 1
    nb_rlouts = FLAGS.ws_nb_rlouts
    buf_size = len(self.vars_full['maskable']) * FLAGS.ws_nb_rlouts_min
    a_lbnd = 0.0
    a_ubnd = 1.0
    agent = DdpgAgent(sess, s_dims, a_dims, nb_rlouts, buf_size, a_lbnd, a_ubnd)

    return rl_helper, agent

  def __calc_uniform_prune_ratios(self):
    """Calculate pruning ratios using the 'uniform' protocol.

    Returns:
    * var_names_n_prune_ratios: list of variable name & pruning ratio pairs
    """

    return [(var.name, FLAGS.ws_prune_ratio) for var in self.vars_full['maskable']]

  def __calc_heurist_prune_ratios(self):
    """Calculate pruning ratios using the 'heurist' protocol.

    Returns:
    * var_names_n_prune_ratios: list of variable name & pruning ratio pairs
    """

    var_shapes = [self.sess.run(tf.shape(var)) for var in self.vars_full['maskable']]
    nb_params = np.array([np.prod(var_shape) for var_shape in var_shapes])
    alpha = FLAGS.ws_prune_ratio * np.sum(nb_params) / np.sum(nb_params * np.log(nb_params))
    var_names_n_prune_ratios = []
    for idx, var_full in enumerate(self.vars_full['maskable']):
      prune_ratio = alpha * np.log(nb_params[idx])
      var_names_n_prune_ratios += [(var_full.name, prune_ratio)]

    return var_names_n_prune_ratios

  def __calc_optimal_prune_ratios(self):
    """Calculate pruning ratios using the 'optimal' protocol.

    Returns:
    * var_names_n_prune_ratios: list of variable name & pruning ratio pairs
    """

    # restore the full-precision model from checkpoint files
    save_path = tf.train.latest_checkpoint(os.path.dirname(self.save_path_full))
    self.saver_full.restore(self.sess_train, save_path)

    # train an RL agent through multiple roll-outs
    if is_primary_worker('global'):
      self.agent.init()
    reward_best = np.NINF  # pylint: disable=no-member
    prune_ratios_best = None
    file_path_prune_ratios = './ws.prune.ratios'
    file_path_reward = './ws.reward'
    for idx_rlout in range(FLAGS.ws_nb_rlouts):
      # compute actions & pruning ratios
      if is_primary_worker('global'):
        tf.logging.info('starting %d-th roll-out' % idx_rlout)
        prune_ratios, states_n_actions = self.__calc_rlout_actions()
        save_vals_to_file(prune_ratios, file_path_prune_ratios)
      if FLAGS.enbl_multi_gpu:
        self.mpi_comm.Barrier()
      prune_ratios = restore_vals_from_file(file_path_prune_ratios)

      # fine-tune the weight sparsified network to compute the reward
      reward = self.__calc_rlout_reward(prune_ratios)
      if is_primary_worker('global'):
        save_vals_to_file(np.array([reward]), file_path_reward)
      if FLAGS.enbl_multi_gpu:
        self.mpi_comm.Barrier()
      reward = restore_vals_from_file(file_path_reward)[0]

      # update the baseline function in DDPG
      if is_primary_worker('global'):
        rewards = reward * np.ones(len(self.vars_full['maskable']))
        self.agent.finalize_rlout(rewards)

      # record transitions to train the RL agent
      if is_primary_worker('global'):
        self.__record_rlout_transitions(states_n_actions, reward)

      # record the best combination of pruning ratios
      if reward_best < reward:
        if is_primary_worker('global'):
          tf.logging.info('best reward updated: %.4f -> %.4f' % (reward_best, reward))
          tf.logging.info('optimal pruning ratios: ' +
                          ' '.join(['%.2f' % prune_ratio for prune_ratio in prune_ratios[:]]))
        reward_best = reward
        prune_ratios_best = np.copy(prune_ratios)

    # setup the optimal pruning ratios
    var_names_n_prune_ratios = []
    for idx, var_full in enumerate(self.vars_full['maskable']):
      var_names_n_prune_ratios += [(var_full.name, prune_ratios_best[idx])]

    return var_names_n_prune_ratios

  def __calc_rlout_actions(self):
    """Calculate actions within one roll-out.

    Returns:
    * prune_ratios: list of pruning ratios
    * states_n_actions: list of state vector and action pairs
    """

    self.agent.init_rlout()
    prune_ratios, states_n_actions = [], []
    for idx in range(len(self.vars_full['maskable'])):
      state = self.rl_helper.calc_state(idx)
      action = self.sess_train.run(self.agent.actions_noisy, feed_dict={self.agent.states: state})
      prune_ratio = self.rl_helper.cvt_action_to_prune_ratio(idx, action[0][0])
      prune_ratios += [prune_ratio]
      states_n_actions += [(state, action)]
      actor_loss, critic_loss, noise_std = self.agent.train()
    tf.logging.info('a-loss = %.2e | c-loss = %.2e | noise std. = %.2e'
                    % (actor_loss, critic_loss, noise_std))

    return prune_ratios, states_n_actions

  def __calc_rlout_reward(self, prune_ratios):
    """Calculate the reward of the current roll-out.

    Args:
    * prune_ratios: list of pruning ratios

    Returns:
    * reward: reward of the current roll-out
    """

    # initialize the weight sparsified network with given pruning ratios
    self.sess_train.run(self.pr_assign_op, feed_dict={self.pr_all: prune_ratios})
    self.sess_train.run([self.init_op, self.rg_init_op, self.ft_init_op])
    if FLAGS.enbl_multi_gpu:
      self.sess_train.run(self.bcast_op)

    # evaluate the network before re-training
    if is_primary_worker('global'):
      loss_pre, metrics_pre = self.__calc_loss_n_metrics()
      assert 'accuracy' in metrics_pre or 'acc_top5' in metrics_pre, \
        'either <accuracy> or <acc_top5> must be evaluated and returned'

    # re-train the network with layerwise regression & network fine-tuning
    self.__retrain_network()

    # evaluate the network after re-training
    if is_primary_worker('global'):
      loss_post, metrics_post = self.__calc_loss_n_metrics()
      assert 'accuracy' in metrics_post or 'acc_top5' in metrics_post, \
          'either <accuracy> or <acc_top5> must be evaluated and returned'

    # evaluate the weight sparsified network
    reward = None
    if is_primary_worker('global'):
      if 'accuracy' in metrics_post:
        reward_pre = self.rl_helper.calc_reward(metrics_pre['accuracy'])
        reward = self.rl_helper.calc_reward(metrics_post['accuracy'])
      elif 'acc_top5' in metrics_post:
        reward_pre = self.rl_helper.calc_reward(metrics_pre['acc_top5'])
        reward = self.rl_helper.calc_reward(metrics_post['acc_top5'])
      prune_ratio = self.rl_helper.calc_overall_prune_ratio()
      metrics_diff = ' | '.join(
        ['%s: %.4f -> %.4f' % (key, metrics_pre[key], metrics_post[key]) for key in metrics_post])
      tf.logging.info('loss: %.4e -> %.4e | %s | reward: %.4f -> %.4f | prune_ratio = %.4f'
                      % (loss_pre, loss_post, metrics_diff, reward_pre, reward, prune_ratio))

    return reward

  def __retrain_network(self):
    """Retrain the network with layerwise regression & network fine-tuning."""

    # determine how many iterations to be executed for regression & fine-tuning
    nb_workers = mgw.size() if FLAGS.enbl_multi_gpu else 1
    nb_iters_rg = int(math.ceil(FLAGS.ws_nb_iters_rg / nb_workers))
    nb_iters_ft = int(math.ceil(FLAGS.ws_nb_iters_ft / nb_workers))

    # re-train the network with layerwise regression
    time_prev = timer()
    for rg_train_op in self.rg_train_ops:
      for __ in range(nb_iters_rg):
        self.sess_train.run(rg_train_op)
    time_rg = timer() - time_prev

    # re-train the network with global fine-tuning
    time_prev = timer()
    for __ in range(nb_iters_ft):
      self.sess_train.run(self.ft_train_op)
    time_ft = timer() - time_prev

    # display the time consumption
    tf.logging.info('time consumption: %.4f (s) - RG | %.4f (s) - FT' % (time_rg, time_ft))

  def __record_rlout_transitions(self, states_n_actions, reward):
    """Record transitions of the current roll-out.

    Args:
    * states_n_actions: list of state vector and action pairs
    * reward: reward of the current roll-out
    """

    # record transitions from the current roll-out to train the RL agent
    for idx, (state, action) in enumerate(states_n_actions):
      if idx != len(states_n_actions) - 1:
        terminal = np.zeros((1, 1))
        state_next = states_n_actions[idx + 1][0]
      else:
        terminal = np.ones((1, 1))
        state_next = np.zeros_like(state)
      self.agent.record(state, action, reward * np.ones((1, 1)), terminal, state_next)

  def __calc_loss_n_metrics(self):
    """Calculate the loss function's value and evaluation metrics.

    Returns:
    * loss: loss function's value
    * metrics: evaluation metrics
    """

    # save the model from the training graph and restore it to the evaluation graph
    save_path = self.saver_prnd_train.save(self.sess_train, self.save_path_prnd)
    self.saver_prnd_eval.restore(self.sess_eval, save_path)

    # evaluate the model's loss & accuracy
    if FLAGS.ws_nb_iters_feval > 0:
      nb_iters = FLAGS.ws_nb_iters_feval
    else:
      nb_iters = FLAGS.nb_smpls_eval // FLAGS.batch_size_eval
    eval_rslts = np.zeros((nb_iters, 1 + len(self.metrics_eval)))
    metric_names = list(self.metrics_eval.keys())
    metric_values = list(self.metrics_eval.values())
    for idx_iter in range(nb_iters):
      eval_rslts[idx_iter] = self.sess_eval.run([self.loss_eval] + metric_values)
    loss = np.mean(eval_rslts[:, 0])
    metrics = {}
    for idx, name in enumerate(metric_names):
      metrics[name] = np.mean(eval_rslts[:, idx + 1])

    return loss, metrics
