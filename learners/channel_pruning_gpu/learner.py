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
"""Channel pruning learner with GPU-based optimization."""

import os
import re
import math
from timeit import default_timer as timer
import numpy as np
import tensorflow as tf

from learners.abstract_learner import AbstractLearner
from learners.distillation_helper import DistillationHelper
from utils.multi_gpu_wrapper import MultiGpuWrapper as mgw

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('cpg_save_path', './models_cpg/model.ckpt', 'CPG: model\'s save path')
tf.app.flags.DEFINE_string('cpg_save_path_eval', './models_cpg_eval/model.ckpt',
                           'CPG: model\'s save path for evaluation')
tf.app.flags.DEFINE_string('cpg_prune_ratio_type', 'uniform',
                           'CPG: pruning ratio type (\'uniform\' OR \'list\')')
tf.app.flags.DEFINE_float('cpg_prune_ratio', 0.5, 'CPG: uniform pruning ratio')
tf.app.flags.DEFINE_boolean('cpg_skip_ht_layers', True, 'CPG: skip head & tail layers for pruning')
tf.app.flags.DEFINE_string('cpg_prune_ratio_file', None,
                           'CPG: file path to the list of pruning ratios')
tf.app.flags.DEFINE_float('cpg_lrn_rate_pgd_init', 1e-10,
                          'CPG: proximal gradient descent\'s initial learning rate')
tf.app.flags.DEFINE_float('cpg_lrn_rate_pgd_incr', 1.4,
                          'CPG: proximal gradient descent\'s learning rate\'s increase ratio')
tf.app.flags.DEFINE_float('cpg_lrn_rate_pgd_decr', 0.7,
                          'CPG: proximal gradient descent\'s learning rate\'s decrease ratio')
tf.app.flags.DEFINE_float('cpg_lrn_rate_adam', 1e-2, 'CPG: Adam\'s initial learning rate')
tf.app.flags.DEFINE_integer('cpg_nb_iters_layer', 1000, 'CPG: # of iterations for layer-wise FT')

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
  vars_dict['maskable'] = []
  conv2d_pattern = re.compile(r'/Conv2D$')
  conv2d_ops = get_ops_by_scope_n_pattern(scope, conv2d_pattern)
  for var in vars_dict['trainable']:
    for op in conv2d_ops:
      for op_input in op.inputs:
        if op_input.name == var.name.replace(':0', '/read:0'):
          vars_dict['maskable'] += [var]

  return vars_dict

def get_ops_by_scope_n_pattern(scope, pattern):
  """Get list of operations within certain name scope and also matches the pattern.

  Args:
  * scope: name scope
  * pattern: name pattern to be matched

  Returns:
  * ops: list of operations
  """

  ops = []
  for op in tf.get_default_graph().get_operations():
    if op.name.startswith(scope) and re.search(pattern, op.name) is not None:
      ops += [op]

  return ops

def calc_prune_ratio(vars_list):
  """Calculate the overall pruning ratio for the given list of variables.

  Args:
  * vars_list: list of variables

  Returns:
  * prune_ratio: overall pruning ratio of the given list of variables
  """

  nb_params_nnz = tf.add_n([tf.count_nonzero(var) for var in vars_list])
  nb_params_all = tf.add_n([tf.size(var) for var in vars_list])
  prune_ratio = 1.0 - tf.cast(nb_params_nnz, tf.float32) / tf.cast(nb_params_all, tf.float32)

  return prune_ratio

class ChannelPrunedGpuLearner(AbstractLearner):  # pylint: disable=too-many-instance-attributes
  """Channel pruning learner with GPU-based optimization."""

  def __init__(self, sm_writer, model_helper):
    """Constructor function.

    Args:
    * sm_writer: TensorFlow's summary writer
    * model_helper: model helper with definitions of model & dataset
    """

    # class-independent initialization
    super(ChannelPrunedGpuLearner, self).__init__(sm_writer, model_helper)

    # define scopes for full & channel-pruned models
    self.model_scope_full = 'model'
    self.model_scope_prnd = 'pruned_model'

    # download the pre-trained model
    if self.is_primary_worker('local'):
      self.download_model()  # pre-trained model is required
    self.auto_barrier()
    tf.logging.info('model files: ' + ', '.join(os.listdir('./models')))

    # class-dependent initialization
    if FLAGS.enbl_dst:
      self.helper_dst = DistillationHelper(sm_writer, model_helper, self.mpi_comm)
    self.__build_train()
    self.__build_eval()

  def train(self):
    """Train a model and periodically produce checkpoint files."""

    # restore the full model from pre-trained checkpoints
    save_path = tf.train.latest_checkpoint(os.path.dirname(self.save_path_full))
    self.saver_full.restore(self.sess_train, save_path)

    # initialization
    self.sess_train.run([self.init_op, self.init_opt_op])
    self.sess_train.run([layer_op['init_opt'] for layer_op in self.layer_ops])
    if FLAGS.enbl_multi_gpu:
      self.sess_train.run(self.bcast_op)

    # choose channels and evaluate the model before re-training
    self.__choose_channels()
    if self.is_primary_worker('global'):
      self.__save_model(is_train=True)
      self.evaluate()
    self.auto_barrier()

    # fine-tune the model with chosen channels only
    time_prev = timer()
    for idx_iter in range(self.nb_iters_train):
      # train the model
      if (idx_iter + 1) % FLAGS.summ_step != 0:
        self.sess_train.run(self.train_op)
      else:
        __, summary, log_rslt = self.sess_train.run([self.train_op, self.summary_op, self.log_op])
        if self.is_primary_worker('global'):
          time_step = timer() - time_prev
          self.__monitor_progress(summary, log_rslt, idx_iter, time_step)
          time_prev = timer()

      # save the model at certain steps
      if self.is_primary_worker('global') and (idx_iter + 1) % FLAGS.save_step == 0:
        self.__save_model(is_train=True)
        self.evaluate()
      self.auto_barrier()

    # save the final model
    if self.is_primary_worker('global'):
      self.__save_model(is_train=True)
      self.__restore_model(is_train=False)
      self.__save_model(is_train=False)
      self.evaluate()

  def evaluate(self):
    """Restore a model from the latest checkpoint files and then evaluate it."""

    self.__restore_model(is_train=False)
    nb_iters = int(np.ceil(float(FLAGS.nb_smpls_eval) / FLAGS.batch_size_eval))
    eval_rslts = np.zeros((nb_iters, len(self.eval_op)))
    self.dump_n_eval(outputs=None, action='init')
    for idx_iter in range(nb_iters):
      if (idx_iter + 1) % 100 == 0:
        tf.logging.info('process the %d-th mini-batch for evaluation' % (idx_iter + 1))
      eval_rslts[idx_iter], outputs = self.sess_eval.run([self.eval_op, self.outputs_eval])
      self.dump_n_eval(outputs=outputs, action='dump')
    self.dump_n_eval(outputs=None, action='eval')
    for idx, name in enumerate(self.eval_op_names):
      tf.logging.info('%s = %.4e' % (name, np.mean(eval_rslts[:, idx])))

  def __build_train(self):  # pylint: disable=too-many-locals,too-many-statements
    """Build the training graph."""

    with tf.Graph().as_default():
      # create a TF session for the current graph
      config = tf.ConfigProto()
      config.gpu_options.visible_device_list = str(mgw.local_rank() if FLAGS.enbl_multi_gpu else 0)  # pylint: disable=no-member
      sess = tf.Session(config=config)

      # data input pipeline
      with tf.variable_scope(self.data_scope):
        iterator = self.build_dataset_train()
        images, labels = iterator.get_next()

      # model definition - distilled model
      if FLAGS.enbl_dst:
        logits_dst = self.helper_dst.calc_logits(sess, images)

      # model definition - full model
      with tf.variable_scope(self.model_scope_full):
        __ = self.forward_train(images)
        self.vars_full = get_vars_by_scope(self.model_scope_full)
        self.saver_full = tf.train.Saver(self.vars_full['all'])
        self.save_path_full = FLAGS.save_path

      # model definition - channel-pruned model
      with tf.variable_scope(self.model_scope_prnd):
        logits_prnd = self.forward_train(images)
        self.vars_prnd = get_vars_by_scope(self.model_scope_prnd)
        self.maskable_var_names = [var.name for var in self.vars_prnd['maskable']]
        self.global_step = tf.train.get_or_create_global_step()
        self.saver_prnd_train = tf.train.Saver(self.vars_prnd['all'] + [self.global_step])

        # loss & extra evaluation metrics
        loss, metrics = self.calc_loss(labels, logits_prnd, self.vars_prnd['trainable'])
        if FLAGS.enbl_dst:
          loss += self.helper_dst.calc_loss(logits_prnd, logits_dst)
        tf.summary.scalar('loss', loss)
        for key, value in metrics.items():
          tf.summary.scalar(key, value)

        # learning rate schedule
        lrn_rate, self.nb_iters_train = self.setup_lrn_rate(self.global_step)

        # overall pruning ratios of trainable & maskable variables
        pr_trainable = calc_prune_ratio(self.vars_prnd['trainable'])
        pr_maskable = calc_prune_ratio(self.vars_prnd['maskable'])
        tf.summary.scalar('pr_trainable', pr_trainable)
        tf.summary.scalar('pr_maskable', pr_maskable)

        # create masks and corresponding operations for channel pruning
        self.masks = []
        self.mask_updt_ops = []
        for idx, var in enumerate(self.vars_prnd['maskable']):
          tf.logging.info('creating a pruning mask for {} of size {}'.format(var.name, var.shape))
          name = '/'.join(var.name.split('/')[1:]).replace(':0', '_mask')
          self.masks += [tf.get_variable(name, initializer=tf.ones(var.shape), trainable=False)]
          var_norm = tf.sqrt(tf.reduce_sum(tf.square(var), axis=[0, 1, 3], keepdims=True))
          mask_vec = tf.cast(var_norm > 0.0, tf.float32)
          mask_new = tf.tile(mask_vec, [var.shape[0], var.shape[1], 1, var.shape[3]])
          self.mask_updt_ops += [self.masks[-1].assign(mask_new)]

        # build extra losses for regression & discrimination
        self.reg_losses = self.__build_extra_losses()
        self.nb_layers = len(self.reg_losses)
        for idx, reg_loss in enumerate(self.reg_losses):
          tf.summary.scalar('reg_loss_%d' % idx, reg_loss)

        # obtain the full list of trainable variables & update operations
        self.vars_all = tf.get_collection(
          tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_scope_prnd)
        self.trainable_vars_all = tf.get_collection(
          tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_prnd)
        self.update_ops_all = tf.get_collection(
          tf.GraphKeys.UPDATE_OPS, scope=self.model_scope_prnd)

        # TF operations for initializing the channel-pruned model
        init_ops = []
        with tf.control_dependencies([tf.variables_initializer(self.vars_all)]):
          for var_full, var_prnd in zip(self.vars_full['all'], self.vars_prnd['all']):
            init_ops += [var_prnd.assign(var_full)]
        init_ops += [self.global_step.initializer]
        self.init_op = tf.group(init_ops)

        # TF operations for layer-wise, block-wise, and whole-network fine-tuning
        self.layer_ops, self.lrn_rates_pgd, self.prune_perctls = self.__build_layer_ops()
        self.train_op, self.init_opt_op = self.__build_network_ops(loss, lrn_rate)

      # TF operations for logging & summarizing
      self.sess_train = sess
      self.summary_op = tf.summary.merge_all()
      self.log_op = [lrn_rate, loss, pr_trainable, pr_maskable] + list(metrics.values())
      self.log_op_names = ['lr', 'loss', 'pr_trn', 'pr_msk'] + list(metrics.keys())
      if FLAGS.enbl_multi_gpu:
        self.bcast_op = mgw.broadcast_global_variables(0)

  def __build_eval(self):
    """Build the evaluation graph."""

    with tf.Graph().as_default():
      # create a TF session for the current graph
      config = tf.ConfigProto()
      config.gpu_options.visible_device_list = str(mgw.local_rank() if FLAGS.enbl_multi_gpu else 0)  # pylint: disable=no-member
      self.sess_eval = tf.Session(config=config)

      # data input pipeline
      with tf.variable_scope(self.data_scope):
        iterator = self.build_dataset_eval()
        images, labels = iterator.get_next()

      # model definition - distilled model
      if FLAGS.enbl_dst:
        logits_dst = self.helper_dst.calc_logits(self.sess_eval, images)

      # model definition - channel-pruned model
      with tf.variable_scope(self.model_scope_prnd):
        logits = self.forward_eval(images)
        vars_prnd = get_vars_by_scope(self.model_scope_prnd)
        global_step = tf.train.get_or_create_global_step()
        self.saver_prnd_eval = tf.train.Saver(vars_prnd['all'] + [global_step])

        # loss & extra evaluation metrics
        loss, metrics = self.calc_loss(labels, logits, vars_prnd['trainable'])
        if FLAGS.enbl_dst:
          loss += self.helper_dst.calc_loss(logits, logits_dst)

        # overall pruning ratios of trainable & maskable variables
        pr_trainable = calc_prune_ratio(vars_prnd['trainable'])
        pr_maskable = calc_prune_ratio(vars_prnd['maskable'])

        # TF operations for evaluation
        self.eval_op = [loss, pr_trainable, pr_maskable] + list(metrics.values())
        self.eval_op_names = ['loss', 'pr_trn', 'pr_msk'] + list(metrics.keys())
        self.outputs_eval = logits

      # add input & output tensors to certain collections
      tf.add_to_collection('images_final', images)
      tf.add_to_collection('logits_final', logits)

  def __build_extra_losses(self):
    """Build extra losses for regression.

    Returns:
    * reg_losses: list of regression losses (one per layer)
    """

    # insert additional losses to intermediate layers
    pattern = re.compile('Conv2D$')
    core_ops_full = get_ops_by_scope_n_pattern(self.model_scope_full, pattern)
    core_ops_prnd = get_ops_by_scope_n_pattern(self.model_scope_prnd, pattern)
    reg_losses = []
    for core_op_full, core_op_prnd in zip(core_ops_full, core_ops_prnd):
      reg_losses += [tf.nn.l2_loss(core_op_full.outputs[0] - core_op_prnd.outputs[0])]

    return reg_losses

  def __build_layer_ops(self):
    """Build layer-wise fine-tuning operations.

    Returns:
    * layer_ops: list of training and initialization operations for each layer
    * lrn_rates_pgd: list of layer-wise learning rate
    * prune_perctls: list of layer-wise pruning percentiles
    """

    layer_ops = []
    lrn_rates_pgd = []  # list of layer-wise learning rate
    prune_perctls = []  # list of layer-wise pruning percentiles
    for idx, var_prnd in enumerate(self.vars_prnd['maskable']):
      # create placeholders
      lrn_rate_pgd = tf.placeholder(tf.float32, shape=[], name='lrn_rate_pgd_%d' % idx)
      prune_perctl = tf.placeholder(tf.float32, shape=[], name='prune_perctl_%d' % idx)

      # select channels for the current convolutional layer
      optimizer = tf.train.GradientDescentOptimizer(lrn_rate_pgd)
      if FLAGS.enbl_multi_gpu:
        optimizer = mgw.DistributedOptimizer(optimizer)
      grads = optimizer.compute_gradients(self.reg_losses[idx], [var_prnd])
      with tf.control_dependencies(self.update_ops_all):
        var_prnd_new = var_prnd - lrn_rate_pgd * grads[0][0]
        var_norm = tf.sqrt(tf.reduce_sum(tf.square(var_prnd_new), axis=[0, 1, 3], keepdims=True))
        threshold = tf.contrib.distributions.percentile(var_norm, prune_perctl)
        shrk_vec = tf.maximum(1.0 - threshold / var_norm, 0.0)
        prune_op = var_prnd.assign(var_prnd_new * shrk_vec)

      # fine-tune with selected channels only
      optimizer_base = tf.train.AdamOptimizer(FLAGS.cpg_lrn_rate_adam)
      if not FLAGS.enbl_multi_gpu:
        optimizer = optimizer_base
      else:
        optimizer = mgw.DistributedOptimizer(optimizer_base)
      grads_origin = optimizer.compute_gradients(self.reg_losses[idx], [var_prnd])
      grads_pruned = self.__calc_grads_pruned(grads_origin)
      with tf.control_dependencies(self.update_ops_all):
        finetune_op = optimizer.apply_gradients(grads_pruned)
      init_opt_op = tf.variables_initializer(optimizer_base.variables())

      # append layer-wise operations & variables
      layer_ops += [{'prune': prune_op, 'finetune': finetune_op, 'init_opt': init_opt_op}]
      lrn_rates_pgd += [lrn_rate_pgd]
      prune_perctls += [prune_perctl]

    return layer_ops, lrn_rates_pgd, prune_perctls

  def __build_network_ops(self, loss, lrn_rate):
    """Build network training operations.

    Returns:
    * train_op: training operation of the whole network
    * init_opt_op: initialization operation of the whole network's optimizer
    """

    optimizer_base = tf.train.MomentumOptimizer(lrn_rate, FLAGS.momentum)
    if not FLAGS.enbl_multi_gpu:
      optimizer = optimizer_base
    else:
      optimizer = mgw.DistributedOptimizer(optimizer_base)
    grads_origin = optimizer.compute_gradients(loss, self.trainable_vars_all)
    grads_pruned = self.__calc_grads_pruned(grads_origin)
    with tf.control_dependencies(self.update_ops_all):
      train_op = optimizer.apply_gradients(grads_pruned, global_step=self.global_step)
    init_opt_op = tf.variables_initializer(optimizer_base.variables())

    return train_op, init_opt_op

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

  def __choose_channels(self):  # pylint: disable=too-many-locals
    """Choose channels for all convolutional layers."""

    # obtain each layer's pruning ratio
    if FLAGS.cpg_prune_ratio_type == 'uniform':
      ratio_list = [FLAGS.cpg_prune_ratio] * self.nb_layers
      if FLAGS.cpg_skip_ht_layers:
        ratio_list[0] = 0.0
        ratio_list[-1] = 0.0
    elif FLAGS.cpg_prune_ratio_type == 'list':
      with open(FLAGS.cpg_prune_ratio_file, 'r') as i_file:
        i_line = i_file.readline().strip()
        ratio_list = [float(sub_str) for sub_str in i_line.split(',')]
    else:
      raise ValueError('unrecognized pruning ratio type: ' + FLAGS.cpg_prune_ratio_type)

    # select channels for all convolutional layers
    nb_workers = mgw.size() if FLAGS.enbl_multi_gpu else 1
    nb_iters_layer = int(FLAGS.cpg_nb_iters_layer / nb_workers)
    for idx_layer in range(self.nb_layers):
      # skip if no pruning is required
      if ratio_list[idx_layer] == 0.0:
        continue
      if self.is_primary_worker('global'):
        tf.logging.info('layer #%d: pr = %.2f (target)' % (idx_layer, ratio_list[idx_layer]))
        tf.logging.info('mask.shape = {}'.format(self.masks[idx_layer].shape))

      # select channels for the current convolutional layer
      time_prev = timer()
      reg_loss_prev = 0.0
      lrn_rate_pgd = FLAGS.cpg_lrn_rate_pgd_init
      for idx_iter in range(nb_iters_layer):
        # take a stochastic proximal gradient descent step
        prune_perctl = ratio_list[idx_layer] * 100.0 * (idx_iter + 1) / nb_iters_layer
        __, reg_loss = self.sess_train.run(
          [self.layer_ops[idx_layer]['prune'], self.reg_losses[idx_layer]],
          feed_dict={self.lrn_rates_pgd[idx_layer]: lrn_rate_pgd,
                     self.prune_perctls[idx_layer]: prune_perctl})
        mask = self.sess_train.run(self.masks[idx_layer])
        if self.is_primary_worker('global'):
          nb_chns_nnz = np.count_nonzero(np.sum(mask, axis=(0, 1, 3)))
          tf.logging.info('iter %d: nnz-chns = %d | loss = %.2e | lr = %.2e | percentile = %.2f'
                          % (idx_iter + 1, nb_chns_nnz, reg_loss, lrn_rate_pgd, prune_perctl))

        # adjust the learning rate
        if reg_loss < reg_loss_prev:
          lrn_rate_pgd *= FLAGS.cpg_lrn_rate_pgd_incr
        else:
          lrn_rate_pgd *= FLAGS.cpg_lrn_rate_pgd_decr
        reg_loss_prev = reg_loss

      # fine-tune with selected channels only
      self.sess_train.run(self.mask_updt_ops[idx_layer])
      for idx_iter in range(nb_iters_layer):
        __, reg_loss = self.sess_train.run(
          [self.layer_ops[idx_layer]['finetune'], self.reg_losses[idx_layer]])
        mask = self.sess_train.run(self.masks[idx_layer])
        if self.is_primary_worker('global'):
          nb_chns_nnz = np.count_nonzero(np.sum(mask, axis=(0, 1, 3)))
          tf.logging.info('iter %d: nnz-chns = %d | loss = %.2e'
                          % (idx_iter + 1, nb_chns_nnz, reg_loss))

      # re-compute the pruning ratio
      mask_vec = np.mean(np.square(self.sess_train.run(self.masks[idx_layer])), axis=(0, 1, 3))
      prune_ratio = 1.0 - float(np.count_nonzero(mask_vec)) / mask_vec.size
      if self.is_primary_worker('global'):
        tf.logging.info('layer #%d: pr = %.2f (actual) | time = %.2f'
                        % (idx_layer, prune_ratio, timer() - time_prev))

    # compute overall pruning ratios
    if self.is_primary_worker('global'):
      log_rslt = self.sess_train.run(self.log_op)
      log_str = ' | '.join(['%s = %.4e' % (name, value)
                            for name, value in zip(self.log_op_names, log_rslt)])

  def __save_model(self, is_train):
    """Save the current model for training or evaluation.

    Args:
    * is_train: whether to save a model for training
    """

    if is_train:
      save_path = self.saver_prnd_train.save(self.sess_train, FLAGS.cpg_save_path, self.global_step)
    else:
      save_path = self.saver_prnd_eval.save(self.sess_eval, FLAGS.cpg_save_path_eval)
    tf.logging.info('model saved to ' + save_path)

  def __restore_model(self, is_train):
    """Restore a model from the latest checkpoint files.

    Args:
    * is_train: whether to restore a model for training
    """

    save_path = tf.train.latest_checkpoint(os.path.dirname(FLAGS.cpg_save_path))
    if is_train:
      self.saver_prnd_train.restore(self.sess_train, save_path)
    else:
      self.saver_prnd_eval.restore(self.sess_eval, save_path)
    tf.logging.info('model restored from ' + save_path)

  def __monitor_progress(self, summary, log_rslt, idx_iter, time_step):
    """Monitor the training progress.

    Args:
    * summary: summary protocol buffer
    * log_rslt: logging operations' results
    * idx_iter: index of the training iteration
    * time_step: time step between two summary operations
    """

    # write summaries for TensorBoard visualization
    self.sm_writer.add_summary(summary, idx_iter)

    # compute the training speed
    speed = FLAGS.batch_size * FLAGS.summ_step / time_step
    if FLAGS.enbl_multi_gpu:
      speed *= mgw.size()

    # display monitored statistics
    log_str = ' | '.join(['%s = %.4e' % (name, value)
                          for name, value in zip(self.log_op_names, log_rslt)])
    tf.logging.info('iter #%d: %s | speed = %.2f pics / sec' % (idx_iter + 1, log_str, speed))
