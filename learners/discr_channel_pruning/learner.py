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
"""Discrimination-aware channel pruning learner."""

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

tf.app.flags.DEFINE_string('dcp_save_path', './models_dcp/model.ckpt', 'DCP: model\'s save path')
tf.app.flags.DEFINE_string('dcp_save_path_eval', './models_dcp_eval/model.ckpt',
                           'DCP: model\'s save path for evaluation')
tf.app.flags.DEFINE_float('dcp_prune_ratio', 0.5, 'DCP: target channel pruning ratio')
tf.app.flags.DEFINE_integer('dcp_nb_stages', 3, 'DCP: # of channel pruning stages')
tf.app.flags.DEFINE_float('dcp_lrn_rate_adam', 1e-3, 'DCP: Adam\'s learning rate')
tf.app.flags.DEFINE_integer('dcp_nb_iters_block', 10000, 'DCP: # of iterations for block-wise FT')
tf.app.flags.DEFINE_integer('dcp_nb_iters_layer', 500, 'DCP: # of iterations for layer-wise FT')

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

class DisChnPrunedLearner(AbstractLearner):  # pylint: disable=too-many-instance-attributes
  """Discrimination-aware channel pruning learner."""

  def __init__(self, sm_writer, model_helper):
    """Constructor function.

    Args:
    * sm_writer: TensorFlow's summary writer
    * model_helper: model helper with definitions of model & dataset
    """

    # class-independent initialization
    super(DisChnPrunedLearner, self).__init__(sm_writer, model_helper)

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
    self.sess_train.run(self.layer_init_opt_ops)  # initialization for layer-wise fine-tuning
    self.sess_train.run(self.block_init_opt_ops)  # initialization for block-wise fine-tuning
    if FLAGS.enbl_multi_gpu:
      self.sess_train.run(self.bcast_op)

    # choose discrimination-aware channels
    self.__choose_discr_chns()

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
    for idx_iter in range(nb_iters):
      eval_rslts[idx_iter] = self.sess_eval.run(self.eval_op)
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
        self.saver_prnd_train = tf.train.Saver(self.vars_prnd['all'])

        # loss & extra evaluation metrics
        loss_bsc, metrics = self.calc_loss(labels, logits_prnd, self.vars_prnd['trainable'])
        if not FLAGS.enbl_dst:
          loss_fnl = loss_bsc
        else:
          loss_fnl = loss_bsc + self.helper_dst.calc_loss(logits_prnd, logits_dst)
        tf.summary.scalar('loss_bsc', loss_bsc)
        tf.summary.scalar('loss_fnl', loss_fnl)
        for key, value in metrics.items():
          tf.summary.scalar(key, value)

        # learning rate schedule
        self.global_step = tf.train.get_or_create_global_step()
        lrn_rate, self.nb_iters_train = self.setup_lrn_rate(self.global_step)

        # overall pruning ratios of trainable & maskable variables
        pr_trainable = calc_prune_ratio(self.vars_prnd['trainable'])
        pr_maskable = calc_prune_ratio(self.vars_prnd['maskable'])
        tf.summary.scalar('pr_trainable', pr_trainable)
        tf.summary.scalar('pr_maskable', pr_maskable)

        # create masks and corresponding operations for channel pruning
        self.masks = []
        self.mask_deltas = []
        self.mask_init_ops = []
        self.mask_updt_ops = []
        self.prune_ops = []
        for idx, var in enumerate(self.vars_prnd['maskable']):
          name = '/'.join(var.name.split('/')[1:]).replace(':0', '_mask')
          self.masks += [tf.get_variable(name, initializer=tf.ones(var.shape), trainable=False)]
          name = '/'.join(var.name.split('/')[1:]).replace(':0', '_mask_delta')
          self.mask_deltas += [tf.placeholder(tf.float32, shape=var.shape, name=name)]
          self.mask_init_ops += [self.masks[idx].assign(tf.zeros(var.shape))]
          self.mask_updt_ops += [self.masks[idx].assign_add(self.mask_deltas[idx])]
          self.prune_ops += [var.assign(var * self.masks[idx])]

        # build extra losses for regression & discrimination
        self.reg_losses, self.dis_losses, self.idxs_layer_to_block = \
          self.__build_extra_losses(labels)
        self.dis_losses += [loss_bsc]  # append discrimination-aware loss for the last block
        self.nb_layers = len(self.reg_losses)
        self.nb_blocks = len(self.dis_losses)
        for idx, reg_loss in enumerate(self.reg_losses):
          tf.summary.scalar('reg_loss_%d' % idx, reg_loss)
        for idx, dis_loss in enumerate(self.dis_losses):
          tf.summary.scalar('dis_loss_%d' % idx, dis_loss)

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
        self.init_op = tf.group(init_ops)

        # TF operations for layer-wise, block-wise, and whole-network fine-tuning
        self.layer_train_ops, self.layer_init_opt_ops, self.grad_norms = self.__build_layer_ops()
        self.block_train_ops, self.block_init_opt_ops = self.__build_block_ops()
        self.train_op, self.init_opt_op = self.__build_network_ops(loss_fnl, lrn_rate)

      # TF operations for logging & summarizing
      self.sess_train = sess
      self.summary_op = tf.summary.merge_all()
      self.log_op = [lrn_rate, loss_fnl, pr_trainable, pr_maskable] + list(metrics.values())
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
        # loss & extra evaluation metrics
        logits = self.forward_eval(images)
        vars_prnd = get_vars_by_scope(self.model_scope_prnd)
        loss, metrics = self.calc_loss(labels, logits, vars_prnd['trainable'])
        if FLAGS.enbl_dst:
          loss += self.helper_dst.calc_loss(logits, logits_dst)

        # overall pruning ratios of trainable & maskable variables
        pr_trainable = calc_prune_ratio(vars_prnd['trainable'])
        pr_maskable = calc_prune_ratio(vars_prnd['maskable'])

        # TF operations for evaluation
        self.eval_op = [loss, pr_trainable, pr_maskable] + list(metrics.values())
        self.eval_op_names = ['loss', 'pr_trn', 'pr_msk'] + list(metrics.keys())
        self.saver_prnd_eval = tf.train.Saver(vars_prnd['all'])

      # add input & output tensors to certain collections
      tf.add_to_collection('images_final', images)
      tf.add_to_collection('logits_final', logits)

  def __build_extra_losses(self, labels):
    """Build extra losses for regression & discrimination.

    Args:
    * labels: one-hot label vectors

    Returns:
    * reg_losses: list of regression losses (one per layer)
    * dis_losses: list of discrimination-aware losses (one per layer)
    * idxs_layer_to_block: list of mappings from layer index to block index
    """

    # insert additional losses to intermediate layers
    pattern = re.compile('Conv2D$')
    core_ops_full = get_ops_by_scope_n_pattern(self.model_scope_full, pattern)
    core_ops_prnd = get_ops_by_scope_n_pattern(self.model_scope_prnd, pattern)
    nb_layers = len(core_ops_full)
    nb_blocks = int(FLAGS.dcp_nb_stages + 1)
    nb_layers_per_block = int(math.ceil((nb_layers + 1) / nb_blocks))
    reg_losses = []
    dis_losses = []
    idxs_layer_to_block = []
    for idx_layer in range(nb_layers):
      reg_losses += \
        [tf.nn.l2_loss(core_ops_full[idx_layer].outputs[0] - core_ops_prnd[idx_layer].outputs[0])]
      idxs_layer_to_block += [int(idx_layer / nb_layers_per_block)]
      if (idx_layer + 1) % nb_layers_per_block == 0:
        x = core_ops_prnd[idx_layer].outputs[0]
        x = tf.layers.batch_normalization(x, axis=3, training=True)
        x = tf.nn.relu(x)
        x = tf.reduce_mean(x, axis=[1, 2])
        x = tf.layers.dense(x, FLAGS.nb_classes)
        dis_losses += [tf.losses.softmax_cross_entropy(labels, x)]
    tf.logging.info('layer-to-block mapping: {}'.format(idxs_layer_to_block))

    return reg_losses, dis_losses, idxs_layer_to_block

  def __build_layer_ops(self):
    """Build layer-wise fine-tuning operations.

    Returns:
    * layer_train_ops: list of training operations for each layer
    * layer_init_opt_ops: list of initialization operations for each layer's optimizer
    * layer_grad_norms: list of gradient norm vectors for each layer
    """

    layer_train_ops = []
    layer_init_opt_ops = []
    grad_norms = []
    for idx, var_prnd in enumerate(self.vars_prnd['maskable']):
      optimizer_base = tf.train.AdamOptimizer(FLAGS.dcp_lrn_rate_adam)
      if not FLAGS.enbl_multi_gpu:
        optimizer = optimizer_base
      else:
        optimizer = mgw.DistributedOptimizer(optimizer_base)
      loss_all = self.reg_losses[idx] + self.dis_losses[self.idxs_layer_to_block[idx]]
      grads_origin = optimizer.compute_gradients(loss_all, [var_prnd])
      grads_pruned = self.__calc_grads_pruned(grads_origin)
      with tf.control_dependencies(self.update_ops_all):
        layer_train_ops += [optimizer.apply_gradients(grads_pruned)]
      layer_init_opt_ops += [tf.variables_initializer(optimizer_base.variables())]
      grad_norms += [tf.reduce_sum(grads_origin[0][0] ** 2, axis=[0, 1, 3])]

    return layer_train_ops, layer_init_opt_ops, grad_norms

  def __build_block_ops(self):
    """Build block-wise fine-tuning operations.

    Returns:
    * block_train_ops: list of training operations for each block
    * block_init_opt_ops: list of initialization operations for each block's optimizer
    """

    block_train_ops = []
    block_init_opt_ops = []
    for dis_loss in self.dis_losses:
      optimizer_base = tf.train.AdamOptimizer(FLAGS.dcp_lrn_rate_adam)
      if not FLAGS.enbl_multi_gpu:
        optimizer = optimizer_base
      else:
        optimizer = mgw.DistributedOptimizer(optimizer_base)
      loss_all = dis_loss + self.dis_losses[-1]  # current stage + final loss
      grads_origin = optimizer.compute_gradients(loss_all, self.trainable_vars_all)
      grads_pruned = self.__calc_grads_pruned(grads_origin)
      with tf.control_dependencies(self.update_ops_all):
        block_train_ops += [optimizer.apply_gradients(grads_pruned)]
      block_init_opt_ops += [tf.variables_initializer(optimizer_base.variables())]

    return block_train_ops, block_init_opt_ops

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
    loss_all = tf.add_n(self.dis_losses[:-1]) * 0 + loss  # all stages + final loss
    grads_origin = optimizer.compute_gradients(loss_all, self.trainable_vars_all)
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

  def __choose_discr_chns(self):  # pylint: disable=too-many-locals
    """Choose discrimination-aware channels."""

    # select the most discriminative channels through multiple stages
    nb_workers = mgw.size() if FLAGS.enbl_multi_gpu else 1
    nb_iters_block = int(FLAGS.dcp_nb_iters_block / nb_workers)
    nb_iters_layer = int(FLAGS.dcp_nb_iters_layer / nb_workers)
    for idx_block in range(self.nb_blocks):
      # fine-tune the current block
      for idx_iter in range(nb_iters_block):
        if (idx_iter + 1) % FLAGS.summ_step != 0:
          self.sess_train.run(self.block_train_ops[idx_block])
        else:
          summary, __ = self.sess_train.run([self.summary_op, self.block_train_ops[idx_block]])
          if self.is_primary_worker('global'):
            tf.logging.info('iter #%d: writing TF-summary to file' % idx_iter)
            self.sm_writer.add_summary(summary, nb_iters_block * idx_block + idx_iter)

      # select the most discriminative channels for each layer
      for idx_layer in range(1, self.nb_layers):  # do not prune the first layer
        if self.idxs_layer_to_block[idx_layer] != idx_block:
          continue

        # initialize the gradient mask
        mask_shape = self.sess_train.run(tf.shape(self.masks[idx_layer]))
        if self.is_primary_worker('global'):
          tf.logging.info('layer #{}: mask\'s shape is {}'.format(idx_layer, mask_shape))
        nb_chns = mask_shape[2]
        idxs_chn_keep = []
        grad_norm_mask = np.ones(nb_chns)

        # sequentially add the most important channel to the non-pruned set
        is_first_entry = True
        while is_first_entry or prune_ratio > FLAGS.dcp_prune_ratio:
          # choose the most important channel
          grad_norm = self.sess_train.run(self.grad_norms[idx_layer])
          idx_chn = np.argmax((grad_norm + 1e-8) * grad_norm_mask)  # avoid all-zero gradients
          assert idx_chn not in idxs_chn_keep, 'channel #%d already in the non-pruned set' % idx_chn
          idxs_chn_keep += [idx_chn]
          grad_norm_mask[idx_chn] = 0.0
          if self.is_primary_worker('global'):
            tf.logging.info('adding channel #%d to the non-pruned set' % idx_chn)

          # update the mask
          mask_delta = np.zeros(mask_shape)
          mask_delta[:, :, idx_chn, :] = 1.0
          if is_first_entry:
            is_first_entry = False
            self.sess_train.run(self.mask_init_ops[idx_layer])
          self.sess_train.run(self.mask_updt_ops[idx_layer],
                              feed_dict={self.mask_deltas[idx_layer]: mask_delta})
          self.sess_train.run(self.prune_ops[idx_layer])

          # fine-tune the current layer
          for idx_iter in range(nb_iters_layer):
            self.sess_train.run(self.layer_train_ops[idx_layer])

          # re-compute the pruning ratio
          mask_vec = np.sum(self.sess_train.run(self.masks[idx_layer]), axis=(0, 1, 3))
          prune_ratio = 1.0 - float(np.count_nonzero(mask_vec)) / mask_vec.size
          if self.is_primary_worker('global'):
            tf.logging.info('layer #%d: prune_ratio = %.4f' % (idx_layer, prune_ratio))

      # compute overall pruning ratios
      if self.is_primary_worker('global'):
        log_rslt = self.sess_train.run(self.log_op)
        log_str = ' | '.join(['%s = %.4e' % (name, value)
                              for name, value in zip(self.log_op_names, log_rslt)])
        tf.logging.info('block #%d: %s' % (idx_block + 1, log_str))

  def __save_model(self, is_train):
    """Save the current model for training or evaluation.

    Args:
    * is_train: whether to save a model for training
    """

    if is_train:
      save_path = self.saver_prnd_train.save(self.sess_train, FLAGS.dcp_save_path, self.global_step)
    else:
      save_path = self.saver_prnd_eval.save(self.sess_eval, FLAGS.dcp_save_path_eval)
    tf.logging.info('model saved to ' + save_path)

  def __restore_model(self, is_train):
    """Restore a model from the latest checkpoint files.

    Args:
    * is_train: whether to restore a model for training
    """

    save_path = tf.train.latest_checkpoint(os.path.dirname(FLAGS.dcp_save_path))
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
