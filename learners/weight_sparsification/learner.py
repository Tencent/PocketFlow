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
"""Weight sparsification learner."""

import os
from timeit import default_timer as timer
import numpy as np
import tensorflow as tf

from learners.abstract_learner import AbstractLearner
from learners.distillation_helper import DistillationHelper
from learners.weight_sparsification.pr_optimizer import PROptimizer
from learners.weight_sparsification.utils import get_maskable_vars
from utils.multi_gpu_wrapper import MultiGpuWrapper as mgw

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('ws_save_path', './models_ws/model.ckpt', 'WS: model\'s save path')
tf.app.flags.DEFINE_float('ws_prune_ratio', 0.75, 'WS: target pruning ratio')
tf.app.flags.DEFINE_string('ws_prune_ratio_prtl', 'optimal',
                           'WS: pruning ratio protocol (\'uniform\' | \'heurist\' | \'optimal\')')
tf.app.flags.DEFINE_integer('ws_nb_rlouts', 200, 'WS: # of roll-outs for the RL agent')
tf.app.flags.DEFINE_integer('ws_nb_rlouts_min', 50,
                            'WS: minimal # of roll-outs for the RL agent to start training')
tf.app.flags.DEFINE_string('ws_reward_type', 'single-obj',
                           'WS: reward type (\'single-obj\' OR \'multi-obj\')')
tf.app.flags.DEFINE_float('ws_lrn_rate_rg', 3e-2, 'WS: learning rate for layerwise regression')
tf.app.flags.DEFINE_integer('ws_nb_iters_rg', 20, 'WS: # of iterations for layerwise regression')
tf.app.flags.DEFINE_float('ws_lrn_rate_ft', 3e-4, 'WS: learning rate for global fine-tuning')
tf.app.flags.DEFINE_integer('ws_nb_iters_ft', 400, 'WS: # of iterations for global fine-tuning')
tf.app.flags.DEFINE_integer('ws_nb_iters_feval', 25, 'WS: # of iterations for fast evaluation')
tf.app.flags.DEFINE_float('ws_prune_ratio_exp', 3.0, 'WS: pruning ratio\'s exponent term')
tf.app.flags.DEFINE_float('ws_iter_ratio_beg', 0.1, 'WS: iteration ratio (at starting time)')
tf.app.flags.DEFINE_float('ws_iter_ratio_end', 0.5, 'WS: iteration ratio (at ending time)')
tf.app.flags.DEFINE_float('ws_mask_update_step', 500, 'WS: step size for updating the pruning mask')

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

class WeightSparseLearner(AbstractLearner):  # pylint: disable=too-many-instance-attributes
  """Weight sparsification learner."""

  def __init__(self, sm_writer, model_helper):
    """Constructor function.

    Args:
    * sm_writer: TensorFlow's summary writer
    * model_helper: model helper with definitions of model & dataset
    """

    # class-independent initialization
    super(WeightSparseLearner, self).__init__(sm_writer, model_helper)

    # define the scope for masks
    self.mask_scope = 'mask'

    # compute the optimal pruning ratios (only when the execution mode is 'train')
    if FLAGS.exec_mode == 'train':
      pr_optimizer = PROptimizer(model_helper, self.mpi_comm)
      if FLAGS.ws_prune_ratio_prtl == 'optimal':
        if self.is_primary_worker('local'):
          self.download_model()  # pre-trained model is required
        self.auto_barrier()
        tf.logging.info('model files: ' + ', '.join(os.listdir('./models')))
      self.var_names_n_prune_ratios = pr_optimizer.run()

    # class-dependent initialization
    if FLAGS.enbl_dst:
      self.helper_dst = DistillationHelper(sm_writer, model_helper, self.mpi_comm)
    if FLAGS.exec_mode == 'train':
      self.__build_train()  # only when the execution mode is 'train'
    self.__build_eval()  # needed whatever the execution mode is

  def train(self):
    """Train a model and periodically produce checkpoint files."""

    # initialization
    self.sess_train.run(self.init_op)
    if FLAGS.enbl_multi_gpu:
      self.sess_train.run(self.bcast_op)

    # train the model through iterations and periodically save & evaluate the model
    last_mask_applied = False
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

      # apply pruning
      if (idx_iter + 1) % FLAGS.ws_mask_update_step == 0:
        iter_ratio = float(idx_iter + 1) / self.nb_iters_train
        if iter_ratio >= FLAGS.ws_iter_ratio_beg:
          if iter_ratio <= FLAGS.ws_iter_ratio_end:
            self.sess_train.run([self.prune_op, self.init_opt_op])
          elif not last_mask_applied:
            last_mask_applied = True
            self.sess_train.run([self.prune_op, self.init_opt_op])

      # save the model at certain steps
      if self.is_primary_worker('global') and (idx_iter + 1) % FLAGS.save_step == 0:
        self.__save_model()
        self.evaluate()

    # save the final model
    if self.is_primary_worker('global'):
      self.__save_model()
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

  def __build_train(self):  # pylint: disable=too-many-locals
    """Build the training graph."""

    with tf.Graph().as_default():
      # create a TF session for the current graph
      config = tf.ConfigProto()
      if FLAGS.enbl_multi_gpu:
        config.gpu_options.visible_device_list = str(mgw.local_rank())  # pylint: disable=no-member
      else:
        config.gpu_options.visible_device_list = '0'  # pylint: disable=no-member
      sess = tf.Session(config=config)

      # data input pipeline
      with tf.variable_scope(self.data_scope):
        iterator = self.build_dataset_train()
        images, labels = iterator.get_next()

      # model definition - distilled model
      if FLAGS.enbl_dst:
        logits_dst = self.helper_dst.calc_logits(sess, images)

      # model definition - weight-sparsified model
      with tf.variable_scope(self.model_scope):
        # loss & extra evaluation metrics
        logits = self.forward_train(images)
        self.maskable_var_names = [var.name for var in self.maskable_vars]
        loss, metrics = self.calc_loss(labels, logits, self.trainable_vars)
        if FLAGS.enbl_dst:
          loss += self.helper_dst.calc_loss(logits, logits_dst)
        tf.summary.scalar('loss', loss)
        for key, value in metrics.items():
          tf.summary.scalar(key, value)

        # learning rate schedule
        self.global_step = tf.train.get_or_create_global_step()
        lrn_rate, self.nb_iters_train = self.setup_lrn_rate(self.global_step)

        # overall pruning ratios of trainable & maskable variables
        pr_trainable = calc_prune_ratio(self.trainable_vars)
        pr_maskable = calc_prune_ratio(self.maskable_vars)
        tf.summary.scalar('pr_trainable', pr_trainable)
        tf.summary.scalar('pr_maskable', pr_maskable)

        # build masks and corresponding operations for weight sparsification
        self.masks, self.prune_op = self.__build_masks()

        # optimizer & gradients
        optimizer_base = tf.train.MomentumOptimizer(lrn_rate, FLAGS.momentum)
        if not FLAGS.enbl_multi_gpu:
          optimizer = optimizer_base
        else:
          optimizer = mgw.DistributedOptimizer(optimizer_base)
        grads_origin = optimizer.compute_gradients(loss, self.trainable_vars)
        grads_pruned = self.__calc_grads_pruned(grads_origin)

      # TF operations & model saver
      self.sess_train = sess
      with tf.control_dependencies(self.update_ops):
        self.train_op = optimizer.apply_gradients(grads_pruned, global_step=self.global_step)
      self.summary_op = tf.summary.merge_all()
      self.log_op = [lrn_rate, loss, pr_trainable, pr_maskable] + list(metrics.values())
      self.log_op_names = ['lr', 'loss', 'pr_trn', 'pr_msk'] + list(metrics.keys())
      self.init_op = tf.variables_initializer(self.vars)
      self.init_opt_op = tf.variables_initializer(optimizer_base.variables())
      if FLAGS.enbl_multi_gpu:
        self.bcast_op = mgw.broadcast_global_variables(0)
      self.saver_train = tf.train.Saver(self.vars)

  def __build_eval(self):
    """Build the evaluation graph."""

    with tf.Graph().as_default():
      # create a TF session for the current graph
      config = tf.ConfigProto()
      if FLAGS.enbl_multi_gpu:
        config.gpu_options.visible_device_list = str(mgw.local_rank())  # pylint: disable=no-member
      else:
        config.gpu_options.visible_device_list = '0'  # pylint: disable=no-member
      self.sess_eval = tf.Session(config=config)

      # data input pipeline
      with tf.variable_scope(self.data_scope):
        iterator = self.build_dataset_eval()
        images, labels = iterator.get_next()

      # model definition - distilled model
      if FLAGS.enbl_dst:
        logits_dst = self.helper_dst.calc_logits(self.sess_eval, images)

      # model definition - weight-sparsified model
      with tf.variable_scope(self.model_scope):
        # loss & extra evaluation metrics
        logits = self.forward_eval(images)
        loss, metrics = self.calc_loss(labels, logits, self.trainable_vars)
        if FLAGS.enbl_dst:
          loss += self.helper_dst.calc_loss(logits, logits_dst)

        # overall pruning ratios of trainable & maskable variables
        pr_trainable = calc_prune_ratio(self.trainable_vars)
        pr_maskable = calc_prune_ratio(self.maskable_vars)

        # TF operations for evaluation
        self.eval_op = [loss, pr_trainable, pr_maskable] + list(metrics.values())
        self.eval_op_names = ['loss', 'pr_trn', 'pr_msk'] + list(metrics.keys())
        self.saver_eval = tf.train.Saver(self.vars)

  def __build_masks(self):
    """build masks and corresponding operations for weight sparsification.

    Returns:
    * masks: list of masks for weight sparsification
    * prune_op: pruning operation
    """

    masks, prune_ops = [], []
    with tf.variable_scope(self.mask_scope):
      for var, var_name_n_prune_ratio in zip(self.maskable_vars, self.var_names_n_prune_ratios):
        # obtain the dynamic pruning ratio
        assert var.name == var_name_n_prune_ratio[0], \
            'unmatched variable names: %s vs. %s' % (var.name, var_name_n_prune_ratio[0])
        prune_ratio = self.__calc_prune_ratio_dyn(var_name_n_prune_ratio[1])

        # create a mask and non-masked backup for each variable
        name = var.name.replace(':0', '_mask')
        mask = tf.get_variable(name, initializer=tf.ones(var.shape), trainable=False)
        name = var.name.replace(':0', '_var_bkup')
        var_bkup = tf.get_variable(name, initializer=var.initialized_value(), trainable=False)

        # create update operations
        var_bkup_update_op = var_bkup.assign(tf.where(mask > 0.5, var, var_bkup))
        with tf.control_dependencies([var_bkup_update_op]):
          mask_thres = tf.contrib.distributions.percentile(tf.abs(var_bkup), prune_ratio * 100)
          mask_update_op = mask.assign(tf.cast(tf.abs(var_bkup) > mask_thres, tf.float32))
        with tf.control_dependencies([mask_update_op]):
          prune_op = var.assign(var_bkup * mask)

        # record pruning masks & operations
        masks += [mask]
        prune_ops += [prune_op]

    return masks, tf.group(prune_ops)

  def __calc_prune_ratio_dyn(self, prune_ratio_fnl):
    """Calculate the dynamic pruning ratio.

    Args:
    * prune_ratio_fnl: final pruning ratio

    Returns:
    * prune_ratio_dyn: dynamic pruning ratio
    """

    idx_iter_beg = int(self.nb_iters_train * FLAGS.ws_iter_ratio_beg)
    idx_iter_end = int(self.nb_iters_train * FLAGS.ws_iter_ratio_end)
    base = tf.cast(self.global_step - idx_iter_beg, tf.float32) / (idx_iter_end - idx_iter_beg)
    base = tf.minimum(1.0, tf.maximum(0.0, base))
    prune_ratio_dyn = prune_ratio_fnl * (1.0 - tf.pow(1.0 - base, FLAGS.ws_prune_ratio_exp))

    return prune_ratio_dyn

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

  def __save_model(self):
    """Save the current model."""

    save_path = self.saver_train.save(self.sess_train, FLAGS.ws_save_path, self.global_step)
    tf.logging.info('model saved to ' + save_path)

  def __restore_model(self, is_train):
    """Restore a model from the latest checkpoint files.

    Args:
    * is_train: whether to restore a model for training
    """

    save_path = tf.train.latest_checkpoint(os.path.dirname(FLAGS.ws_save_path))
    if is_train:
      self.saver_train.restore(self.sess_train, save_path)
    else:
      self.saver_eval.restore(self.sess_eval, save_path)
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

  @property
  def maskable_vars(self):
    """List of all maskable variables."""

    return get_maskable_vars(self.trainable_vars)
