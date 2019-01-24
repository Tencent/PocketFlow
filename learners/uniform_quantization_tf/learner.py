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
"""Uniform quantization learner with TensorFlow's quantization APIs."""

import os
from timeit import default_timer as timer
import numpy as np
import tensorflow as tf

from learners.abstract_learner import AbstractLearner
from learners.distillation_helper import DistillationHelper
from learners.uniform_quantization_tf.utils import find_unquant_act_nodes
from learners.uniform_quantization_tf.utils import insert_quant_op
from utils.multi_gpu_wrapper import MultiGpuWrapper as mgw

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('uqtf_save_path', './models_uqtf/model.ckpt',
                           'UQ-TF: model\'s save path')
tf.app.flags.DEFINE_string('uqtf_save_path_eval', './models_uqtf_eval/model.ckpt',
                           'UQ-TF: model\'s save path for evaluation')
tf.app.flags.DEFINE_integer('uqtf_weight_bits', 8, 'UQ-TF: # of bits for weight quantization')
tf.app.flags.DEFINE_integer('uqtf_activation_bits', 8,
                            'UQ-TF: # of bits for activation quantization')
tf.app.flags.DEFINE_integer('uqtf_quant_delay', 0,
                            'UQ-TF: # of steps after which weights and activations are quantized')
tf.app.flags.DEFINE_integer('uqtf_freeze_bn_delay', None,
                            'UT-TF: # of steps after which moving mean and variance are frozen \
                            and used instead of batch statistics during training.')
tf.app.flags.DEFINE_float('uqtf_lrn_rate_dcy', 1e-2, 'UQ-TF: learning rate\'s decaying factor')
tf.app.flags.DEFINE_boolean('uqtf_enbl_manual_quant', False,
                            'UQ-TF: enable manually inserting quantization operations')

def get_vars_by_scope(scope):
  """Get list of variables within certain name scope.

  Args:
  * scope: name scope

  Returns:
  * vars_dict: dictionary of list of all and trainable variables
  """

  vars_dict = {}
  vars_dict['all'] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
  vars_dict['trainable'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

  return vars_dict

class UniformQuantTFLearner(AbstractLearner):  # pylint: disable=too-many-instance-attributes
  """Uniform quantization learner with TensorFlow's quantization APIs."""

  def __init__(self, sm_writer, model_helper):
    """Constructor function.

    Args:
    * sm_writer: TensorFlow's summary writer
    * model_helper: model helper with definitions of model & dataset
    """

    # class-independent initialization
    super(UniformQuantTFLearner, self).__init__(sm_writer, model_helper)

    # define scopes for full & uniform quantized models
    self.model_scope_full = 'model'
    self.model_scope_quan = 'quant_model'

    # download the pre-trained model
    if self.is_primary_worker('local'):
      self.download_model()  # pre-trained model is required
    self.auto_barrier()
    tf.logging.info('model files: ' + ', '.join(os.listdir('./models')))

    # detect unquantized activations nodes
    self.unquant_node_names = []
    if FLAGS.uqtf_enbl_manual_quant:
      self.unquant_node_names = find_unquant_act_nodes(
        model_helper, self.data_scope, self.model_scope_quan, self.mpi_comm)
    tf.logging.info('unquantized activation nodes: {}'.format(self.unquant_node_names))

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
    if FLAGS.enbl_multi_gpu:
      self.sess_train.run(self.bcast_op)

    # train the model through iterations and periodically save & evaluate the model
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

    with tf.Graph().as_default() as graph:
      # create a TF session for the current graph
      config = tf.ConfigProto()
      config.gpu_options.visible_device_list = str(mgw.local_rank() if FLAGS.enbl_multi_gpu else 0)  # pylint: disable=no-member
      config.gpu_options.allow_growth = True  # pylint: disable=no-member
      sess = tf.Session(config=config)

      # data input pipeline
      with tf.variable_scope(self.data_scope):
        iterator = self.build_dataset_train()
        images, labels = iterator.get_next()

      # model definition - uniform quantized model - part 1
      with tf.variable_scope(self.model_scope_quan):
        logits_quan = self.forward_train(images)
        if not isinstance(logits_quan, dict):
          outputs = tf.nn.softmax(logits_quan)
        else:
          outputs = tf.nn.softmax(logits_quan['cls_pred'])
        tf.contrib.quantize.experimental_create_training_graph(
          weight_bits=FLAGS.uqtf_weight_bits,
          activation_bits=FLAGS.uqtf_activation_bits,
          quant_delay=FLAGS.uqtf_quant_delay,
          freeze_bn_delay=FLAGS.uqtf_freeze_bn_delay,
          scope=self.model_scope_quan)
        for node_name in self.unquant_node_names:
          insert_quant_op(graph, node_name, is_train=True)
        self.vars_quan = get_vars_by_scope(self.model_scope_quan)
        self.global_step = tf.train.get_or_create_global_step()
        self.saver_quan_train = tf.train.Saver(self.vars_quan['all'] + [self.global_step])

      # model definition - distilled model
      if FLAGS.enbl_dst:
        logits_dst = self.helper_dst.calc_logits(sess, images)

      # model definition - full model
      with tf.variable_scope(self.model_scope_full):
        __ = self.forward_train(images)
        self.vars_full = get_vars_by_scope(self.model_scope_full)
        self.saver_full = tf.train.Saver(self.vars_full['all'])
        self.save_path_full = FLAGS.save_path

      # model definition - uniform quantized model - part 2
      with tf.variable_scope(self.model_scope_quan):
        # loss & extra evaluation metrics
        loss_bsc, metrics = self.calc_loss(labels, logits_quan, self.vars_quan['trainable'])
        if not FLAGS.enbl_dst:
          loss_fnl = loss_bsc
        else:
          loss_fnl = loss_bsc + self.helper_dst.calc_loss(logits_quan, logits_dst)
        tf.summary.scalar('loss_bsc', loss_bsc)
        tf.summary.scalar('loss_fnl', loss_fnl)
        for key, value in metrics.items():
          tf.summary.scalar(key, value)

        # learning rate schedule
        lrn_rate, self.nb_iters_train = self.setup_lrn_rate(self.global_step)
        lrn_rate *= FLAGS.uqtf_lrn_rate_dcy

        # decrease the learning rate by a constant factor
        #if self.dataset_name == 'cifar_10':
        #  lrn_rate *= 1e-3
        #elif self.dataset_name == 'ilsvrc_12':
        #  lrn_rate *= 1e-4
        #else:
        #  raise ValueError('unrecognized dataset\'s name: ' + self.dataset_name)

        # obtain the full list of trainable variables & update operations
        self.vars_all = tf.get_collection(
          tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_scope_quan)
        self.trainable_vars_all = tf.get_collection(
          tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope_quan)
        self.update_ops_all = tf.get_collection(
          tf.GraphKeys.UPDATE_OPS, scope=self.model_scope_quan)

        # TF operations for initializing the uniform quantized model
        init_ops = []
        with tf.control_dependencies([tf.variables_initializer(self.vars_all)]):
          for var_full, var_quan in zip(self.vars_full['all'], self.vars_quan['all']):
            init_ops += [var_quan.assign(var_full)]
        init_ops += [self.global_step.initializer]
        self.init_op = tf.group(init_ops)

        # TF operations for fine-tuning
        #optimizer_base = tf.train.MomentumOptimizer(lrn_rate, FLAGS.momentum)
        optimizer_base = tf.train.AdamOptimizer(lrn_rate)
        if not FLAGS.enbl_multi_gpu:
          optimizer = optimizer_base
        else:
          optimizer = mgw.DistributedOptimizer(optimizer_base)
        grads = optimizer.compute_gradients(loss_fnl, self.trainable_vars_all)
        with tf.control_dependencies(self.update_ops_all):
          self.train_op = optimizer.apply_gradients(grads, global_step=self.global_step)
        self.init_opt_op = tf.variables_initializer(optimizer_base.variables())

      # TF operations for logging & summarizing
      self.sess_train = sess
      self.summary_op = tf.summary.merge_all()
      self.log_op = [lrn_rate, loss_fnl] + list(metrics.values())
      self.log_op_names = ['lr', 'loss'] + list(metrics.keys())
      if FLAGS.enbl_multi_gpu:
        self.bcast_op = mgw.broadcast_global_variables(0)

  def __build_eval(self):
    """Build the evaluation graph."""

    with tf.Graph().as_default() as graph:
      # create a TF session for the current graph
      config = tf.ConfigProto()
      config.gpu_options.visible_device_list = str(mgw.local_rank() if FLAGS.enbl_multi_gpu else 0)  # pylint: disable=no-member
      config.gpu_options.allow_growth = True  # pylint: disable=no-member
      self.sess_eval = tf.Session(config=config)

      # data input pipeline
      with tf.variable_scope(self.data_scope):
        iterator = self.build_dataset_eval()
        images, labels = iterator.get_next()

      # model definition - uniform quantized model - part 1
      with tf.variable_scope(self.model_scope_quan):
        logits = self.forward_eval(images)
        if not isinstance(logits, dict):
          outputs = tf.nn.softmax(logits)
        else:
          outputs = tf.nn.softmax(logits['cls_pred'])
        tf.contrib.quantize.experimental_create_eval_graph(
          weight_bits=FLAGS.uqtf_weight_bits,
          activation_bits=FLAGS.uqtf_activation_bits,
          scope=self.model_scope_quan)
        for node_name in self.unquant_node_names:
          insert_quant_op(graph, node_name, is_train=False)
        vars_quan = get_vars_by_scope(self.model_scope_quan)
        global_step = tf.train.get_or_create_global_step()
        self.saver_quan_eval = tf.train.Saver(vars_quan['all'] + [global_step])

      # model definition - distilled model
      if FLAGS.enbl_dst:
        logits_dst = self.helper_dst.calc_logits(self.sess_eval, images)

      # model definition - uniform quantized model -part 2
      with tf.variable_scope(self.model_scope_quan):
        # loss & extra evaluation metrics
        loss, metrics = self.calc_loss(labels, logits, vars_quan['trainable'])
        if FLAGS.enbl_dst:
          loss += self.helper_dst.calc_loss(logits, logits_dst)

        # TF operations for evaluation
        self.eval_op = [loss] + list(metrics.values())
        self.eval_op_names = ['loss'] + list(metrics.keys())
        self.outputs_eval = logits

      # add input & output tensors to certain collections
      if not isinstance(images, dict):
        tf.add_to_collection('images_final', images)
      else:
        tf.add_to_collection('images_final', images['image'])
      if not isinstance(logits, dict):
        tf.add_to_collection('logits_final', logits)
      else:
        tf.add_to_collection('logits_final', logits['cls_pred'])

  def __save_model(self, is_train):
    """Save the current model for training or evaluation.

    Args:
    * is_train: whether to save a model for training
    """

    if is_train:
      save_path = self.saver_quan_train.save(
        self.sess_train, FLAGS.uqtf_save_path, self.global_step)
    else:
      save_path = self.saver_quan_eval.save(self.sess_eval, FLAGS.uqtf_save_path_eval)
    tf.logging.info('model saved to ' + save_path)

  def __restore_model(self, is_train):
    """Restore a model from the latest checkpoint files.

    Args:
    * is_train: whether to restore a model for training
    """

    save_path = tf.train.latest_checkpoint(os.path.dirname(FLAGS.uqtf_save_path))
    if is_train:
      self.saver_quan_train.restore(self.sess_train, save_path)
    else:
      self.saver_quan_eval.restore(self.sess_eval, save_path)
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
