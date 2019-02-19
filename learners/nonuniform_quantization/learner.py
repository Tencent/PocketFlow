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
"""
  Non-uniform Quantization Learner.
  Activations are not quantized by default.
"""

import os
from timeit import default_timer as timer
import numpy as np
import tensorflow as tf
from utils.multi_gpu_wrapper import MultiGpuWrapper as mgw
from learners.abstract_learner import AbstractLearner
from learners.distillation_helper import DistillationHelper
from learners.nonuniform_quantization.utils import NonUniformQuantization
from learners.nonuniform_quantization.bit_optimizer import BitOptimizer

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('nuql_init_style', 'quantile', \
    'Initialization: quantile(default), uniform]')
tf.app.flags.DEFINE_string('nuql_opt_mode', 'weights', \
    'Optimize: weights(default), clusters, both')
tf.app.flags.DEFINE_integer('nuql_weight_bits', 4, \
    'Number of bits to use for quantizing weights')
tf.app.flags.DEFINE_integer('nuql_activation_bits', 32, \
    'WARNING: Useless for activation quantization in non-uniform mode')
tf.app.flags.DEFINE_boolean('nuql_use_buckets', False, 'Use bucketing or not')
tf.app.flags.DEFINE_integer('nuql_bucket_size', 256, 'Number of bucket size')
tf.app.flags.DEFINE_integer('nuql_quant_epochs', 60, 'Number of finetune steps for quantization')
tf.app.flags.DEFINE_string('nuql_save_quant_model_path', \
    './nuql_quant_models/model.ckpt', 'dir to save quantization model')
tf.app.flags.DEFINE_boolean('nuql_quantize_all_layers', False, \
    'True for quantizing all layers and Flase for leaving first and last layers unquantized')
tf.app.flags.DEFINE_string('nuql_bucket_type', 'split', '[split, channel]')


def setup_bnds_decay_rates(model_name, dataset_name):
  """ NOTE: The bnd_decay_rates here is mgw_size invariant """
  batch_size = FLAGS.batch_size if not FLAGS.enbl_multi_gpu else FLAGS.batch_size * mgw.size()
  nb_batches_per_epoch = int(FLAGS.nb_smpls_train / batch_size)
  tf.logging.info('nb_batches_per_epoch:%d' % nb_batches_per_epoch)
  mgw_size = int(mgw.size()) if FLAGS.enbl_multi_gpu else 1
  init_lr = FLAGS.lrn_rate_init * FLAGS.batch_size * mgw_size / FLAGS.batch_size_norm \
      if FLAGS.enbl_multi_gpu else FLAGS.lrn_rate_init
  if dataset_name == 'cifar_10':
    if model_name.startswith('resnet'):
      bnds = [nb_batches_per_epoch * 40, nb_batches_per_epoch * 80]
      decay_rates = [1e-4, 1e-5, 1e-6]
  elif dataset_name == 'ilsvrc_12':
    if model_name.startswith('resnet'):
      bnds = [nb_batches_per_epoch * 5, nb_batches_per_epoch * 20]
      decay_rates = [5e-4, 5e-5, 5e-6]
    elif model_name.startswith('mobilenet'):
      bnds = [nb_batches_per_epoch * 5, nb_batches_per_epoch * 30]
      decay_rates = [1e-4, 1e-5, 1e-6]
  finetune_steps = nb_batches_per_epoch * FLAGS.nuql_quant_epochs
  init_lr = init_lr if FLAGS.enbl_warm_start else FLAGS.lrn_rate_init
  return init_lr, bnds, decay_rates, finetune_steps


class NonUniformQuantLearner(AbstractLearner):
  # pylint: disable=too-many-instance-attributes
  '''
  Nonuniform quantization for weights and activations
  '''

  def __init__(self, sm_writer, model_helper):
    # class-independent initialization
    super(NonUniformQuantLearner, self).__init__(sm_writer, model_helper)

    # class-dependent initialization
    if FLAGS.enbl_dst:
      self.helper_dst = DistillationHelper(sm_writer, model_helper, self.mpi_comm)

    # initialize class attributes
    self.ops = {}
    self.bit_placeholders = {}
    self.statistics = {}

    self.__build_train()  # for train
    self.__build_eval() # for eval

    if self.is_primary_worker('local'):
      self.download_model()
    self.auto_barrier()

    # determine the optimal policy.
    bit_optimizer = BitOptimizer(self.dataset_name,
                                 self.weights,
                                 self.statistics,
                                 self.bit_placeholders,
                                 self.ops,
                                 self.layerwise_tune_list,
                                 self.sess_train,
                                 self.sess_eval,
                                 self.saver_train,
                                 self.saver_quant,
                                 self.saver_eval,
                                 self.auto_barrier)
    self.optimal_w_bit_list, self.optimal_a_bit_list = bit_optimizer.run()
    self.auto_barrier()

  def train(self):
    # initialization
    self.sess_train.run(self.ops['non_cluster_init'])
    # mgw_size = int(mgw.size()) if FLAGS.enbl_multi_gpu else 1

    total_iters = self.finetune_steps
    if FLAGS.enbl_warm_start:
      self.__restore_model(is_train=True) # use the latest model for warm start

    # NOTE: initialize the clusters after restore weights
    self.sess_train.run(self.ops['cluster_init'], \
        feed_dict={self.bit_placeholders['w_train']: self.optimal_w_bit_list})

    feed_dict = {self.bit_placeholders['w_train']: self.optimal_w_bit_list, \
        self.bit_placeholders['a_train']: self.optimal_a_bit_list}

    if FLAGS.enbl_multi_gpu:
      self.sess_train.run(self.ops['bcast'])

    time_prev = timer()

    for idx_iter in range(total_iters):
      # train the model
      if (idx_iter + 1) % FLAGS.summ_step != 0:
        self.sess_train.run(self.ops['train'], feed_dict=feed_dict)
      else:
        _, summary, log_rslt = self.sess_train.run([self.ops['train'], \
            self.ops['summary'], self.ops['log']], feed_dict=feed_dict)
        time_prev = self.__monitor_progress(summary, log_rslt, time_prev, idx_iter)

      # save & evaluate the model at certain steps
      if (idx_iter + 1) % FLAGS.save_step == 0:
        self.__save_model()
        self.evaluate()
        tf.logging.info("Optimal Weight Quantization:{}".format(self.optimal_w_bit_list))
        self.auto_barrier()

    # save the final model
    self.__save_model()
    self.evaluate()

  def evaluate(self):
    # early break for non-primary workers
    if not self.is_primary_worker():
      return

    # evaluate the model
    self.__restore_model(is_train=False)
    losses, accuracies = [], []
    nb_iters = int(np.ceil(float(FLAGS.nb_smpls_eval) / FLAGS.batch_size_eval))

    # build the quantization bits
    feed_dict = {self.bit_placeholders['w_eval']: self.optimal_w_bit_list, \
        self.bit_placeholders['a_eval']: self.optimal_a_bit_list}

    for _ in range(nb_iters):
      eval_rslt = self.sess_eval.run(self.ops['eval'], feed_dict=feed_dict)
      losses.append(eval_rslt[0])
      accuracies.append(eval_rslt[1])
    tf.logging.info('loss: {}'.format(np.mean(np.array(losses))))
    tf.logging.info('accuracy: {}'.format(np.mean(np.array(accuracies))))
    tf.logging.info("Optimal Weight Quantization:{}".format(self.optimal_w_bit_list))

    if FLAGS.nuql_use_buckets:
      bucket_storage = self.sess_eval.run(self.ops['bucket_storage'], feed_dict=feed_dict)
      self.__show_bucket_storage(bucket_storage)

  def __build_train(self):
    with tf.Graph().as_default():
      # TensorFlow session
      config = tf.ConfigProto()
      config.gpu_options.visible_device_list = str(mgw.local_rank() \
          if FLAGS.enbl_multi_gpu else 0)
      self.sess_train = tf.Session(config=config)

      # data input pipeline
      with tf.variable_scope(self.data_scope):
        iterator = self.build_dataset_train()
        images, labels = iterator.get_next()
        images.set_shape((FLAGS.batch_size, images.shape[1], images.shape[2], \
            images.shape[3]))

      # model definition - distilled model
      if FLAGS.enbl_dst:
        logits_dst = self.helper_dst.calc_logits(self.sess_train, images)

      # model definition
      with tf.variable_scope(self.model_scope, reuse=tf.AUTO_REUSE):
        # forward pass
        logits = self.forward_train(images)

        self.saver_train = tf.train.Saver(self.vars)

        self.weights = [v for v in self.trainable_vars if 'kernel' in v.name or 'weight' in v.name]
        if not FLAGS.nuql_quantize_all_layers:
          self.weights = self.weights[1:-1]
        self.statistics['num_weights'] = \
            [tf.reshape(v, [-1]).shape[0].value for v in self.weights]

        self.__quantize_train_graph()

        # loss & accuracy
        # Strictly speaking, clusters should be not included for regularization.
        loss, metrics = self.calc_loss(labels, logits, self.trainable_vars)
        if self.dataset_name == 'cifar_10':
          acc_top1, acc_top5 = metrics['accuracy'], tf.constant(0.)
        elif self.dataset_name == 'ilsvrc_12':
          acc_top1, acc_top5 = metrics['acc_top1'], metrics['acc_top5']
        else:
          raise ValueError("Unrecognized dataset name")

        model_loss = loss
        if FLAGS.enbl_dst:
          dst_loss = self.helper_dst.calc_loss(logits, logits_dst)
          loss += dst_loss
          tf.summary.scalar('dst_loss', dst_loss)
        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('acc_top1', acc_top1)
        tf.summary.scalar('acc_top5', acc_top5)

        self.ft_step = tf.get_variable('finetune_step', shape=[], dtype=tf.int32, trainable=False)

      # optimizer & gradients
      init_lr, bnds, decay_rates, self.finetune_steps = \
          setup_bnds_decay_rates(self.model_name, self.dataset_name)
      lrn_rate = tf.train.piecewise_constant(self.ft_step, [i for i in bnds], \
          [init_lr * decay_rate for decay_rate in decay_rates])

      # optimizer = tf.train.MomentumOptimizer(lrn_rate, FLAGS.momentum)
      optimizer = tf.train.AdamOptimizer(lrn_rate)
      if FLAGS.enbl_multi_gpu:
        optimizer = mgw.DistributedOptimizer(optimizer)

      # acquire non-cluster-vars and clusters.
      clusters = [v for v in self.trainable_vars if 'clusters' in v.name]
      rest_trainable_vars = [v for v in self.trainable_vars \
          if v not in clusters]

      # determine the var_list optimize
      if FLAGS.nuql_opt_mode in ['cluster', 'both']:
        if FLAGS.nuql_opt_mode == 'both':
          optimizable_vars = self.trainable_vars
        else:
          optimizable_vars = clusters
        if FLAGS.nuql_enbl_rl_agent:
          optimizer_fintune = tf.train.GradientDescentOptimizer(lrn_rate)
          if FLAGS.enbl_multi_gpu:
            optimizer_fintune = mgw.DistributedOptimizer(optimizer_fintune)
          grads_fintune = optimizer_fintune.compute_gradients(loss, var_list=optimizable_vars)

      elif FLAGS.nuql_opt_mode == 'weights':
        optimizable_vars = rest_trainable_vars
      else:
        raise ValueError("Unknown optimization mode")

      grads = optimizer.compute_gradients(loss, var_list=optimizable_vars)

      # sm write graph
      self.sm_writer.add_graph(self.sess_train.graph)

      # define the ops
      with tf.control_dependencies(self.update_ops):
        self.ops['train'] = optimizer.apply_gradients(grads, global_step=self.ft_step)
        if FLAGS.nuql_opt_mode in ['both', 'cluster'] and FLAGS.nuql_enbl_rl_agent:
          self.ops['rl_fintune'] = optimizer_fintune.apply_gradients(grads_fintune, global_step=self.ft_step)
        else:
          self.ops['rl_fintune'] = self.ops['train']
      self.ops['summary'] = tf.summary.merge_all()
      if FLAGS.enbl_dst:
        self.ops['log'] = [lrn_rate, dst_loss, model_loss, loss, acc_top1, acc_top5]
      else:
        self.ops['log'] = [lrn_rate, model_loss, loss, acc_top1, acc_top5]

      # NOTE: first run non_cluster_init_op, then cluster_init_op
      cluster_global_vars = [v for v in self.vars if 'clusters' in v.name]
      # pay attention to beta1_power, beta2_power.
      non_cluster_global_vars = [v for v in tf.global_variables() if v not in cluster_global_vars]

      self.ops['non_cluster_init'] = tf.variables_initializer(var_list=non_cluster_global_vars)
      self.ops['cluster_init'] = tf.variables_initializer(var_list=cluster_global_vars)
      self.ops['bcast'] = mgw.broadcast_global_variables(0) \
          if FLAGS.enbl_multi_gpu else None
      self.ops['reset_ft_step'] = tf.assign(self.ft_step, \
          tf.constant(0, dtype=tf.int32))

      self.saver_quant = tf.train.Saver(self.vars)

  def __build_eval(self):
    with tf.Graph().as_default():
      # TensorFlow session
      config = tf.ConfigProto()
      config.gpu_options.visible_device_list = str(mgw.local_rank() \
          if FLAGS.enbl_multi_gpu else 0)
      self.sess_eval = tf.Session(config=config)

      # data input pipeline
      with tf.variable_scope(self.data_scope):
        iterator = self.build_dataset_eval()
        images, labels = iterator.get_next()
        images.set_shape((FLAGS.batch_size, images.shape[1], images.shape[2], \
            images.shape[3]))
        self.images_eval = images

      # model definition - distilled model
      if FLAGS.enbl_dst:
        logits_dst = self.helper_dst.calc_logits(self.sess_eval, images)

      # model definition
      with tf.variable_scope(self.model_scope, reuse=tf.AUTO_REUSE):
        # forward pass
        logits = self.forward_eval(images)

        self.__quantize_eval_graph()

        # loss & accuracy
        loss, metrics = self.calc_loss(labels, logits, self.trainable_vars)
        if self.dataset_name == 'cifar_10':
          acc_top1, acc_top5 = metrics['accuracy'], tf.constant(0.)
        elif self.dataset_name == 'ilsvrc_12':
          acc_top1, acc_top5 = metrics['acc_top1'], metrics['acc_top5']
        else:
          raise ValueError("Unrecognized dataset name")

        if FLAGS.enbl_dst:
          dst_loss = self.helper_dst.calc_loss(logits, logits_dst)
          loss += dst_loss

      self.ops['eval'] = [loss, acc_top1, acc_top5]
      self.saver_eval = tf.train.Saver(self.vars)

  def __quantize_train_graph(self):
    """ Insert Quantization nodes to the training graph """
    nonuni_quant = NonUniformQuantization(self.sess_train,
                                          FLAGS.nuql_bucket_size,
                                          FLAGS.nuql_use_buckets,
                                          FLAGS.nuql_init_style,
                                          FLAGS.nuql_bucket_type)

    # Find Matmul Op and activation Op
    matmul_ops = nonuni_quant.search_matmul_op(FLAGS.nuql_quantize_all_layers)
    act_ops = nonuni_quant.search_activation_op()

    self.statistics['nb_matmuls'] = len(matmul_ops)
    self.statistics['nb_activations'] = len(act_ops)

    # Replace Conv2d Op with quantized weights
    matmul_op_names = [op.name for op in matmul_ops]
    act_op_names = [op.name for op in act_ops]

    # build the placeholder for nonuniform quantization bits.
    self.bit_placeholders['w_train'] = tf.placeholder(tf.int64, \
        shape=[self.statistics['nb_matmuls']], name="w_bit_list")
    self.bit_placeholders['a_train'] = tf.placeholder(tf.int64, \
        shape=[self.statistics['nb_activations']], name="a_bit_list")
    w_bit_dict_train = self.__build_quant_dict(matmul_op_names, \
        self.bit_placeholders['w_train'])
    a_bit_dict_train = self.__build_quant_dict(act_op_names, \
        self.bit_placeholders['a_train'])

    # Insert Quant Op for weights and activations
    nonuni_quant.insert_quant_op_for_weights(w_bit_dict_train)
    # NOTE: Not necessary for activation quantization in non-uniform
    nonuni_quant.insert_quant_op_for_activations(a_bit_dict_train)

    # TODO: add layerwise finetuning. working not very weill
    self.layerwise_tune_list = nonuni_quant.get_layerwise_tune_op(self.weights) \
        if FLAGS.nuql_enbl_rl_layerwise_tune else (None, None)

  def __quantize_eval_graph(self):
    """ Insert Quantization nodes to the eval graph """
    nonuni_quant = NonUniformQuantization(self.sess_eval,
                                          FLAGS.nuql_bucket_size,
                                          FLAGS.nuql_use_buckets,
                                          FLAGS.nuql_init_style,
                                          FLAGS.nuql_bucket_type)

    # Find Matmul Op and activation Op
    matmul_ops = nonuni_quant.search_matmul_op(FLAGS.nuql_quantize_all_layers)
    act_ops = nonuni_quant.search_activation_op()
    assert self.statistics['nb_matmuls'] == len(matmul_ops), \
        'the length of matmul_ops does not match'
    assert self.statistics['nb_activations'] == len(act_ops), \
        'the length of act_ops does not match'

    # Replace Conv2d Op with quantized weights
    matmul_op_names = [op.name for op in matmul_ops]
    act_op_names = [op.name for op in act_ops]

    # build the placeholder for eval
    self.bit_placeholders['w_eval'] = tf.placeholder(tf.int64, \
        shape=[self.statistics['nb_matmuls']], name="w_bit_list")
    self.bit_placeholders['a_eval'] = tf.placeholder(tf.int64, \
        shape=[self.statistics['nb_activations']], name="a_bit_list")

    w_bit_dict_eval = self.__build_quant_dict(matmul_op_names, \
        self.bit_placeholders['w_eval'])
    a_bit_dict_eval = self.__build_quant_dict(act_op_names, \
        self.bit_placeholders['a_eval'])

    # Insert Quant Op for weights and activations
    nonuni_quant.insert_quant_op_for_weights(w_bit_dict_eval)
    # NOTE: no need for activation quantization
    nonuni_quant.insert_quant_op_for_activations(a_bit_dict_eval)

    self.ops['bucket_storage'] = nonuni_quant.bucket_storage if FLAGS.nuql_use_buckets \
        else tf.constant(0, tf.int32)

  def __save_model(self):
    # early break for non-primary workers
    if not self.is_primary_worker():
      return
    # save quantization model
    save_quant_model_path = self.saver_quant.save(self.sess_train, \
        FLAGS.nuql_save_quant_model_path, self.ft_step)
    tf.logging.info('quantized model saved to ' + save_quant_model_path)

  def __restore_model(self, is_train):
    if is_train:
      save_path = tf.train.latest_checkpoint(os.path.dirname(FLAGS.save_path))
      self.saver_train.restore(self.sess_train, save_path)
    else:
      save_path = tf.train.latest_checkpoint(os.path.dirname(FLAGS.nuql_save_quant_model_path))
      self.saver_eval.restore(self.sess_eval, save_path)
    tf.logging.info('model restored from ' + save_path)

  def __monitor_progress(self, summary, log_rslt, time_prev, idx_iter):
    # early break for non-primary workers
    if not self.is_primary_worker():
      return None

    # write summaries for TensorBoard visualization
    self.sm_writer.add_summary(summary, idx_iter)

    # display monitored statistics
    speed = FLAGS.batch_size * FLAGS.summ_step / (timer() - time_prev)
    if FLAGS.enbl_multi_gpu:
      speed *= mgw.size()

    if FLAGS.enbl_dst:
      lrn_rate, dst_loss, model_loss, loss, acc_top1, acc_top5 = log_rslt[0], \
      log_rslt[1], log_rslt[2], log_rslt[3], log_rslt[4], log_rslt[5]
      tf.logging.info('iter #%d: lr = %e | dst_loss = %.4f | model_loss = %.4f | loss = %.4f | acc_top1 = %.4f | acc_top5 = %.4f | speed = %.2f pics / sec' \
          % (idx_iter + 1, lrn_rate, dst_loss, model_loss, loss, acc_top1, acc_top5, speed))
    else:
      lrn_rate, model_loss, loss, acc_top1, acc_top5 = log_rslt[0], \
      log_rslt[1], log_rslt[2], log_rslt[3], log_rslt[4]
      tf.logging.info('iter #%d: lr = %e | model_loss = %.4f | loss = %.4f | acc_top1 = %.4f | acc_top5 = %.4f | speed = %.2f pics / sec' \
          % (idx_iter + 1, lrn_rate, model_loss, loss, acc_top1, acc_top5, speed))

    return timer()

  def __show_bucket_storage(self, bucket_storage):
    weight_storage = sum(self.statistics['num_weights']) * FLAGS.nuql_weight_bits \
        if not FLAGS.nuql_enbl_rl_agent \
        else sum(self.statistics['num_weights']) * FLAGS.nuql_equivalent_bits
    tf.logging.info('bucket storage: %d bit / %.3f kb | weight storage: %d bit / %.3f kb | ratio: %.3f' % \
        (bucket_storage, bucket_storage/(8.*1024.), weight_storage, \
            weight_storage/(8.*1024.), bucket_storage*1./weight_storage))

  @staticmethod
  def __build_quant_dict(keys, values):
    """ Bind keys and values to dictionaries.

    Args:
    * keys: A list of op_names;
    * values: A Tensor with len(keys) elements;

    Returns:
    * dict: (key, value) for weight name and quant bits respectively.
    """

    dict_ = {}
    for (idx, v) in enumerate(keys):
      dict_[v] = values[idx]
    return dict_

