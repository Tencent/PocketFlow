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
"""Full-precision learner (no model compression applied)."""

import os
from timeit import default_timer as timer
import numpy as np
import tensorflow as tf

from learners.abstract_learner import AbstractLearner
from learners.distillation_helper import DistillationHelper
from utils.multi_gpu_wrapper import MultiGpuWrapper as mgw

FLAGS = tf.app.flags.FLAGS

class FullPrecLearner(AbstractLearner):  # pylint: disable=too-many-instance-attributes
  """Full-precision learner (no model compression applied)."""

  def __init__(self, sm_writer, model_helper, model_scope=None, enbl_dst=None):
    """Constructor function.

    Args:
    * sm_writer: TensorFlow's summary writer
    * model_helper: model helper with definitions of model & dataset
    * model_scope: name scope in which to define the model
    * enbl_dst: whether to create a model with distillation loss
    """

    # class-independent initialization
    super(FullPrecLearner, self).__init__(sm_writer, model_helper)

    # over-ride the model scope and distillation loss switch
    if model_scope is not None:
      self.model_scope = model_scope
    self.enbl_dst = enbl_dst if enbl_dst is not None else FLAGS.enbl_dst

    # class-dependent initialization
    if self.enbl_dst:
      self.helper_dst = DistillationHelper(sm_writer, model_helper, self.mpi_comm)
    self.__build(is_train=True)
    self.__build(is_train=False)

  def train(self):
    """Train a model and periodically produce checkpoint files."""

    # initialization
    self.sess_train.run(self.init_op)
    self.warm_start(self.sess_train)
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

      # save & evaluate the model at certain steps
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
    self.dump_n_eval(outputs=None, action='init')
    for idx_iter in range(nb_iters):
      eval_rslts[idx_iter], outputs = self.sess_eval.run([self.eval_op, self.outputs_eval])
      self.dump_n_eval(outputs=outputs, action='dump')
    self.dump_n_eval(outputs=None, action='eval')
    for idx, name in enumerate(self.eval_op_names):
      tf.logging.info('%s = %.4e' % (name, np.mean(eval_rslts[:, idx])))

  def __build(self, is_train):  # pylint: disable=too-many-locals
    """Build the training / evaluation graph.

    Args:
    * is_train: whether to create the training graph
    """

    with tf.Graph().as_default():
      # TensorFlow session
      config = tf.ConfigProto()
      config.gpu_options.visible_device_list = str(mgw.local_rank() if FLAGS.enbl_multi_gpu else 0)  # pylint: disable=no-member
      sess = tf.Session(config=config)

      # data input pipeline
      with tf.variable_scope(self.data_scope):
        iterator = self.build_dataset_train() if is_train else self.build_dataset_eval()
        images, labels = iterator.get_next()
        if not isinstance(images, dict):
          tf.add_to_collection('images_final', images)
        else:
          tf.add_to_collection('images_final', images['image'])

      # model definition - distilled model
      if self.enbl_dst:
        logits_dst = self.helper_dst.calc_logits(sess, images)

      # model definition - primary model
      with tf.variable_scope(self.model_scope):
        # forward pass
        if is_train and self.forward_w_labels:
          logits = self.forward_train(images, labels)
        else:
          logits = self.forward_train(images) if is_train else self.forward_eval(images)
        if not isinstance(logits, dict):
          tf.add_to_collection('logits_final', logits)
        else:
          for value in logits.values():
            tf.add_to_collection('logits_final', value)

        # loss & extra evalution metrics
        loss, metrics = self.calc_loss(labels, logits, self.trainable_vars)
        if self.enbl_dst:
          loss += self.helper_dst.calc_loss(logits, logits_dst)
        tf.summary.scalar('loss', loss)
        for key, value in metrics.items():
          tf.summary.scalar(key, value)

        # optimizer & gradients
        if is_train:
          self.global_step = tf.train.get_or_create_global_step()
          lrn_rate, self.nb_iters_train = self.setup_lrn_rate(self.global_step)
          optimizer = tf.train.MomentumOptimizer(lrn_rate, FLAGS.momentum)
          if FLAGS.enbl_multi_gpu:
            optimizer = mgw.DistributedOptimizer(optimizer)
          grads = optimizer.compute_gradients(loss, self.trainable_vars)

      # TF operations & model saver
      if is_train:
        self.sess_train = sess
        with tf.control_dependencies(self.update_ops):
          self.train_op = optimizer.apply_gradients(grads, global_step=self.global_step)
        self.summary_op = tf.summary.merge_all()
        self.log_op = [lrn_rate, loss] + list(metrics.values())
        self.log_op_names = ['lr', 'loss'] + list(metrics.keys())
        self.init_op = tf.variables_initializer(self.vars)
        if FLAGS.enbl_multi_gpu:
          self.bcast_op = mgw.broadcast_global_variables(0)
        self.saver_train = tf.train.Saver(self.vars)
      else:
        self.sess_eval = sess
        self.eval_op = [loss] + list(metrics.values())
        self.eval_op_names = ['loss'] + list(metrics.keys())
        self.outputs_eval = logits
        self.saver_eval = tf.train.Saver(self.vars)

  def __save_model(self, is_train):
    """Save the model to checkpoint files for training or evaluation.

    Args:
    * is_train: whether to save a model for training
    """

    if is_train:
      save_path = self.saver_train.save(self.sess_train, FLAGS.save_path, self.global_step)
    else:
      save_path = self.saver_eval.save(self.sess_eval, FLAGS.save_path_eval)
    tf.logging.info('model saved to ' + save_path)

  def __restore_model(self, is_train):
    """Restore a model from the latest checkpoint files.

    Args:
    * is_train: whether to restore a model for training
    """

    save_path = tf.train.latest_checkpoint(os.path.dirname(FLAGS.save_path))
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
