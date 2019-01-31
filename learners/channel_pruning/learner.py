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
"""Channel Pruned Learner"""
import os
import math
import pathlib
import string
import random
from collections import deque
from timeit import default_timer as timer
import numpy as np
import tensorflow as tf
from tensorflow.contrib import graph_editor

from utils.multi_gpu_wrapper import MultiGpuWrapper as mgw
from learners.distillation_helper import DistillationHelper
from learners.abstract_learner import AbstractLearner
from learners.channel_pruning.model_wrapper import Model
from learners.channel_pruning.channel_pruner import ChannelPruner
from rl_agents.ddpg.agent import Agent as DdpgAgent

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
  'cp_prune_option',
  'auto',
  """the action we want to prune the channel you can select one of the following option:
     uniform:
        prune with a uniform compression ratio
     list:
        prune with a list of compression ratio""")

tf.app.flags.DEFINE_string(
  'cp_prune_list_file',
  'ratio.list',
  'the prune list file which contains the compression ratio of each convolution layers')
tf.app.flags.DEFINE_string(
  'cp_channel_pruned_path',
  './models/pruned_model.ckpt',
  'channel pruned model\'s save path')
tf.app.flags.DEFINE_string(
  'cp_best_path',
  './models/best_model.ckpt',
  'channel pruned model\'s temporary save path')
tf.app.flags.DEFINE_string(
  'cp_original_path',
  './models/original_model.ckpt',
  'channel pruned model\'s temporary save path')
tf.app.flags.DEFINE_float(
  'cp_preserve_ratio',
  0.5, 'How much computation cost desired to be preserved after pruning')
tf.app.flags.DEFINE_float(
  'cp_uniform_preserve_ratio',
  0.6, 'How much computation cost desired to be preserved each layer')
tf.app.flags.DEFINE_float(
  'cp_noise_tolerance',
  0.15,
  'the noise tolerance which is used to restrict the maximum reward to avoid an unexpected speedup')
tf.app.flags.DEFINE_float('cp_lrn_rate_ft', 1e-4, 'CP: learning rate for global fine-tuning')
tf.app.flags.DEFINE_float('cp_nb_iters_ft_ratio', 0.2,
                          'CP: the ratio of total iterations for global fine-tuning')
tf.app.flags.DEFINE_boolean('cp_finetune', False, 'CP: whether finetuning between each list group')
tf.app.flags.DEFINE_boolean('cp_retrain', False, 'CP: whether retraining between each list group')
tf.app.flags.DEFINE_integer('cp_list_group', 1000, 'CP: # of iterations for fast evaluation')
tf.app.flags.DEFINE_integer('cp_nb_rlouts', 200, 'CP: # of roll-outs for the RL agent')
tf.app.flags.DEFINE_integer('cp_nb_rlouts_min', 50, 'CP: # of roll-outs for the RL agent')

class ChannelPrunedLearner(AbstractLearner):  # pylint: disable=too-many-instance-attributes
  """Learner with channel/filter pruning"""

  def __init__(self, sm_writer, model_helper):
    # class-independent initialization
    super(ChannelPrunedLearner, self).__init__(sm_writer, model_helper)

    # class-dependent initialization
    if FLAGS.enbl_dst:
      self.learner_dst = DistillationHelper(sm_writer, model_helper, self.mpi_comm)

    self.model_scope = 'model'

    self.sm_writer = sm_writer
    #self.max_eval_acc = 0
    self.max_save_path = ''
    self.saver = None
    self.saver_train = None
    self.saver_eval = None
    self.model = None
    self.pruner = None
    self.sess_train = None
    self.sess_eval = None
    self.log_op = None
    self.train_op = None
    self.bcast_op = None
    self.train_init_op = None
    self.time_prev = None
    self.agent = None
    self.idx_iter = None
    self.accuracy_keys = None
    self.eval_op = None
    self.global_step = None
    self.summary_op = None
    self.nb_iters_train = 0
    self.bestinfo = None

    self.__build(is_train=True)
    self.__build(is_train=False)

  def train(self):
    """Train the pruned model"""
    # download pre-trained model
    if self.__is_primary_worker():
      self.download_model()
      self.__restore_model(True)
      self.saver_train.save(self.sess_train, FLAGS.cp_original_path)
      self.create_pruner()

    if FLAGS.enbl_multi_gpu:
      self.mpi_comm.Barrier()

    tf.logging.info('Start pruning')

    # channel pruning and finetuning
    if FLAGS.cp_prune_option == 'list':
      self.__prune_and_finetune_list()
    elif FLAGS.cp_prune_option == 'auto':
      self.__prune_and_finetune_auto()
    elif FLAGS.cp_prune_option == 'uniform':
      self.__prune_and_finetune_uniform()

  def create_pruner(self):
    """create a pruner"""
    with tf.Graph().as_default():
      config = tf.ConfigProto()
      config.gpu_options.visible_device_list = str(0) # pylint: disable=no-member
      sess = tf.Session(config=config)
      self.saver = tf.train.import_meta_graph(FLAGS.cp_original_path + '.meta')
      self.saver.restore(sess, FLAGS.cp_original_path)
      self.sess_train = sess
      self.sm_writer.add_graph(sess.graph)
      train_images = tf.get_collection('train_images')[0]
      train_labels = tf.get_collection('train_labels')[0]
      mem_images = tf.get_collection('mem_images')[0]
      mem_labels = tf.get_collection('mem_labels')[0]
      summary_op = tf.get_collection('summary_op')[0]
      loss = tf.get_collection('loss')[0]

      accuracy = tf.get_collection('accuracy')[0]
      #accuracy1 = tf.get_collection('top1')[0]
      #metrics = {'loss': loss, 'accuracy': accuracy['top1']}
      metrics = {'loss': loss, 'accuracy': accuracy}
      for key in self.accuracy_keys:
        metrics[key] = tf.get_collection(key)[0]
      self.model = Model(self.sess_train)
      pruner = ChannelPruner(
        self.model,
        images=train_images,
        labels=train_labels,
        mem_images=mem_images,
        mem_labels=mem_labels,
        metrics=metrics,
        lbound=self.lbound,
        summary_op=summary_op,
        sm_writer=self.sm_writer)

      self.pruner = pruner

  def evaluate(self):
    """evaluate the model"""
    # early break for non-primary workers
    if not self.__is_primary_worker():
      return

    if self.saver_eval is None:
      self.saver_eval = tf.train.Saver()
    self.__restore_model(is_train=False)
    losses, accuracy = [], []

    nb_iters = FLAGS.nb_smpls_eval // FLAGS.batch_size_eval

    self.sm_writer.add_graph(self.sess_eval.graph)

    accuracies = [[] for i in range(len(self.accuracy_keys))]
    for _ in range(nb_iters):
      eval_rslt = self.sess_eval.run(self.eval_op)
      losses.append(eval_rslt[0])
      for i in range(len(self.accuracy_keys)):
        accuracies[i].append(eval_rslt[i + 1])
    loss = np.mean(np.array(losses))
    tf.logging.info('loss: {}'.format(loss))
    for i in range(len(self.accuracy_keys)):
      accuracy.append(np.mean(np.array(accuracies[i])))
      tf.logging.info('{}: {}'.format(self.accuracy_keys[i], accuracy[i]))

    # save the checkpoint if its evaluatin result is best so far
    #if accuracy[0] > self.max_eval_acc:
    #  self.max_eval_acc = accuracy[0]
    #  self.__save_in_progress_pruned_model()

  def __build(self, is_train): # pylint: disable=too-many-locals
    # early break for non-primary workers
    if not self.__is_primary_worker():
      return

    if not is_train:
      self.__build_pruned_evaluate_model()
      return

    with tf.Graph().as_default():
      # create a TF session for the current graph
      config = tf.ConfigProto()
      config.gpu_options.visible_device_list = str(0) # pylint: disable=no-member
      sess = tf.Session(config=config)

      # data input pipeline
      with tf.variable_scope(self.data_scope):
        train_images, train_labels = self.build_dataset_train().get_next()
        eval_images, eval_labels = self.build_dataset_eval().get_next()
        image_shape = train_images.shape.as_list()
        label_shape = train_labels.shape.as_list()
        image_shape[0] = FLAGS.batch_size
        label_shape[0] = FLAGS.batch_size

        mem_images = tf.placeholder(dtype=train_images.dtype,
                                    shape=image_shape)
        mem_labels = tf.placeholder(dtype=train_labels.dtype,
                                    shape=label_shape)

        tf.add_to_collection('train_images', train_images)
        tf.add_to_collection('train_labels', train_labels)
        tf.add_to_collection('eval_images', eval_images)
        tf.add_to_collection('eval_labels', eval_labels)
        tf.add_to_collection('mem_images', mem_images)
        tf.add_to_collection('mem_labels', mem_labels)

      # model definition
      with tf.variable_scope(self.model_scope):
        # forward pass
        logits = self.forward_train(mem_images)
        loss, accuracy = self.calc_loss(mem_labels, logits, self.trainable_vars)
        self.accuracy_keys = list(accuracy.keys())
        for key in self.accuracy_keys:
          tf.add_to_collection(key, accuracy[key])
        tf.add_to_collection('loss', loss)
        tf.add_to_collection('logits', logits)

        #self.loss = loss
        tf.summary.scalar('loss', loss)
        for key in accuracy.keys():
          tf.summary.scalar(key, accuracy[key])

      # learning rate & pruning ratio
      self.sess_train = sess
      self.summary_op = tf.summary.merge_all()
      tf.add_to_collection('summary_op', self.summary_op)
      self.saver_train = tf.train.Saver(self.vars)

      self.lbound = math.log(FLAGS.cp_preserve_ratio + 1, 10) * 1.5
      self.rbound = 1.0

  def __build_pruned_evaluate_model(self, path=None):
    ''' build a evaluation model from pruned model '''
    # early break for non-primary workers
    if not self.__is_primary_worker():
      return

    if path is None:
      path = FLAGS.save_path

    if not tf.train.checkpoint_exists(path):
      return

    with tf.Graph().as_default():
      config = tf.ConfigProto()
      config.gpu_options.visible_device_list = str(# pylint: disable=no-member
        mgw.local_rank() if FLAGS.enbl_multi_gpu else 0)
      self.sess_eval = tf.Session(config=config)
      self.saver_eval = tf.train.import_meta_graph(path + '.meta')
      self.saver_eval.restore(self.sess_eval, path)
      eval_logits = tf.get_collection('logits')[0]
      tf.add_to_collection('logits_final', eval_logits)
      eval_images = tf.get_collection('eval_images')[0]
      tf.add_to_collection('images_final', eval_images)
      eval_labels = tf.get_collection('eval_labels')[0]
      mem_images = tf.get_collection('mem_images')[0]
      mem_labels = tf.get_collection('mem_labels')[0]

      self.sess_eval.close()

      graph_editor.reroute_ts(eval_images, mem_images)
      graph_editor.reroute_ts(eval_labels, mem_labels)

      self.sess_eval = tf.Session(config=config)
      self.saver_eval.restore(self.sess_eval, path)
      trainable_vars = self.trainable_vars
      loss, accuracy = self.calc_loss(eval_labels, eval_logits, trainable_vars)
      self.eval_op = [loss] + list(accuracy.values())
      self.sm_writer.add_graph(self.sess_eval.graph)

  def __build_pruned_train_model(self, path=None, finetune=False): # pylint: disable=too-many-locals
    ''' build a training model from pruned model '''
    if path is None:
      path = FLAGS.save_path

    with tf.Graph().as_default():
      config = tf.ConfigProto()
      config.gpu_options.visible_device_list = str(# pylint: disable=no-member
        mgw.local_rank() if FLAGS.enbl_multi_gpu else 0)
      self.sess_train = tf.Session(config=config)
      self.saver_train = tf.train.import_meta_graph(path + '.meta')
      self.saver_train.restore(self.sess_train, path)
      logits = tf.get_collection('logits')[0]
      train_images = tf.get_collection('train_images')[0]
      train_labels = tf.get_collection('train_labels')[0]
      mem_images = tf.get_collection('mem_images')[0]
      mem_labels = tf.get_collection('mem_labels')[0]

      self.sess_train.close()

      graph_editor.reroute_ts(train_images, mem_images)
      graph_editor.reroute_ts(train_labels, mem_labels)

      self.sess_train = tf.Session(config=config)
      self.saver_train.restore(self.sess_train, path)

      trainable_vars = self.trainable_vars
      loss, accuracy = self.calc_loss(train_labels, logits, trainable_vars)
      self.accuracy_keys = list(accuracy.keys())

      if FLAGS.enbl_dst:
        logits_dst = self.learner_dst.calc_logits(self.sess_train, train_images)
        loss += self.learner_dst.calc_loss(logits, logits_dst)

      tf.summary.scalar('loss', loss)
      for key in accuracy.keys():
        tf.summary.scalar(key, accuracy[key])
      self.summary_op = tf.summary.merge_all()

      global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32, trainable=False)
      self.global_step = global_step
      lrn_rate, self.nb_iters_train = self.setup_lrn_rate(self.global_step)

      if finetune and not FLAGS.cp_retrain:
        mom_optimizer = tf.train.AdamOptimizer(FLAGS.cp_lrn_rate_ft)
        self.log_op = [tf.constant(FLAGS.cp_lrn_rate_ft), loss, list(accuracy.values())]
      else:
        mom_optimizer = tf.train.MomentumOptimizer(lrn_rate, FLAGS.momentum)
        self.log_op = [lrn_rate, loss, list(accuracy.values())]

      if FLAGS.enbl_multi_gpu:
        optimizer = mgw.DistributedOptimizer(mom_optimizer)
      else:
        optimizer = mom_optimizer
      grads_origin = optimizer.compute_gradients(loss, trainable_vars)
      grads_pruned, masks = self.__calc_grads_pruned(grads_origin)


      with tf.control_dependencies(self.update_ops):
        self.train_op = optimizer.apply_gradients(grads_pruned, global_step=global_step)

      self.sm_writer.add_graph(tf.get_default_graph())
      self.train_init_op = \
        tf.initialize_variables(mom_optimizer.variables() + [global_step] + masks)

      if FLAGS.enbl_multi_gpu:
        self.bcast_op = mgw.broadcast_global_variables(0)

  def __calc_grads_pruned(self, grads_origin):
    """Calculate the pruned gradients
    Args:
    * grads_origin: the original gradient

    Return:
    * the pruned gradients
    * the corresponding mask of the pruned gradients
    """
    grads_pruned = []
    masks = []
    maskable_var_names = {}
    fake_pruning_dict = {}
    if self.__is_primary_worker():
      fake_pruning_dict = self.pruner.fake_pruning_dict
      maskable_var_names = {
        self.pruner.model.get_var_by_op(
          self.pruner.model.g.get_operation_by_name(op_name)).name: \
            op_name for op_name, ratio in fake_pruning_dict.items()}
      tf.logging.debug('maskable var names {}'.format(maskable_var_names))

    if FLAGS.enbl_multi_gpu:
      fake_pruning_dict = self.mpi_comm.bcast(fake_pruning_dict, root=0)
      maskable_var_names = self.mpi_comm.bcast(maskable_var_names, root=0)

    for grad in grads_origin:
      if grad[1].name not in maskable_var_names.keys():
        grads_pruned.append(grad)
      else:
        pruned_idxs = fake_pruning_dict[maskable_var_names[grad[1].name]]
        mask_tensor = np.ones(grad[0].shape)
        mask_tensor[:, :, [not i for i in pruned_idxs[0]], :] = 0
        mask_tensor[:, :, :, [not i for i in pruned_idxs[1]]] = 0
        mask_initializer = tf.constant_initializer(mask_tensor)
        mask = tf.get_variable(
          grad[1].name.split(':')[0] + '_mask',
          shape=mask_tensor.shape, initializer=mask_initializer, trainable=False)
        masks.append(mask)
        grads_pruned.append((grad[0] * mask, grad[1]))

    return grads_pruned, masks

  def __train_pruned_model(self, finetune=False):
    """Train pruned model"""
    # Initialize varialbes
    self.sess_train.run(self.train_init_op)

    if FLAGS.enbl_multi_gpu:
      self.sess_train.run(self.bcast_op)

    ## Fintuning & distilling
    self.time_prev = timer()

    nb_iters = int(FLAGS.cp_nb_iters_ft_ratio * self.nb_iters_train) \
      if finetune and not FLAGS.cp_retrain else self.nb_iters_train

    for self.idx_iter in range(nb_iters):
      # train the model
      if (self.idx_iter + 1) % FLAGS.summ_step != 0:
        self.sess_train.run(self.train_op)
      else:
        __, summary, log_rslt = self.sess_train.run([self.train_op, self.summary_op, self.log_op])
        self.__monitor_progress(summary, log_rslt)

      # save the model at certain steps
      if (self.idx_iter + 1) % FLAGS.save_step == 0:
        #summary, log_rslt = self.sess_train.run([self.summary_op, self.log_op])
        #self.__monitor_progress(summary, log_rslt)
        if self.__is_primary_worker():
          self.__save_model()
          self.evaluate()

        if FLAGS.enbl_multi_gpu:
          self.mpi_comm.Barrier()

    if self.__is_primary_worker():
      self.__save_model()
      self.evaluate()
      self.__save_in_progress_pruned_model()

    if FLAGS.enbl_multi_gpu:
      self.max_save_path = self.mpi_comm.bcast(self.max_save_path, root=0)
    if self.__is_primary_worker():
      with self.pruner.model.g.as_default():
        #save_path = tf.train.latest_checkpoint(os.path.dirname(FLAGS.channel_pruned_path))
        self.pruner.saver = tf.train.Saver()
        self.pruner.saver.restore(self.pruner.model.sess, self.max_save_path)
        #self.pruner.save_model()

      #self.saver_train.restore(self.sess_train, self.max_save_path)
      #self.__save_model()

  def __save_best_pruned_model(self):
    """ save a in best purned model with a max evaluation result"""
    best_path = tf.train.Saver().save(self.pruner.model.sess, FLAGS.cp_best_path)
    tf.logging.info('model saved best model to ' + best_path)

  def __save_in_progress_pruned_model(self):
    """ save a in progress training model with a max evaluation result"""
    self.max_save_path = self.saver_eval.save(self.sess_eval, FLAGS.cp_best_path)
    tf.logging.info('model saved best model to ' + self.max_save_path)

  def __save_model(self):
    save_path = self.saver_train.save(self.sess_train, FLAGS.save_path, self.global_step)
    tf.logging.info('model saved to ' + save_path)

  def __restore_model(self, is_train):
    save_path = tf.train.latest_checkpoint(os.path.dirname(FLAGS.save_path))
    if is_train:
      self.saver_train.restore(self.sess_train, save_path)
    else:
      self.saver_eval.restore(self.sess_eval, save_path)
    tf.logging.info('model restored from ' + save_path)

  def __monitor_progress(self, summary, log_rslt):
    # early break for non-primary workers
    if not self.__is_primary_worker():
      return
    # write summaries for TensorBoard visualization
    self.sm_writer.add_summary(summary, self.idx_iter)

    # display monitored statistics
    lrn_rate, loss, accuracy = log_rslt[0], log_rslt[1], log_rslt[2]
    speed = FLAGS.batch_size * FLAGS.summ_step / (timer() - self.time_prev)
    if FLAGS.enbl_multi_gpu:
      speed *= mgw.size()
    tf.logging.info('iter #%d: lr = %e | loss = %e | speed = %.2f pics / sec'
                    % (self.idx_iter + 1, lrn_rate, loss, speed))
    for i in range(len(self.accuracy_keys)):
      tf.logging.info('{} = {}'.format(self.accuracy_keys[i], accuracy[i]))
    self.time_prev = timer()

  def __prune_and_finetune_uniform(self):
    '''prune with a list of compression ratio'''
    if self.__is_primary_worker():
      done = False
      self.pruner.extract_features()

      start = timer()
      while not done:
        _, _, done, _ = self.pruner.compress(FLAGS.cp_uniform_preserve_ratio)

      tf.logging.info('uniform channl pruning time cost: {}s'.format(timer() - start))
      self.pruner.save_model()

    if FLAGS.enbl_multi_gpu:
      self.mpi_comm.Barrier()

    self.__finetune_pruned_model(path=FLAGS.cp_channel_pruned_path)

  def __prune_and_finetune_list(self):
    '''prune with a list of compression ratio'''
    try:
      ratio_list = np.loadtxt(FLAGS.cp_prune_list_file, delimiter=',')
      ratio_list = list(ratio_list)
    except IOError as err:
      tf.logging.error('The prune list file format is not correct. \n \
        It\'s content should be a float list delimited by a comma.')
      raise err
    ratio_list.reverse()
    queue = deque(ratio_list)

    done = False
    while not done:
      done = self.__prune_list_layers(queue, [FLAGS.cp_list_group])


  def __prune_list_layers(self, queue, ps=None):
    for p in ps:
      done = self.__prune_n_layers(p, queue)
    return done

  def __prune_n_layers(self, n, queue):
    #self.max_eval_acc = 0
    done = False
    if self.__is_primary_worker():
      self.pruner.extract_features()
      done = False
      i = 0
      while not done and i < n:
        if not queue:
          ratio = 1
        else:
          ratio = queue.pop()
        _, _, done, _ = self.pruner.compress(ratio)
        i += 1

      self.pruner.save_model()

    if FLAGS.enbl_multi_gpu:
      self.mpi_comm.Barrier()
      done = self.mpi_comm.bcast(done, root=0)

    if done:
      self.__finetune_pruned_model(path=FLAGS.cp_channel_pruned_path, finetune=False)
    else:
      self.__finetune_pruned_model(path=FLAGS.cp_channel_pruned_path, finetune=FLAGS.cp_finetune)

    return done

  def __finetune_pruned_model(self, path=None, finetune=False):
    if path is None:
      path = FLAGS.cp_channel_pruned_path
    start = timer()
    tf.logging.info('build pruned evaluating model')
    self.__build_pruned_evaluate_model(path)
    tf.logging.info('build pruned training model')
    self.__build_pruned_train_model(path, finetune=finetune)
    tf.logging.info('training pruned model')
    self.__train_pruned_model(finetune=finetune)
    tf.logging.info('fintuning time cost: {}s'.format(timer() - start))

  def __prune_and_finetune_auto(self):
    if self.__is_primary_worker():
      self.__prune_rl()
      self.pruner.initialize_state()

    if FLAGS.enbl_multi_gpu:
      self.mpi_comm.Barrier()
      self.bestinfo = self.mpi_comm.bcast(self.bestinfo, root=0)

    ratio_list = self.bestinfo[0]
    tf.logging.info('best split ratio is: {}'.format(ratio_list))
    ratio_list.reverse()
    queue = deque(ratio_list)

    done = False
    while not done:
      done = self.__prune_list_layers(queue, [FLAGS.cp_list_group])

  @classmethod
  def __calc_reward(cls, accuracy, flops):
    if FLAGS.cp_reward_policy == 'accuracy':
      reward = accuracy * np.ones((1, 1))
    elif FLAGS.cp_reward_policy == 'flops':
      reward = -np.maximum(
        FLAGS.cp_noise_tolerance, (1 - accuracy)) * np.log(flops) * np.ones((1, 1))
    else:
      raise ValueError('unrecognized reward type: ' + FLAGS.cp_reward_policy)

    return reward

  def __prune_rl(self): # pylint: disable=too-many-locals
    """ search pruning strategy with reinforcement learning"""
    tf.logging.info(
      'preserve lower bound: {}, preserve ratio: {}, preserve upper bound: {}'.format(
        self.lbound, FLAGS.cp_preserve_ratio, self.rbound))
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(0) # pylint: disable=no-member
    buf_size = len(self.pruner.states) * FLAGS.cp_nb_rlouts_min
    nb_rlouts = FLAGS.cp_nb_rlouts
    self.agent = DdpgAgent(
      tf.Session(config=config),
      len(self.pruner.states.loc[0].tolist()),
      1,
      nb_rlouts,
      buf_size,
      self.lbound,
      self.rbound)
    self.agent.init()
    self.bestinfo = None
    reward_best = np.NINF  # pylint: disable=no-member

    for idx_rlout in range(FLAGS.cp_nb_rlouts):
      # execute roll-outs to obtain pruning ratios
      self.agent.init_rlout()
      states_n_actions = []
      self.create_pruner()
      self.pruner.initialize_state()
      self.pruner.extract_features()
      state = np.array(self.pruner.currentStates.loc[0].tolist())[None, :]

      start = timer()
      while True:
        tf.logging.info('state is {}'.format(state))
        action = self.agent.sess.run(self.agent.actions_noisy, feed_dict={self.agent.states: state})
        tf.logging.info('RL choosed preserv ratio: {}'.format(action))
        state_next, acc_flops, done, real_action = self.pruner.compress(action)
        tf.logging.info('Actural preserv ratio: {}'.format(real_action))
        states_n_actions += [(state, real_action * np.ones((1, 1)))]
        state = state_next[None, :]
        actor_loss, critic_loss, noise_std = self.agent.train()
        if done:
          break
      tf.logging.info('roll-out #%d: a-loss = %.2e | c-loss = %.2e | noise std. = %.2e'
                      % (idx_rlout, actor_loss, critic_loss, noise_std))

      reward = self.__calc_reward(acc_flops[0], acc_flops[1])

      rewards = reward * np.ones(len(self.pruner.states))
      self.agent.finalize_rlout(rewards)

      # record transactions for RL training
      strategy = []
      for idx, (state, action) in enumerate(states_n_actions):
        strategy.append(action[0, 0])
        if idx != len(states_n_actions) - 1:
          terminal = np.zeros((1, 1))
          state_next = states_n_actions[idx + 1][0]
        else:
          terminal = np.ones((1, 1))
          state_next = np.zeros_like(state)
        self.agent.record(state, action, reward, terminal, state_next)

      # record the best combination of pruning ratios
      if reward_best < reward:
        tf.logging.info('best reward updated: %.4f -> %.4f' % (reward_best, reward))
        reward_best = reward
        self.bestinfo = [strategy, acc_flops[0], acc_flops[1]]
        tf.logging.info("""The best pruned model occured with
                strategy: {},
                accuracy: {} and
                pruned ratio: {}""".format(self.bestinfo[0], self.bestinfo[1], self.bestinfo[2]))

      tf.logging.info('automatic channl pruning time cost: {}s'.format(timer() - start))


  @classmethod
  def __is_primary_worker(cls):
    """Weather it is the primary worker"""
    return not FLAGS.enbl_multi_gpu or mgw.rank() == 0
