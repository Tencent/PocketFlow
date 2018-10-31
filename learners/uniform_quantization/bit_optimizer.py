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
""" Bit Optimizer for Uniform Quantization """

import os
from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
from utils.multi_gpu_wrapper import MultiGpuWrapper as mgw
from learners.uniform_quantization.rl_helper import RLHelper
from rl_agents.ddpg.agent import Agent as DdpgAgent

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('uql_equivalent_bits', 4, \
                            'equivalent compression bits for non-rl quantization')
tf.app.flags.DEFINE_integer('uql_nb_rlouts', 200, \
                            'total number of rlouts for rl training')
tf.app.flags.DEFINE_integer('uql_w_bit_min', 2, 'minimum number of bits for weights')
tf.app.flags.DEFINE_integer('uql_w_bit_max', 8, 'maximum number of bits for weights')
tf.app.flags.DEFINE_integer('uql_tune_layerwise_steps', 100, 'fine tuning steps for each layer')
tf.app.flags.DEFINE_integer('uql_tune_global_steps', 2000, 'fine tuning steps for each layer')
tf.app.flags.DEFINE_string('uql_tune_save_path', './rl_tune_models/model.ckpt', \
                           'dir to save tuned models during rl trianing')
tf.app.flags.DEFINE_integer('uql_tune_disp_steps', 300, 'interval steps to show tuning details')
tf.app.flags.DEFINE_boolean('uql_enbl_random_layers', True, \
                            'enable random permutation of layers for the rl agent')
tf.app.flags.DEFINE_boolean('uql_enbl_rl_agent', False, \
                            'enable rl agent for uniform quantization')
tf.app.flags.DEFINE_boolean('uql_enbl_rl_global_tune', True, \
                            'Tune the weights globally before get reward or not')
tf.app.flags.DEFINE_boolean('uql_enbl_rl_layerwise_tune', False, \
                            'Tune the weights layerwisely before get reward or not')


class BitOptimizer(object):
  # pylint: disable=too-many-instance-attributes
  """ Currently only weight bits are inferred via RL. Activations later."""

  def __init__(self,
               dataset_name,
               weights,
               statistics,
               bit_placeholders,
               ops,
               layerwise_tune_list,
               sess_train,
               sess_eval,
               saver_train,
               saver_eval,
               barrier_fn):
    """ By passing the ops in the learner, we do not need to build the graph
    again for training and testing.

    Args:
    * dataset_name: a string that indicates which dataset to use
    * weights: a list of Tensors, the weights of networks to quantize
    * statistics: a dict, recording the number of weights, activations e.t.c.
    * bit_placeholders: a dict of placeholder Tensors, the input of bits
    * ops: a dict of ops, including trian_op, eval_op e.t.c.
    * layerwise_tune_list: a tuple, in which [0] records the layerwise op and
                          [1] records the layerwise l2_norm
    * sess_train: a session for train
    * sess_eval: a session for eval
    * saver_train: a Tensorflow Saver for the training graph
    * saver_eval: a Tensorflow Saver for the eval graph
    * barrier_fn: a function that implements barrier
    """
    self.dataset_name = dataset_name
    self.weights = weights
    self.statistics = statistics
    self.bit_placeholders = bit_placeholders
    self.ops = ops
    self.layerwise_tune_ops, self.layerwise_diff = \
        layerwise_tune_list[0], layerwise_tune_list[1]
    self.sess_train = sess_train
    self.sess_eval = sess_eval
    self.saver_train = saver_train
    self.saver_eval = saver_eval
    self.auto_barrier = barrier_fn

    self.total_num_weights = sum(self.statistics['num_weights'])
    self.total_bits = self.total_num_weights * FLAGS.uql_equivalent_bits

    self.w_rl_helper = RLHelper(self.sess_train,
                                self.total_bits,
                                self.statistics['num_weights'],
                                self.weights,
                                random_layers=FLAGS.uql_enbl_random_layers)

    self.mgw_size = int(mgw.size()) if FLAGS.enbl_multi_gpu else 1
    self.tune_global_steps = int(FLAGS.uql_tune_global_steps / self.mgw_size)
    self.tune_global_disp_steps = int(FLAGS.uql_tune_disp_steps / self.mgw_size)

    # build the rl trianing graph
    with tf.Graph().as_default():
      config = tf.ConfigProto()
      config.gpu_options.visible_device_list = str(mgw.local_rank() \
          if FLAGS.enbl_multi_gpu else 0)
      self.sess_rl = tf.Session(config=config)

      # train an RL agent through multiple roll-outs
      self.s_dims = self.w_rl_helper.s_dims
      self.a_dims = 1
      buff_size = len(self.weights) * int(FLAGS.uql_nb_rlouts // 4)
      self.agent = DdpgAgent(self.sess_rl,
                             self.s_dims,
                             self.a_dims,
                             FLAGS.uql_nb_rlouts,
                             buff_size,
                             a_min=0.,
                             a_max=FLAGS.uql_w_bit_max-FLAGS.uql_w_bit_min)

  def run(self):
    """ get the bit allocation strategy either with RL or not """
    if FLAGS.uql_enbl_rl_agent:
      optimal_w_bits, optimal_a_bits = self.__calc_optimal_bits()
    else:
      optimal_w_bits = [FLAGS.uql_weight_bits] * self.statistics['nb_matmuls']
      optimal_a_bits = [FLAGS.uql_activation_bits] * self.statistics['nb_activations']
    return optimal_w_bits, optimal_a_bits

  def __calc_optimal_bits(self):
    self.sess_train.run(self.ops['init'])

    if FLAGS.enbl_multi_gpu:
      self.sess_train.run(self.ops['bcast'])

    # fp_w_bit_list = [32] * self.statistics['nb_matmuls']
    fp_a_bit_list = [32] * self.statistics['nb_activations']

    if self.__is_primary_worker():
      # initialize rl agent
      self.agent.init()
      reward_list = []
      optimal_reward = np.NINF
      optimal_arranged_w_bit_list = None

    for idx_rlout in range(FLAGS.uql_nb_rlouts):
      # execute roll-outs to collect transactions
      if self.__is_primary_worker():
        tf.logging.info('starting %d-th roll-out:' % idx_rlout)
        states_n_actions = self.__calc_rollout_actions(idx_rlout)

      # wait until all workers can read file
      self.auto_barrier()
      arranged_layer_bits = self.__sync_list_read('./arranged_layer_bits.txt')
      feed_dict_train = {self.bit_placeholders['w_train']: arranged_layer_bits,
                         self.bit_placeholders['a_train']: fp_a_bit_list}
      feed_dict_eval = {self.bit_placeholders['w_eval']: arranged_layer_bits,
                        self.bit_placeholders['a_eval']: fp_a_bit_list}

      reward = self.__calc_rollout_reward(feed_dict_train,
                                          feed_dict_eval,
                                          arranged_layer_bits)
      self.auto_barrier()

      if self.__is_primary_worker():
        os.remove('./arranged_layer_bits.txt')
        reward_list.append(reward[0][0])
        self.agent.finalize_rlout(reward)

        # record transactions for RL training
        self.__record_rollout_transitions(states_n_actions, reward)
        self.__train_rl_agent(idx_rlout)

        if optimal_reward < reward:
          optimal_reward = reward
          optimal_arranged_w_bit_list = arranged_layer_bits

      self.auto_barrier()

    if self.__is_primary_worker():
      tf.logging.info("Finished RL training")
      tf.logging.info("Optimal reward: {0}, Optimal w_bit_list: {1}" \
        .format(optimal_reward, optimal_arranged_w_bit_list))
      self.__sync_list_write(optimal_arranged_w_bit_list, './optimal_arranged_w_bit_list.txt')

    self.auto_barrier()
    optimal_arranged_w_bit_list = self.__sync_list_read('./optimal_arranged_w_bit_list.txt')
    return optimal_arranged_w_bit_list, fp_a_bit_list

  def __calc_rollout_reward(self, feed_dict_train, feed_dict_eval, layer_bits):
    # Before getting the reward, do the finetuning if necessary
    if FLAGS.uql_enbl_rl_global_tune or FLAGS.uql_enbl_rl_layerwise_tune:
      # restore the training graph for fine tune
      save_path = tf.train.latest_checkpoint(os.path.dirname(FLAGS.save_path))
      self.saver_train.restore(self.sess_train, save_path)
      if FLAGS.enbl_multi_gpu:
        self.sess_train.run(self.ops['bcast'])

    # TODO: Layerwise FT can only be done by one card currently.
    if FLAGS.uql_enbl_rl_layerwise_tune and self.__is_primary_worker():
      self.__layerwise_finetune(feed_dict_train, layer_bits)
    self.auto_barrier()

    # start global tune
    if FLAGS.uql_enbl_rl_global_tune:
      self.__global_finetune(feed_dict_train)

    # save the tuned models and restore it to acquire the rewardh
    if self.__is_primary_worker():
      self.saver_train.save(self.sess_train, FLAGS.uql_tune_save_path)
      save_path = tf.train.latest_checkpoint(os.path.dirname(FLAGS.uql_tune_save_path))
      self.saver_eval.restore(self.sess_eval, save_path)
      __, acc_top1, acc_top5 = self.__calc_loss_n_accuracy(feed_dict=feed_dict_eval)
      if self.dataset_name == 'cifar_10':
        reward = self.w_rl_helper.calc_reward(acc_top1)
      elif self.dataset_name == 'ilsvrc_12':
        reward = self.w_rl_helper.calc_reward(acc_top5)
      else:
        raise ValueError("Unknown dataset name")
      tf.logging.info('acc_top1 = %.4f | acc_top5 = %.4f | reward = %.4f'\
                      % (acc_top1, acc_top5, reward[0][0]))
      return reward
    else:
      return None

  def __layerwise_finetune(self, feed_dict_train, layer_bits):
    # can only be called in a single thread
    # TODO: the results are somehow not good yet.
    for n in range(self.statistics['nb_matmuls']):
      for t_step in range(FLAGS.uql_tune_layerwise_steps):
        _, diff = self.sess_train.run([self.layerwise_tune_ops[n], self.layerwise_diff[n]],
                                      feed_dict=feed_dict_train)
        if (t_step+1) % 20 == 0:
          tf.logging.info("Layerwise Tuning: {}, Step: {}, Bit: {}, Layer diff norm: {}"\
                          .format(n, t_step+1, layer_bits[n], diff))
    tf.logging.info("Layerwise finetuning done")

  def __global_finetune(self, feed_dict_train):
    time_prev = timer()
    for t_step in range(self.tune_global_steps):
      _, log_rslt = self.sess_train.run([self.ops['train'], self.ops['log']],
                                        feed_dict=feed_dict_train)
      if (t_step+1) % self.tune_global_disp_steps == 0:
        time_prev = self.__monitor_progress(t_step, log_rslt, time_prev)

    # reset the fine-tune step to initialize the lrn_rate
    self.sess_train.run(self.ops['reset_ft_step'])

  def __calc_rollout_actions(self, idx_rlout):
    # can only be called in single thread
    self.agent.init_rlout()
    self.w_rl_helper.reset()
    states_n_actions = [(None, None)] * self.statistics['nb_matmuls']  # store the unshuffled data
    w_bit_list = [] # store the real quant strategy. NOTE: could be shuffled.

    for idx in self.w_rl_helper.layer_idxs:
      state = self.w_rl_helper.calc_state(idx)
      action = self.sess_rl.run(self.agent.actions_noisy, feed_dict={self.agent.states: state})
      action = self.w_rl_helper.calc_w(action, idx)
      assert action[0][0] >= 1 and action[0][0] <= 32, \
          'the quantization bits must be in [1, 32]'
      assert np.shape(action) == (1, 1), '"action" must be in shape (1,1)'
      states_n_actions[idx] = (state, action)
      w_bit_list.append(action[0][0])

    arranged_layer_bits, ratio = self.__arrange_layer_bits(self.w_rl_helper.layer_idxs, w_bit_list)
    tf.logging.info('Un-allocated bit percentage: %.3f' % ratio)
    tf.logging.info('#_rlout: {0}, layer_bits: {1}'.format(idx_rlout, arranged_layer_bits))
    self.__sync_list_write(arranged_layer_bits, './arranged_layer_bits.txt')

    return states_n_actions

  def __train_rl_agent(self, idx_rlout):
    # can only be called in single thread
    for _ in range(self.statistics['nb_matmuls']):
      actor_loss, critic_loss, param_noise_std = self.agent.train()
      tf.logging.info('roll-out #%d: a-loss = %.2e | c-loss = %.2e | noise std. = %.2e'
                      % (idx_rlout, actor_loss, critic_loss, param_noise_std))

  def __calc_loss_n_accuracy(self, feed_dict):
    """ evaluate on the validation set """
    losses, acc_top1_list, acc_top5_list = [], [], []
    nb_iters = FLAGS.nb_smpls_eval // FLAGS.batch_size_eval
    for _ in range(nb_iters):
      eval_rslt = self.sess_eval.run(self.ops['eval'], feed_dict=feed_dict)
      losses.append(eval_rslt[0])
      acc_top1_list.append(eval_rslt[1])
      acc_top5_list.append(eval_rslt[2])
    return np.mean(np.array(losses)), \
        np.mean(np.array(acc_top1_list)), np.mean(np.array(acc_top5_list))

  def __record_rollout_transitions(self, states_n_actions, reward):
    # record transitions from the current rollout to train the RL agent
    for n in range(self.statistics['nb_matmuls']):
      state, action = states_n_actions[n]
      if n != self.statistics['nb_matmuls'] - 1:
        terminal = np.zeros((1, 1))
        state_next = states_n_actions[n+1][0]
      else:
        terminal = np.ones((1, 1))
        state_next = np.zeros((1, self.s_dims))
      self.agent.record(state, action, reward, terminal, state_next)

  def __arrange_layer_bits(self, layer_idxs, w_bit_list):
    arranged_w_bit_list = [-1] * self.statistics['nb_matmuls']
    for i in range(self.statistics['nb_matmuls']):
      arranged_w_bit_list[layer_idxs[i]] = w_bit_list[i]
    assert -1 not in arranged_w_bit_list, "Some layers are not assigned with proper bits"
    ratio = self.__check_bits(arranged_w_bit_list)
    return arranged_w_bit_list, ratio

  def __check_bits(self, bit_list):
    used_bits = 0
    for (v, p) in zip(bit_list, self.statistics['num_weights']):
      used_bits += v * p
    if self.total_bits >= used_bits:
      return (self.total_bits-used_bits)/self.total_bits
    else:
      raise ValueError("The average bit is out of constraint")

  def __monitor_progress(self, idx_iter, log_rslt, time_prev):
    if not self.__is_primary_worker():
      return None

    # display monitored statistics
    speed = FLAGS.batch_size * self.tune_global_disp_steps / (timer() - time_prev)
    if FLAGS.enbl_multi_gpu:
      speed *= mgw.size()
    if FLAGS.enbl_dst:
      lrn_rate, dst_loss, model_loss, loss, acc_top1, acc_top5 = log_rslt[0], \
          log_rslt[1], log_rslt[2], log_rslt[3], log_rslt[4], log_rslt[5]
      tf.logging.info('iter #%d: lr = %e | dst_loss = %e | model_loss = %e | loss = %e | acc_top1 = %e | acc_top5 = %e | speed = %.2f pics / sec ' \
          % (idx_iter + 1, lrn_rate, dst_loss, model_loss, loss, acc_top1, acc_top5, speed))
    else:
      lrn_rate, model_loss, loss, acc_top1, acc_top5 = log_rslt[0], \
          log_rslt[1], log_rslt[2], log_rslt[3], log_rslt[4]
      tf.logging.info('iter #%d: lr = %e | model_loss = %e | loss = %e | acc_top1 = %e | acc_top5 = %e| speed = %.2f pics / sec' \
          % (idx_iter + 1, lrn_rate, model_loss, loss, acc_top1, acc_top5, speed))
    return timer()

  @classmethod
  def __is_primary_worker(cls):
    return not FLAGS.enbl_multi_gpu or mgw.rank() == 0

  @classmethod
  def __sync_list_write(cls, bit_list, file_name):
    f = open(file_name, 'w')
    for bit in bit_list:
      f.write(str(bit) + ' ')
    f.close()

  @classmethod
  def __sync_list_read(cls, file_name):
    f = open(file_name, 'r')
    for l in f.readlines():
      bit_list = l.split()
    f.close()
    return [round(float(bit)) for bit in bit_list]

