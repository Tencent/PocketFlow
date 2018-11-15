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
"""Channel Pruner
--Ref.
A. https://arxiv.org/abs/1707.06168
"""
from collections import OrderedDict
import os
import math
from timeit import default_timer as timer
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LinearRegression
import tensorflow as tf
import numpy as np
import pandas as pd


slim = tf.contrib.slim # pylint: disable=no-member
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean(
  'cp_lasso',
  True,
  'If True use lasso and reconstruction otherwise prune according to weight magnitude')
tf.app.flags.DEFINE_boolean('cp_quadruple', False,
                            'Restric the channels after pruning is a mutiple of 4')
tf.app.flags.DEFINE_string('cp_reward_policy', 'accuracy',
                           '''If reward_policy equals accuracy, it means learning to
                           achieve high accuracy with guaranteed low flops, else if
                           reward_policy equals flops, it means learning to
                           achieve low flops with guaranted accuracy.''')
tf.app.flags.DEFINE_integer('cp_nb_points_per_layer', 10,
                            'Sample how many point for each layer')
tf.app.flags.DEFINE_integer('cp_nb_batches', 30,
                            'Input how many bathes data into a model')


class ChannelPruner(object): # pylint: disable=too-many-instance-attributes
  """ The Channel Prunner """
  def __init__(self, model,
               images=None,
               labels=None,
               mem_images=None,
               mem_labels=None,
               metrics=None,
               summary_op=None,
               sm_writer=None,
               lbound=0,
               state=0):
    self._model = model
    self.metrics = metrics
    self.summary_op = summary_op
    self.sm_writer = sm_writer
    self.metrics = metrics
    self.acc = metrics['accuracy']
    self.data_format = self._model.data_format
    self.images = images
    self.labels = labels
    self.mem_images = mem_images
    self.mem_labels = mem_labels
    self.saver = tf.train.Saver()
    self.temp_saver = tf.train.Saver()
    self.state = state
    self.lbound = lbound
    self.best = -math.inf
    self.bestinfo = []
    self.states = []
    self.names = []
    self.state = 0
    self.thisconvs = self._model.get_operations_by_type()
    self.sm_writer.add_graph(self._model.g)
    self.drop_trainable_vars = set([])
    self.currentStates = []
    self.desired_reduce = 0
    self.feats_dict = {}
    self.points_dict = {}
    self.desired_preserve = 0
    self.drop_conv = set([])
    self.layer_flops = 0
    self.model_flops = 0
    self.max_reduced_flops = 0
    self.config = tf.ConfigProto()
    self.config.gpu_options.visible_device_list = str(0) # pylint: disable=no-member
    self.max_strategy_dict = {}
    self.fake_pruning_dict = {}
    self.extractors = {}
    self.__build()

  def __build(self):
    self.__extract_output_of_conv_and_sum()
    self.__create_extractor()
    self.initialize_state()

  def initialize_state(self):
    """Initialize state"""
    self.best = -math.inf
    self.bestinfo = []
    allstate = []

    self.state = 0
    while self.state < len(self.thisconvs):
      allstate.append(self.getState(self.thisconvs[self.state]))
      self.state += 1
    feature_names = ['layer', 'n', 'c', 'H',
                     'W', 'stride', 'maxreduce',
                     'layercomp']
    states = pd.DataFrame(allstate, columns=feature_names)
    self.state = 0
    self.states = states / states.max()
    self.layer_flops = np.array(self.states['layercomp'].tolist())
    self.model_flops = self.__compute_model_flops()
    tf.logging.info('The original model flops is {}'.format(self.model_flops))

    self.currentStates = self.states.copy()
    self.desired_reduce = (1 - FLAGS.cp_preserve_ratio) * self.model_flops
    self.desired_preserve = FLAGS.cp_preserve_ratio * self.model_flops
    self.max_strategy_dict = {} # collection of intilial max [inp preserve, out preserve]
    self.fake_pruning_dict = {} # collection of fake pruning indices
    for i, conv in enumerate(self.thisconvs):
      if self._model.is_W1_prunable(conv):
        tf.logging.info('current conv ' + conv.name)
        father_conv_name = self._model.fathers[conv.name]
        father_conv = self._model.g.get_operation_by_name(father_conv_name)
        if father_conv.type == \
          'DepthwiseConv2dNative':
          if self._model.is_W1_prunable(father_conv):
            self.max_strategy_dict[self._model.fathers[father_conv_name]][1] = self.lbound
        else:
          self.max_strategy_dict[father_conv_name][1] = self.lbound
      if not (i == 0 or i == len(self.thisconvs) - 1):
        self.max_strategy_dict[conv.name] = [self.lbound, 1.]
      else:
        self.max_strategy_dict[conv.name] = [1., 1.]
      conv_def = self._model.get_conv_def(conv)
      self.fake_pruning_dict[conv.name] = [[True] * conv_def['c'], [True] * conv_def['n']]

    tf.logging.info('current states:\n {}'.format(self.currentStates))
    tf.logging.info('max_strategy_dict\n {}'.format(self.max_strategy_dict))

  def getState(self, op):
    """Get state"""
    conv_def = self._model.get_conv_def(op)
    n, c, _, _ = conv_def['n'], conv_def['c'], conv_def['h'], conv_def['w']
    conv = self._model.get_outname_by_opname(op.name)
    W = self._model.output_width(conv)
    H = self._model.output_height(conv)
    stride = conv_def['strides'][1]

    return [self.state, n, c, H,
            W, stride, 1., self._model.compute_layer_flops(op)]

  def __action_constraint(self, action):
    """constraint action during reinfocement learning search"""
    action = float(action)
    if action > 1.:
      action = 1.
    if action < 0.:
      action = 0.

    # final layer is not prunable
    if self.finallayer():
      return 1

    conv_op = self.thisconvs[self.state]

    prunable = self._model.is_W1_prunable(conv_op)
    if prunable:
      father_opname = self._model.fathers[conv_op.name]

    conv_left = self.__conv_left()
    this_flops = 0
    other_flops = 0
    behind_layers_start = False
    for conv in conv_left:
      curr_flops = self._model.compute_layer_flops(conv)

      if prunable and conv.name == father_opname:
        this_flops += curr_flops * self.max_strategy_dict[conv.name][0]
      elif conv.name == conv_op.name:
        this_flops += curr_flops * self.max_strategy_dict[conv.name][1]
        behind_layers_start = True
      elif behind_layers_start and 'pruned' not in conv.name:
        other_flops += curr_flops * self.max_strategy_dict[conv.name][0] * \
          self.max_strategy_dict[conv.name][1]
      else:
        other_flops += curr_flops * self.max_strategy_dict[conv.name][0] * \
          self.max_strategy_dict[conv.name][1]

    self.max_reduced_flops = other_flops + this_flops * action
    if FLAGS.cp_reward_policy != 'accuracy' or self.state == 0:
      return action

    recommand_action = (self.desired_preserve - other_flops) / this_flops
    tf.logging.info('max_reduced_flops {}'.format(self.max_reduced_flops))
    tf.logging.info('desired_preserce {}'.format(self.desired_preserve))
    tf.logging.info('this flops {}'.format(this_flops))
    tf.logging.info('recommand action {}'.format(recommand_action))

    return np.minimum(action, recommand_action)

  def __extract_output_of_conv_and_sum(self):
    """Extract output tensor name of convolution layers and sum layers in a residual block"""
    conv_outputs = self._model.get_outputs_by_type()
    conv_add_outputs = []
    for conv in conv_outputs:
      conv_add_outputs.append(conv)
      add_output = self._model.get_Add_if_is_last_in_resblock(conv.op)
      if add_output != None:
        conv_add_outputs.append(add_output)

    tf.logging.debug('extracted outputs {}'.format(conv_add_outputs))

    self.names = self._model.get_names(conv_add_outputs)

  def __conv_left(self):
    """Get the left convolutions after pruning so far"""
    conv_ops = self._model.get_operations_by_type()
    conv_left = []
    for conv in conv_ops:
      if conv.name not in self.drop_conv:
        conv_left.append(conv)

    tf.logging.debug('drop conv {}'.format(self.drop_conv))
    tf.logging.debug('conv left {}'.format(conv_left))

    return conv_left


  def __compute_model_flops(self, fake=False):
    """Compute the convolution computation flops of the model"""
    conv_left = self.__conv_left()

    flops = 0.
    for op in conv_left:
      if fake:
        flops += self._model.compute_layer_flops(op) * \
          self.max_strategy_dict[op.name][0] * self.max_strategy_dict[op.name][1]
      else:
        flops += self._model.compute_layer_flops(op)
    tf.logging.info('The current model flops is {}'.format(flops))

    return flops

  @property
  def model(self):
    """Return the model"""
    return self._model

  def extract_features( # pylint: disable=too-many-locals
      self,
      names=None,
      init_fn=None,
      images=None,
      labels=None):
    """
    Extract feature-maps and do sampling for some convolutions with given images

    Args:
      names: convolution operation names
      init_fn: initialization function
      images: input images
      labels: input lables
    """
    if names is None:
      names = self.names
    if images is None:
      images = self.images
    if labels is None:
      labels = self.labels

    def set_points_dict(name, data):
      """ set data to point dict"""
      if name in points_dict:
        raise ValueError("{} is in the points_dict".format(name))
      points_dict[name] = data

    # remove duplicates
    names = list(OrderedDict.fromkeys(names))
    points_dict = dict()
    feats_dict = {}
    shapes = {}

    nb_points_per_batch = FLAGS.cp_nb_points_per_layer * FLAGS.batch_size
    set_points_dict("nb_points_per_batch", nb_points_per_batch)

    nb_points_total = nb_points_per_batch * FLAGS.cp_nb_batches
    for name in names:
      shapes[name] = (self._model.output_height(name), self._model.output_width(name))
      feats_dict[name] = np.ndarray(shape=(nb_points_total, self._model.output_channels(name)))

    # extract bathes of input images and labels
    with self._model.g.as_default():
      with slim.queues.QueueRunners(self._model.sess):
        if init_fn:
          init_fn(self._model.sess)
        batches = []
        for batch in range(FLAGS.cp_nb_batches):
          np_images_raw, np_labels = self._model.sess.run([images, labels])
          batches.append([np_images_raw, np_labels])
        data_and_label = batches

    ## get the output of corresponding layer and do smampling
    idx = 0
    for batch in range(FLAGS.cp_nb_batches):
      data = data_and_label[batch][0]
      label = data_and_label[batch][1]
      set_points_dict((batch, 0), data)
      set_points_dict((batch, 1), label)
      with tf.variable_scope('train'):
        feats = self._model.sess.run(names, feed_dict={self.mem_images: data})
      for feat, name in zip(feats, names):
        shape = shapes[name]
        x_samples = np.random.randint(0, shape[0] - 0, FLAGS.cp_nb_points_per_layer)
        y_samples = np.random.randint(0, shape[1] - 0, FLAGS.cp_nb_points_per_layer)
        set_points_dict((batch, name, "x_samples"), x_samples.copy())
        set_points_dict((batch, name, "y_samples"), y_samples.copy())
        if self.data_format == 'NCHW':
          feats_dict[name][idx:(idx + nb_points_per_batch)] = \
            np.transpose(
              feat[:, :, x_samples, y_samples], (0, 2, 1)).reshape((nb_points_per_batch, -1))
        else:
          feats_dict[name][idx:(idx + nb_points_per_batch)] = \
            feat[:, x_samples, y_samples, :].reshape((nb_points_per_batch, -1))
      idx += nb_points_per_batch

    self.feats_dict = feats_dict
    self.points_dict = points_dict

  def __create_extractor(self):
    """ create extracters which would be used to extract input of a convolution"""
    with self._model.g.as_default():
      ops = self._model.get_operations_by_type()
      self.extractors = {}
      for op in ops:
        inp = self._model.get_input_by_op(op)
        if self.data_format == 'NCHW':
          inp = tf.transpose(inp, [0, 2, 3, 1])
        defs = self._model.get_conv_def(op)
        strides = defs['strides']
        extractor = tf.extract_image_patches(inp,
                                             ksizes=defs['ksizes'],
                                             strides=strides,
                                             rates=[1, 1, 1, 1],
                                             padding=defs['padding'])
        self.extractors[op.name] = extractor

  def __extract_new_features(self, names=None):
    """ extract new feature map via re-sampling some points"""
    nb_points_per_batch = self.points_dict["nb_points_per_batch"]
    feats_dict = {}
    shapes = {}
    nb_points_total = nb_points_per_batch * FLAGS.cp_nb_batches
    idx = 0

    for name in names:
      shapes[name] = (self._model.output_height(name), self._model.output_width(name))
      feats_dict[name] = np.ndarray(shape=(nb_points_total, self._model.output_channels(name)))

    for batch in range(FLAGS.cp_nb_batches):
      feats = self._model.sess.run(names,
                                   feed_dict={self.mem_images: self.points_dict[(batch, 0)]})

      for feat, name in zip(feats, names):
        x_samples = self.points_dict[(batch, name, "x_samples")]
        y_samples = self.points_dict[(batch, name, "y_samples")]
        if self.data_format == 'NCHW':
          feats_dict[name][idx:(idx + nb_points_per_batch)] = \
            np.transpose(
              feat[:, :, x_samples, y_samples], (0, 2, 1)).reshape((nb_points_per_batch, -1))
        else:
          feats_dict[name][idx:(idx + nb_points_per_batch)] = \
            feat[:, x_samples, y_samples, :].reshape((nb_points_per_batch, -1))

      idx += nb_points_per_batch
    return feats_dict

  def __extract_input(self, conv):
    """extract the input X (k_h, k_w, c) of a conv layer
    Args:
        conv: a convolution operation
    Returns:
        bathces of X (N, k_h, k_w, c)
    """
    opname = conv.name
    outname = self._model.get_outname_by_opname(opname)
    extractor = self.extractors[opname]
    Xs = []
    def_ = self._model.get_conv_def(conv)
    for batch in range(FLAGS.cp_nb_batches):
      feat = self._model.sess.run(extractor,
                                  feed_dict={self.mem_images: self.points_dict[(batch, 0)]})
      x_samples = self.points_dict[(batch, outname, "x_samples")]
      y_samples = self.points_dict[(batch, outname, "y_samples")]

      X = feat[:, x_samples, y_samples, :].reshape((-1, feat.shape[-1]))
      X = X.reshape((X.shape[0], def_['h'], def_['w'], def_['c']))
      Xs.append(X)
    return np.vstack(Xs)

  def accuracy(self):
    """Calculate the accuracy of pruned model"""
    acc_list = []
    metrics_list = [[] for i in range(len(list(self.metrics.keys())))]
    for batch in range(FLAGS.cp_nb_batches):
      metrics = self._model.sess.run([self.acc, self.summary_op] + \
        list(self.metrics.values()), feed_dict={
          self.mem_images: self.points_dict[(batch, 0)],
          self.mem_labels: self.points_dict[(batch, 1)]})
      acc_list.append(metrics[0])
      self.sm_writer.add_summary(metrics[1], batch)
      for i, m in enumerate(metrics_list):
        m.append(metrics[i + 2])
    acc = np.mean(acc_list)
    i = 0
    for key in self.metrics.keys():
      value = np.mean(metrics_list[i])
      tf.logging.info('{}: {}'.format(key, value))
      i += 1

    return float(acc)

  @classmethod
  def rel_error(cls, A, B):
    """calcualte relative error"""
    return np.mean((A - B) ** 2) ** .5 / np.mean(A ** 2) ** .5


  @classmethod
  def featuremap_reconstruction(cls, x, y, copy_x=True, fit_intercept=False):
    """Given changed input X, used linear regression to reconstruct original Y

      Args:
        x: The pruned input
        y: The original feature map of the convolution layer
      Return:
        new weights and bias which can reconstruct the feature map with small loss given X
    """
    _reg = LinearRegression(n_jobs=-1, copy_X=copy_x, fit_intercept=fit_intercept)
    _reg.fit(x, y)
    return _reg.coef_, _reg.intercept_

  def compute_pruned_kernel( # pylint: disable=too-many-locals,too-many-branches,too-many-statements
      self,
      X,
      W2,
      Y,
      alpha=1e-4,
      c_new=None,
      tolerance=0.02):
    """compute which channels to be pruned by lasso"""

    tf.logging.info('computing pruned kernel')

    nb_samples = X.shape[0]
    c_in = X.shape[-1]
    c_out = W2.shape[-1]
    samples = np.random.randint(0, nb_samples, min(400, nb_samples // 20))
    reshape_X = np.rollaxis(
      np.transpose(X, (0, 3, 1, 2)).reshape((nb_samples, c_in, -1))[samples], 1, 0)
    reshape_W2 = np.transpose(np.transpose(W2, (3, 2, 0, 1)).reshape((c_out, c_in, -1)), [1, 2, 0])
    product = np.matmul(reshape_X, reshape_W2).reshape((c_in, -1)).T
    reshape_Y = Y[samples].reshape(-1)

    # feature
    tmp = np.nonzero(np.sum(np.abs(product), 0))[0].size
    if FLAGS.debug:
      tf.logging.info('feature num: {}, non zero: {}'.format(product.shape[1], tmp))

    solver = LassoLars(alpha=alpha, fit_intercept=False, max_iter=3000)

    def solve(alpha):
      """ Solve the Lasso"""
      solver.alpha = alpha
      solver.fit(product, reshape_Y)
      idxs = solver.coef_ != 0.
      tmp = sum(idxs)
      return idxs, tmp, solver.coef_

    tf.logging.info('pruned channel selecting')
    start = timer()

    if c_new == c_in:
      idxs = np.array([True] * c_new)
    else:
      left = 0
      right = alpha
      lbound = c_new - tolerance * c_in / 2
      rbound = c_new + tolerance * c_in / 2

      while True:
        _, tmp, coef = solve(right)
        if tmp < c_new:
          break
        else:
          right *= 2
          if FLAGS.debug:
            tf.logging.debug("relax right to {}".format(right))
            tf.logging.debug(
              "we expect got less than {} channels, but got {} channels".format(c_new, tmp))

      while True:
        if lbound < 0:
          lbound = 1
        idxs, tmp, coef = solve(alpha)
        # print loss
        loss = 1 / (2 * float(product.shape[0])) * \
          np.sqrt(np.sum((reshape_Y - np.matmul(product, coef)) ** 2, axis=0)) + \
            alpha * np.sum(np.fabs(coef))

        if FLAGS.debug:
          tf.logging.debug(
            'loss: {}, alpha: {}, feature nums: {}, left: {}, right: {}, \
              left_bound: {}, right_bound: {}'.format(
                loss, alpha, tmp, left, right, lbound, rbound))

        if FLAGS.debug:
          tf.logging.info('tmp {}, lbound {}, rbound {}, alpha {}, left {}, right {}'.format(
            tmp, lbound, rbound, alpha, left, right))
        if FLAGS.cp_quadruple:
          if tmp % 4 == 0 and abs(tmp - lbound) <= 2:
            break

        if lbound <= tmp and tmp <= rbound:
          if FLAGS.cp_quadruple:
            if tmp % 4 == 0:
              break
            elif tmp % 4 <= 2:
              rbound = tmp - 1
              lbound = lbound - 2
            else:
              lbound = tmp + 1
              rbound = rbound + 2
          else:
            break
        elif abs(left - right) <= right * 0.1:
          if lbound > 1:
            lbound = lbound - 1
          if rbound < c_in:
            rbound = rbound + 1
          left = left / 1.2
          right = right * 1.2
        elif tmp > rbound:
          left = left + (alpha - left) / 2
        else:
          right = right - (right - alpha) / 2

        if alpha < 1e-10:
          break

        alpha = (left + right) / 2
      c_new = tmp

    tf.logging.info('Channel selection time cost: {}s'.format(timer() - start))

    start = timer()
    tf.logging.info('Feature map reconstructing')
    newW2, _ = self.featuremap_reconstruction(X[:, :, :, idxs].reshape((nb_samples, -1)),
                                              Y,
                                              fit_intercept=False)

    tf.logging.info('Feature map reconstruction time cost: {}s'.format(timer() - start))

    return idxs, newW2

  def residual_branch_diff(self, sum_name):
    """ calculate the difference between before and after weight pruning for a certain branch sum"""
    tf.logging.info("approximating residual branch diff")
    residual_diff = 0
    feats_dict = self.__extract_new_features([sum_name])
    residual_diff = (self.feats_dict[sum_name]) - (feats_dict[sum_name])

    return residual_diff

  def prune_kernel(self, op, nb_channel_new): # pylint: disable=too-many-locals
    """prune the input of op by nb_channel_new
    Args:
        op: the convolution operation to be pruned.
        nb_channel_new: preserving ratio (0, 1]
    Return:
        idxs: the indices of channels to be kept
        newW2: new weight after pruning
        nb_channel_new: actual channel after pruned
    """
    tf.logging.info('pruning kernel')
    definition = self._model.get_conv_def(op)
    h, w, c, _ = definition['h'], definition['w'], definition['c'], definition['n']

    try:
      assert nb_channel_new <= 1., \
        'pruning rate should be less than or equal to 1, while it\'s {}'.format(nb_channel_new)
    except AssertionError as error:
      tf.logging.error(error)
    nb_channel_new = max(int(np.around(c * nb_channel_new)), 1)
    outname = self._model.get_outname_by_opname(op.name)
    newX = self.__extract_input(op)
    Y = self.feats_dict[outname]
    add = self._model.get_Add_if_is_last_in_resblock(op)
    if add != None:
      Y = Y + self.residual_branch_diff(add.name)
      tf.logging.debug('residual_branch_diff: {}'.format(self.residual_branch_diff(add.name)))
    W2 = self._model.param_data(op)
    tf.logging.debug('original feature map rmse: {}'.format(
      self.rel_error(newX.reshape(newX.shape[0], -1).dot(W2.reshape(-1, W2.shape[-1])), Y)))

    if FLAGS.cp_lasso:
      idxs, newW2 = self.compute_pruned_kernel(newX, W2, Y, c_new=nb_channel_new)
      #tf.logging.info('idxs1 {}'.format(idxs))
    else:
      idxs = np.argsort(-np.abs(W2).sum((0, 1, 3)))
      mask = np.zeros(len(idxs), bool)
      idxs = idxs[:nb_channel_new]
      mask[idxs] = True
      idxs = mask
      reg = LinearRegression(fit_intercept=False)
      reg.fit(newX[:, :, :, idxs].reshape(newX.shape[0], -1), Y)
      newW2 = reg.coef_
      #tf.logging.info('idxs2 {}'.format(idxs))

    tf.logging.debug(
      'feature map rmse: {}'.format(
        self.rel_error(newX[:, :, :, idxs].reshape(newX.shape[0], -1).dot(newW2.T), Y)))
    tf.logging.info('Prune {} c_in from {} to {}'.format(op.name, newX.shape[-1], sum(idxs)))
    nb_channel_new = sum(idxs)
    newW2 = newW2.reshape(-1, h, w, nb_channel_new)
    newW2 = np.transpose(newW2, (1, 2, 3, 0))
    return idxs, newW2, nb_channel_new / len(idxs)

  def finallayer(self, offset=1):
    """ whether final layer reached"""
    return len(self.thisconvs) - offset == self.state

  def __add_drop_train_vars(self, op):
    """ Add the dropped train variable to the `self.drop_trainable_vars`
      and dropped convolution name to the `drop_conv`

      Args:
        op: An drop operation
    """
    with self._model.g.as_default():
      train_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=os.path.split(op.name)[0] + '/')
      train_vars += tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope=os.path.split(op.name)[0].replace('MobilenetV1/MobilenetV1', 'MobilenetV1') + '/')
      train_vars_names = list(map(lambda x: x.name, train_vars))
      self.drop_trainable_vars.update(train_vars_names)
      self.drop_conv.update([op.name])
      if FLAGS.debug is True:
        tf.logging.info('op {}'.format(op.name))

  def prune_W1(self, father_conv, idxs):
    """ Prune the previous layer weight (channel dimension of the input)

      Args:
        father_conv: previous convolution
        idxs: the indices of channels to be kept
        conv_input: the original input of the convolution operation
      return:
        the output of the previous layer after pruning
    """
    self.max_strategy_dict[father_conv.name][1] = sum(idxs) / len(idxs)
    self.fake_pruning_dict[father_conv.name][1] = idxs

    # assign fake pruned weights
    weight = self._model.get_var_by_op(father_conv)
    fake_pruned_weight = weight.eval(self._model.sess)
    not_idxs = [not i for i in idxs]
    if father_conv.type == 'DepthwiseConv2dNative':
      fake_pruned_weight[:, :, not_idxs, :] = 0
    else:
      fake_pruned_weight[:, :, :, not_idxs] = 0
    self._model.sess.run(tf.assign(weight, fake_pruned_weight))

    # assign fake pruned bias
    bias_list = slim.get_variables_by_name(os.path.split(father_conv.name)[0] + '/bias')
    if bias_list:
      bias = bias_list[0]
      fake_pruned_bias = bias.eval(self._model.sess)
      fake_pruned_bias[not_idxs] = 0
      self._model.sess.run(tf.assign(bias, fake_pruned_bias))
    output = None

    self.sm_writer.add_graph(self._model.g)

    return output

  def prune_W2(self, conv_op, idxs, W2=None):
    """ Prune the current layer weight (channel dimension of the output)

      Args:
        conv_op: the current convolution operation
        conv_input: the original input of the convolution operation
        W2: the new W2
      return:
        the output of the current convolution opeartion
    """

    self.max_strategy_dict[conv_op.name][0] = sum(idxs) / len(idxs)
    self.fake_pruning_dict[conv_op.name][0] = idxs
    # assign fake pruned weights
    weight = self._model.get_var_by_op(conv_op)
    fake_pruned_weight = weight.eval(self._model.sess)
    if W2 is not None:
      fake_pruned_weight[:, :, idxs, :] = W2
    not_idxs = [not i for i in idxs]
    fake_pruned_weight[:, :, not_idxs, :] = 0
    self._model.sess.run(tf.assign(weight, fake_pruned_weight))

    output = None

    return output

  def compress(self, c_ratio): # pylint: disable=too-many-branches
    """ Compress the model by channel pruning

    Args:
        action: preserving ratio
    """
    # first layer is not prunable
    if self.state == 0:
      c_ratio = 1.0
      self.accuracy()


    # final layer is not prunable
    if self.finallayer():
      c_ratio = 1

    if FLAGS.cp_prune_option == 'auto':
      tf.logging.info('preserve ratio before constraint {}'.format(c_ratio))
      c_ratio = self.__action_constraint(c_ratio)
      tf.logging.info('preserve ratio after constraint {}'.format(c_ratio))
    conv_op = self.thisconvs[self.state]

    #if c_ratio == 1:
    if c_ratio == 1:
      if FLAGS.cp_prune_option == 'auto':
        self.max_strategy_dict[conv_op.name][0] = c_ratio
        if self._model.is_W1_prunable(conv_op):
          self.max_strategy_dict[self._model.fathers[conv_op.name]][1] = c_ratio
    else:
      idxs, W2, c_ratio = self.prune_kernel(conv_op, c_ratio)
      with self._model.g.as_default():
        if self._model.is_W1_prunable(conv_op):
          father_conv = self._model.g.get_operation_by_name(self._model.fathers[conv_op.name])
          while father_conv.type in ['DepthwiseConv2dNative']:
            if self._model.is_W1_prunable(father_conv):
              father_conv = \
                self._model.g.get_operation_by_name(self._model.fathers[father_conv.name])
          tf.logging.info('father conv {}'.format(father_conv.name))
          tf.logging.info('father conv input {}'.format(father_conv.inputs[0]))
          self.prune_W1(father_conv, idxs)

        self.prune_W2(conv_op, idxs, W2)

    tf.logging.info('Channel pruning the {} layer, \
      the pruning rate is {}'.format(conv_op.name, c_ratio))

    if self.finallayer():
      acc = self.accuracy()
      tf.logging.info('Pruning accuracy {}'.format(acc))
      pruned_flops = self.__compute_model_flops(fake=True)
      tf.logging.info('Pruned flops {}'.format(pruned_flops))

      preserve_ratio = pruned_flops / self.model_flops
      reward = [acc, pruned_flops]
      tf.logging.info(
        'The accuracy is {} and the flops after pruning is {}'.format(reward[0], reward[1]))
      tf.logging.info('The speedup ratio is {}'.format(preserve_ratio))
      tf.logging.info('The original model flops is {}'.format(self.model_flops))
      tf.logging.info('The pruned flops is {}'.format(pruned_flops))
      tf.logging.info('The max strategy dict is {}'.format(self.max_strategy_dict))

      state, reward = self.currentStates.loc[self.state].copy(), reward
      #if FLAGS.prune_option != 'auto':
      #  self.save_model()
      return state, reward, True, c_ratio

    reward = [0, 1]
    self.state += 1
    if FLAGS.cp_prune_option == 'auto':
      self.currentStates['maxreduce'][self.state] = self.max_reduced_flops / self.model_flops
    state = self.currentStates.loc[self.state].copy()

    return state, reward, False, c_ratio

  def save_model(self):
    """ save the current model to the `FLAGS.channel_pruned_path`"""
    with self._model.g.as_default():
      self.accuracy()
      self.sm_writer.add_graph(self._model.g)
      self.saver = tf.train.Saver()
      self.saver.save(self._model.sess, FLAGS.cp_channel_pruned_path)
      tf.logging.info('saved pruned model to {}'.format(FLAGS.cp_channel_pruned_path))
