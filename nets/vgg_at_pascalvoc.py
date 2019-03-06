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
"""Model helper for creating a VGG model for the Pascal VOC dataset."""

import os
import shutil
import numpy as np
import tensorflow as tf
from utils.multi_gpu_wrapper import MultiGpuWrapper as mgw

from nets.abstract_model_helper import AbstractModelHelper
from datasets.pascalvoc_dataset import PascalVocDataset
from utils.misc_utils import is_primary_worker

from utils.external.ssd_tensorflow.preprocessing.ssd_preprocessing import preprocess_image
from utils.external.ssd_tensorflow.net import ssd_net
from utils.external.ssd_tensorflow.utility import anchor_manipulator
from utils.external.ssd_tensorflow.utility import scaffolds
from utils.external.ssd_tensorflow.voc_eval import do_python_eval

FLAGS = tf.app.flags.FLAGS

# model related configuration
tf.app.flags.DEFINE_integer('nb_iters_train', 120000, 'The number of training iterations.')
tf.app.flags.DEFINE_float('negative_ratio', 3.0, 'Negative ratio in the loss function.')
tf.app.flags.DEFINE_float('match_threshold', 0.5, 'Matching threshold in the loss function.')
tf.app.flags.DEFINE_float('neg_threshold', 0.5,
                          'Matching threshold for the negtive examples in the loss function.')
tf.app.flags.DEFINE_float('select_threshold', 0.01,
                          'Class-specific confidence score threshold for selecting a box.')
tf.app.flags.DEFINE_float('min_size', 0.03, 'The min size of bboxes to keep.')
tf.app.flags.DEFINE_float('nms_threshold', 0.45, 'Matching threshold in NMS algorithm.')
tf.app.flags.DEFINE_integer('nms_topk', 200, 'Number of total object to keep after NMS.')
tf.app.flags.DEFINE_integer('keep_topk', 400,
                            'Number of total object to keep for each image before nms.')

# optimizer related configuration
tf.app.flags.DEFINE_float('lrn_rate_init', 1e-3, 'The initial learning rate.')
tf.app.flags.DEFINE_float('lrn_rate_min', 1e-6, 'The minimal learning rate')
tf.app.flags.DEFINE_string('lrn_rate_dcy_bnds', '500, 80000, 100000',
                           'Learning rate decay boundaries.')
tf.app.flags.DEFINE_string('lrn_rate_dcy_rates', '0.1, 1, 0.1, 0.01',
                           'Learning rate decay rates for each segment between boundaries')
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum coefficient')
tf.app.flags.DEFINE_integer('nb_iters_cls_wmup', 10000,
                            'The number of iterations for warming-up the classification loss')
tf.app.flags.DEFINE_float('loss_w_dcy', 5e-4, 'weight decaying loss\'s coefficient')

# checkpoint related configuration
tf.app.flags.DEFINE_string('backbone_ckpt_dir', './backbone_models/',
                           'The backbone model\'s (e.g. VGG-16) checkpoint directory')
tf.app.flags.DEFINE_string('backbone_model_scope', 'vgg_16',
                           'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string('model_scope', 'ssd300',
                           'Model scope name used to replace the name_scope in checkpoint.')
tf.app.flags.DEFINE_string('warm_start_excl_scopes',
                           'ssd300/multibox_head, ssd300/additional_layers, ssd300/conv4_3_scale',
                           'List of scopes to be excluded when restoring from a backbone model')
tf.app.flags.DEFINE_boolean('ignore_missing_vars', True,
                            'When restoring a checkpoint would ignore missing variables.')

# evaluation related configuration
tf.app.flags.DEFINE_string('outputs_dump_dir', './ssd_outputs/', 'outputs\'s dumping directory')

def parse_comma_list(args):
  """Convert a comma-separated list to a list of floating-point numbers."""

  return [float(s.strip()) for s in args.split(',')]

def setup_anchor_info():
  """Setup the anchor bounding boxes' information."""

  # get all anchor bounding boxes
  out_shape = [FLAGS.image_size] * 2
  anchor_creator = anchor_manipulator.AnchorCreator(
    out_shape,
    layers_shapes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
    anchor_scales = [(0.1,), (0.2,), (0.375,), (0.55,), (0.725,), (0.9,)],
    extra_anchor_scales = [(0.1414,), (0.2739,), (0.4541,), (0.6315,), (0.8078,), (0.9836,)],
    anchor_ratios = [(1., 2., .5), (1., 2., 3., .5, 0.3333), (1., 2., 3., .5, 0.3333),
                     (1., 2., 3., .5, 0.3333), (1., 2., .5), (1., 2., .5)],
    layer_steps = [8, 16, 32, 64, 100, 300])
  all_anchors, all_num_anchors_depth, all_num_anchors_spatial = anchor_creator.get_all_anchors()

  # construct the anchor bounding boxes' encoder & decoder
  num_anchors_per_layer = []
  for ind in range(len(all_anchors)):
    num_anchors_per_layer.append(all_num_anchors_depth[ind] * all_num_anchors_spatial[ind])
  anchor_encoder = anchor_manipulator.AnchorEncoder(
    allowed_borders=[1.0] * 6, positive_threshold=FLAGS.match_threshold,
    ignore_threshold=FLAGS.neg_threshold, prior_scaling=[0.1, 0.1, 0.2, 0.2])

  # pack all the information into one dictionary
  anchor_info = {
    'init_fn': lambda: anchor_encoder.init_all_anchors(
      all_anchors, all_num_anchors_depth, all_num_anchors_spatial),
    'encode_fn': lambda glabels_, gbboxes_: anchor_encoder.encode_all_anchors(
      glabels_, gbboxes_, all_anchors, all_num_anchors_depth, all_num_anchors_spatial),
    'decode_fn': lambda pred: anchor_encoder.decode_all_anchors(pred, num_anchors_per_layer),
    'num_anchors_per_layer': num_anchors_per_layer,
    'all_num_anchors_depth': all_num_anchors_depth,
  }

  return anchor_info

def modified_smooth_l1(
    bbox_pred, bbox_targets, bbox_inside_weights=1., bbox_outside_weights=1., sigma=1.):
  """Modified smooth L1-loss.

  Description:
  * ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
  * SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                  |x| - 0.5 / sigma^2,    otherwise
  """

  with tf.name_scope('smooth_l1', [bbox_pred, bbox_targets]):
    sigma2 = sigma * sigma
    inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))
    smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
    smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
    smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
    smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                              tf.multiply(smooth_l1_option2, tf.abs(smooth_l1_sign - 1.0)))
    outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

  return outside_mul

def select_bboxes(scores_pred, bboxes_pred, num_classes, select_threshold):
  selected_bboxes = {}
  selected_scores = {}
  with tf.name_scope('select_bboxes', [scores_pred, bboxes_pred]):
    for class_ind in range(1, num_classes):
      class_scores = scores_pred[:, class_ind]
      select_mask = class_scores > select_threshold
      select_mask = tf.cast(select_mask, tf.float32)
      selected_bboxes[class_ind] = tf.multiply(bboxes_pred, tf.expand_dims(select_mask, axis=-1))
      selected_scores[class_ind] = tf.multiply(class_scores, select_mask)

  return selected_bboxes, selected_scores

def clip_bboxes(ymin, xmin, ymax, xmax, name):
  with tf.name_scope(name, 'clip_bboxes', [ymin, xmin, ymax, xmax]):
    ymin = tf.maximum(ymin, 0.)
    xmin = tf.maximum(xmin, 0.)
    ymax = tf.minimum(ymax, 1.)
    xmax = tf.minimum(xmax, 1.)
    ymin = tf.minimum(ymin, ymax)
    xmin = tf.minimum(xmin, xmax)

  return ymin, xmin, ymax, xmax

def filter_bboxes(scores_pred, ymin, xmin, ymax, xmax, min_size, name):
  with tf.name_scope(name, 'filter_bboxes', [scores_pred, ymin, xmin, ymax, xmax]):
    width = xmax - xmin
    height = ymax - ymin
    filter_mask = tf.logical_and(width > min_size, height > min_size)
    filter_mask = tf.cast(filter_mask, tf.float32)

  return tf.multiply(ymin, filter_mask), tf.multiply(xmin, filter_mask), \
    tf.multiply(ymax, filter_mask), tf.multiply(xmax, filter_mask), \
    tf.multiply(scores_pred, filter_mask)

def sort_bboxes(scores_pred, ymin, xmin, ymax, xmax, keep_topk, name):
  with tf.name_scope(name, 'sort_bboxes', [scores_pred, ymin, xmin, ymax, xmax]):
    cur_bboxes = tf.shape(scores_pred)[0]
    scores, idxes = tf.nn.top_k(scores_pred, k=tf.minimum(keep_topk, cur_bboxes), sorted=True)
    ymin, xmin, ymax, xmax = \
      tf.gather(ymin, idxes), tf.gather(xmin, idxes), tf.gather(ymax, idxes), tf.gather(xmax, idxes)
    paddings_scores = \
      tf.expand_dims(tf.stack([0, tf.maximum(keep_topk-cur_bboxes, 0)], axis=0), axis=0)

  return tf.pad(ymin, paddings_scores, "CONSTANT"), tf.pad(xmin, paddings_scores, "CONSTANT"),\
    tf.pad(ymax, paddings_scores, "CONSTANT"), tf.pad(xmax, paddings_scores, "CONSTANT"),\
    tf.pad(scores, paddings_scores, "CONSTANT")

def nms_bboxes(scores_pred, bboxes_pred, nms_topk, nms_threshold, name):
  with tf.name_scope(name, 'nms_bboxes', [scores_pred, bboxes_pred]):
    idxes = tf.image.non_max_suppression(bboxes_pred, scores_pred, nms_topk, nms_threshold)

  return tf.gather(scores_pred, idxes), tf.gather(bboxes_pred, idxes)

def parse_by_class(cls_pred, bboxes_pred, num_classes,
                   select_threshold, min_size, keep_topk, nms_topk, nms_threshold):
  with tf.name_scope('select_bboxes', [cls_pred, bboxes_pred]):
    scores_pred = tf.nn.softmax(cls_pred)
    selected_bboxes, selected_scores = \
      select_bboxes(scores_pred, bboxes_pred, num_classes, select_threshold)
    for class_ind in range(1, num_classes):
      ymin, xmin, ymax, xmax = tf.unstack(selected_bboxes[class_ind], 4, axis=-1)
      ymin, xmin, ymax, xmax = \
        clip_bboxes(ymin, xmin, ymax, xmax, 'clip_bboxes_{}'.format(class_ind))
      ymin, xmin, ymax, xmax, selected_scores[class_ind] = filter_bboxes(
        selected_scores[class_ind], ymin, xmin, ymax, xmax,
        min_size, 'filter_bboxes_{}'.format(class_ind))
      ymin, xmin, ymax, xmax, selected_scores[class_ind] = sort_bboxes(
        selected_scores[class_ind], ymin, xmin, ymax, xmax,
        keep_topk, 'sort_bboxes_{}'.format(class_ind))
      selected_bboxes[class_ind] = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
      selected_scores[class_ind], selected_bboxes[class_ind] = nms_bboxes(
        selected_scores[class_ind], selected_bboxes[class_ind],
        nms_topk, nms_threshold, 'nms_bboxes_{}'.format(class_ind))

  return selected_bboxes, selected_scores

def forward_fn(inputs, is_train, data_format, anchor_info):
  """Forward pass function.

  Args:
  * inputs: input tensor to the network's forward pass
  * is_train: whether to use the forward pass with training operations inserted
  * data_format: data format ('channels_last' OR 'channels_first')
  * anchor_info: anchor bounding boxes' information

  Returns:
  * outputs: a dictionary of output tensors
  """

  tf.logging.info('building forward with is_train = {}'.format(is_train))

  # extract anchor boundiing boxes' information
  images = inputs['image']
  filenames = inputs['filename']
  shapes = inputs['shape']
  decode_fn = anchor_info['decode_fn']
  all_num_anchors_depth = anchor_info['all_num_anchors_depth']

  # initialize anchor bounding boxes
  anchor_info['init_fn']()

  # compute output tensors
  with tf.variable_scope(FLAGS.model_scope, values=[images], reuse=tf.AUTO_REUSE):
    # obtain the current model scope
    model_scope = tf.get_default_graph().get_name_scope()

    # obtain predictions for localization & classification
    backbone = ssd_net.VGG16Backbone(data_format)
    feature_layers = backbone.forward(images, training=is_train)
    loc_pred, cls_pred = ssd_net.multibox_head(
      feature_layers, FLAGS.nb_classes, all_num_anchors_depth, data_format=data_format)
    if data_format == 'channels_first':
      cls_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in cls_pred]
      loc_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in loc_pred]

    # flatten predictions
    def reshape_fn(preds, nb_dims):
      preds = [tf.reshape(pred, [tf.shape(images)[0], -1, nb_dims]) for pred in preds]
      preds = tf.concat(preds, axis=1)
      preds = tf.reshape(preds, [-1, nb_dims])
      return preds
    cls_pred = reshape_fn(cls_pred, FLAGS.nb_classes)
    loc_pred = reshape_fn(loc_pred, 4)

    # obtain per-class predictions on bounding boxes and scores
    if is_train:
      predictions = None#tf.no_op()
    else:
      bboxes_pred = decode_fn(loc_pred)  # evaluation batch size is 1
      bboxes_pred = tf.concat(bboxes_pred, axis=0)
      selected_bboxes, selected_scores = parse_by_class(
        cls_pred, bboxes_pred, FLAGS.nb_classes, FLAGS.select_threshold,
        FLAGS.min_size, FLAGS.keep_topk, FLAGS.nms_topk, FLAGS.nms_threshold)
      predictions = {'filename': filenames, 'shape': shapes}
      for idx_cls in range(1, FLAGS.nb_classes):
        predictions['scores_%d' % idx_cls] = tf.expand_dims(selected_scores[idx_cls], axis=0)
        predictions['bboxes_%d' % idx_cls] = tf.expand_dims(selected_bboxes[idx_cls], axis=0)

  # pack all the output tensors together
  outputs = {'cls_pred': cls_pred, 'loc_pred': loc_pred, 'predictions': predictions}

  return outputs, model_scope

def calc_loss_fn(objects, outputs, trainable_vars, anchor_info, batch_size):
  """Calculate the loss function's value.

  Args:
  * objects: one tensor with all the annotations packed together
  * outputs: a dictionary of output tensors
  * trainable_vars: list of trainable variables
  * anchor_info: anchor bounding boxes' information
  * batch_size: batch size

  Returns:
  * loss: loss function's value
  * metrics: dictionary of extra evaluation metrics
  """

  # extract output tensors
  #batch_size = FLAGS.batch_size
  cls_pred = outputs['cls_pred']
  loc_pred = outputs['loc_pred']

  # extract anchor bounding boxes' information
  encode_fn = anchor_info['encode_fn']
  decode_fn = anchor_info['decode_fn']
  num_anchors_per_layer = anchor_info['num_anchors_per_layer']
  all_num_anchors_depth = anchor_info['all_num_anchors_depth']

  # extract target values & predicted localization results
  def encode_objects_n_decode_loc_pred(objects_n_loc_pred):
    objects = objects_n_loc_pred[0]
    loc_pred = objects_n_loc_pred[1]
    flags, bboxes, labels = tf.split(objects, [1, 4, 1], axis=-1)
    flags = tf.squeeze(tf.cast(flags, dtype=tf.int64), axis=-1)
    labels = tf.squeeze(tf.cast(labels, dtype=tf.int64), axis=-1)
    index = tf.where(flags > 0)
    loc, cls, scr = encode_fn(tf.gather_nd(labels, index), tf.gather_nd(bboxes, index))
    bbox = decode_fn(loc_pred)
    return loc, cls, scr, bbox

  # post-forward operations
  with tf.control_dependencies([cls_pred, loc_pred]):
    with tf.name_scope('post_forward'):
      # obtain target values & localization predictions
      loc_targets, cls_targets, match_scores, bboxes_pred = tf.map_fn(
        encode_objects_n_decode_loc_pred,
        (tf.reshape(objects, [batch_size, -1, 6]), tf.reshape(loc_pred, [batch_size, -1, 4])),
        dtype=(tf.float32, tf.int64, tf.float32, [tf.float32] * len(num_anchors_per_layer)),
        back_prop=False)
      flatten_loc_targets = tf.reshape(loc_targets, [-1, 4])
      flatten_cls_targets = tf.reshape(cls_targets, [-1])
      flatten_match_scores = tf.reshape(match_scores, [-1])
      bboxes_pred = [tf.reshape(preds, [-1, 4]) for preds in bboxes_pred]
      bboxes_pred = tf.concat(bboxes_pred, axis=0)

      # each positive examples has one label
      positive_mask = flatten_cls_targets > 0
      n_positives = tf.count_nonzero(positive_mask)
      batch_n_positives = tf.count_nonzero(cls_targets, -1)
      batch_negtive_mask = tf.equal(cls_targets, 0)
      batch_n_negtives = tf.count_nonzero(batch_negtive_mask, -1)
      batch_n_neg_select = tf.cast(
        FLAGS.negative_ratio * tf.cast(batch_n_positives, tf.float32), tf.int32)
      batch_n_neg_select = tf.minimum(batch_n_neg_select, tf.cast(batch_n_negtives, tf.int32))

      # hard negative mining for classification
      predictions_for_bg = tf.nn.softmax(
        tf.reshape(cls_pred, [batch_size, -1, FLAGS.nb_classes]))[:, :, 0]
      prob_for_negtives = tf.where(batch_negtive_mask,
                                   0. - predictions_for_bg,
                                   0. - tf.ones_like(predictions_for_bg))
      topk_prob_for_bg, _ = tf.nn.top_k(prob_for_negtives, k=tf.shape(prob_for_negtives)[1])
      score_at_k = tf.gather_nd(topk_prob_for_bg,
                                tf.stack([tf.range(batch_size), batch_n_neg_select - 1], axis=-1))
      selected_neg_mask = prob_for_negtives >= tf.expand_dims(score_at_k, axis=-1)

      # include both selected negtive and all positive examples
      final_mask = tf.stop_gradient(tf.logical_or(
        tf.reshape(tf.logical_and(batch_negtive_mask, selected_neg_mask), [-1]), positive_mask))
      total_examples = tf.count_nonzero(final_mask)

      cls_pred = tf.boolean_mask(cls_pred, final_mask)
      loc_pred = tf.boolean_mask(loc_pred, tf.stop_gradient(positive_mask))
      flatten_cls_targets = tf.boolean_mask(
        tf.clip_by_value(flatten_cls_targets, 0, FLAGS.nb_classes), final_mask)
      flatten_loc_targets = tf.stop_gradient(tf.boolean_mask(flatten_loc_targets, positive_mask))

      # final predictions & classification accuracy
      predictions = {
        'classes': tf.argmax(cls_pred, axis=-1),
        'probabilities': tf.reduce_max(tf.nn.softmax(cls_pred, name='softmax_tensor'), axis=-1),
        'loc_predict': bboxes_pred,
      }
      accuracy = tf.reduce_mean(
        tf.cast(tf.equal(flatten_cls_targets, predictions['classes']), tf.float32))
      metrics = {'accuracy': accuracy}

  # cross-entropy loss
  ce_loss = (FLAGS.negative_ratio + 1.) * \
    tf.losses.sparse_softmax_cross_entropy(flatten_cls_targets, cls_pred)
  tf.identity(ce_loss, name='ce_loss')
  tf.summary.scalar('ce_loss', ce_loss)

  # localization loss
  loc_loss = tf.reduce_mean(
    tf.reduce_sum(modified_smooth_l1(loc_pred, flatten_loc_targets, sigma=1.), axis=-1))
  tf.identity(loc_loss, name='loc_loss')
  tf.summary.scalar('loc_loss', loc_loss)

  # L2-regularization loss
  l2_loss_list = []
  for var in trainable_vars:
    if '_bn' not in var.name:
      if 'conv4_3_scale' not in var.name:
        l2_loss_list.append(tf.nn.l2_loss(var))
      else:
        l2_loss_list.append(tf.nn.l2_loss(var) * 0.1)
  l2_loss = tf.add_n(l2_loss_list)
  tf.identity(l2_loss, name='l2_loss')
  tf.summary.scalar('l2_loss', l2_loss)

  # overall loss
  global_step = tf.train.get_or_create_global_step()
  loss_w_cls = tf.minimum(
    tf.cast(global_step, tf.float32) / tf.constant(FLAGS.nb_iters_cls_wmup, dtype=tf.float32), 1.0)
  loss = loss_w_cls * ce_loss + loc_loss + FLAGS.loss_w_dcy * l2_loss

  return loss, metrics

class ModelHelper(AbstractModelHelper):
  """Model helper for creating a VGG model for the VOC dataset."""

  def __init__(self, data_format='channels_last'):
    """Constructor function."""

    # class-independent initialization
    super(ModelHelper, self).__init__(data_format)

    # initialize training & evaluation subsets
    self.dataset_train = PascalVocDataset(preprocess_fn=preprocess_image, is_train=True)
    self.dataset_eval = PascalVocDataset(preprocess_fn=preprocess_image, is_train=False)

    # setup hyper-parameters & anchor information
    self.anchor_info = None  # track the most recently-used one
    self.batch_size = None  # track the most recently-used one
    self.model_scope = None

  def build_dataset_train(self, enbl_trn_val_split=False):
    """Build the data subset for training, usually with data augmentation."""

    return self.dataset_train.build()

  def build_dataset_eval(self):
    """Build the data subset for evaluation, usually without data augmentation."""

    return self.dataset_eval.build()

  def forward_train(self, inputs):
    """Forward computation at training."""

    anchor_info = setup_anchor_info()
    outputs, self.model_scope = forward_fn(inputs, True, self.data_format, anchor_info)
    self.anchor_info = anchor_info
    self.batch_size = tf.shape(inputs['image'])[0]
    self.trainable_vars = tf.get_collection(
      tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_scope)

    return outputs

  def forward_eval(self, inputs):
    """Forward computation at evaluation."""

    anchor_info = setup_anchor_info()
    outputs, __ = forward_fn(inputs, False, self.data_format, anchor_info)
    self.anchor_info = anchor_info
    self.batch_size = tf.shape(inputs['image'])[0]

    return outputs

  def calc_loss(self, objects, outputs, trainable_vars):
    """Calculate loss (and some extra evaluation metrics)."""

    return calc_loss_fn(objects, outputs, trainable_vars, self.anchor_info, self.batch_size)

  def setup_lrn_rate(self, global_step):
    """Setup the learning rate (and number of training iterations)."""

    bnds = [int(x) for x in parse_comma_list(FLAGS.lrn_rate_dcy_bnds)]
    vals = [FLAGS.lrn_rate_init * x for x in parse_comma_list(FLAGS.lrn_rate_dcy_rates)]
    lrn_rate = tf.train.piecewise_constant(global_step, bnds, vals)
    lrn_rate = tf.maximum(lrn_rate, tf.constant(FLAGS.lrn_rate_min, dtype=lrn_rate.dtype))
    nb_iters = FLAGS.nb_iters_train

    return lrn_rate, nb_iters

  def warm_start(self, sess):
    """Initialize the model for warm-start.

    Description:
    * We use a pre-trained ImageNet classification model to initialize the backbone part of the SSD
      model for feature extraction. If the SSD model's checkpoint files already exist, then the
      learner should restore model weights by itself.
    """

    # obtain a list of scopes to be excluded from initialization
    excl_scopes = []
    if FLAGS.warm_start_excl_scopes:
      excl_scopes = [scope.strip() for scope in FLAGS.warm_start_excl_scopes.split(',')]
    tf.logging.info('excluded scopes: {}'.format(excl_scopes))

    # obtain a list of variables to be initialized
    vars_list = []
    for var in self.trainable_vars:
      excluded = False
      for scope in excl_scopes:
        if scope in var.name:
          excluded = True
          break
      if not excluded:
        vars_list.append(var)

    # rename the variables' scope
    if FLAGS.backbone_model_scope is not None:
      backbone_model_scope = FLAGS.backbone_model_scope.strip()
      if backbone_model_scope == '':
        vars_list = {var.op.name.replace(self.model_scope + '/', ''): var for var in vars_list}
      else:
        vars_list = {var.op.name.replace(
          self.model_scope, backbone_model_scope): var for var in vars_list}

    # re-map the variables' names
    name_remap = {'/kernel': '/weights', '/bias': '/biases'}
    vars_list_remap = {}
    for var_name, var in vars_list.items():
      for name_old, name_new in name_remap.items():
        if name_old in var_name:
          var_name = var_name.replace(name_old, name_new)
          break
      vars_list_remap[var_name] = var
    vars_list = vars_list_remap

    # display all the variables to be initialized
    for var_name, var in vars_list.items():
      tf.logging.info('using %s to initialize %s' % (var_name, var.op.name))
    if not vars_list:
      raise ValueError('variables to be restored cannot be empty')

    # obtain the checkpoint files' path
    ckpt_path = tf.train.latest_checkpoint(FLAGS.backbone_ckpt_dir)
    tf.logging.info('restoring model weights from ' + ckpt_path)

    # remove missing variables from the list
    if FLAGS.ignore_missing_vars:
      reader = tf.train.NewCheckpointReader(ckpt_path)
      vars_list_avail = {}
      for var in vars_list:
        if reader.has_tensor(var):
          vars_list_avail[var] = vars_list[var]
        else:
          tf.logging.warning('variable %s not found in checkpoint files %s.' % (var, ckpt_path))
      vars_list = vars_list_avail
    if not vars_list:
      tf.logging.warning('no variables to restore.')
      return

    # restore variables from checkpoint files
    saver = tf.train.Saver(vars_list, reshape=False)
    saver.build()
    saver.restore(sess, ckpt_path)

  def dump_n_eval(self, outputs, action):
    """Dump the model's outputs to files and evaluate."""

    if not is_primary_worker('global'):
      return

    if action == 'init':
      if os.path.exists(FLAGS.outputs_dump_dir):
        shutil.rmtree(FLAGS.outputs_dump_dir)
      os.mkdir(FLAGS.outputs_dump_dir)
    elif action == 'dump':
      filename = outputs['predictions']['filename'][0].decode('utf8')[:-4]
      shape = outputs['predictions']['shape'][0]
      for idx_cls in range(1, FLAGS.nb_classes):
        with open(os.path.join(FLAGS.outputs_dump_dir, 'results_%d.txt' % idx_cls), 'a') as o_file:
          scores = outputs['predictions']['scores_%d' % idx_cls][0]
          bboxes = outputs['predictions']['bboxes_%d' % idx_cls][0]
          bboxes[:, 0] = (bboxes[:, 0] * shape[0]).astype(np.int32, copy=False) + 1
          bboxes[:, 1] = (bboxes[:, 1] * shape[1]).astype(np.int32, copy=False) + 1
          bboxes[:, 2] = (bboxes[:, 2] * shape[0]).astype(np.int32, copy=False) + 1
          bboxes[:, 3] = (bboxes[:, 3] * shape[1]).astype(np.int32, copy=False) + 1
          for idx_bbox in range(bboxes.shape[0]):
            bbox = bboxes[idx_bbox][:]
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
              o_file.write('%s %.3f %.1f %.1f %.1f %.1f\n'
                           % (filename, scores[idx_bbox], bbox[1], bbox[0], bbox[3], bbox[2]))
    elif action == 'eval':
      do_python_eval(os.path.join(self.dataset_eval.data_dir, 'test'), FLAGS.outputs_dump_dir)
    else:
      raise ValueError('unrecognized action in dump_n_eval(): ' + action)

  @property
  def model_name(self):
    """Model's name."""

    return 'ssd_vgg_300'

  @property
  def dataset_name(self):
    """Dataset's name."""

    return 'pascalvoc'
