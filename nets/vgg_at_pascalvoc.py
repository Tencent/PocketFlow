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

import tensorflow as tf
from utils.multi_gpu_wrapper import MultiGpuWrapper as mgw

from nets.abstract_model_helper import AbstractModelHelper
from datasets.pascalvoc_dataset import PascalVocDataset

from utils.external.ssd_tensorflow.net import ssd_net
from utils.external.ssd_tensorflow.utility import anchor_manipulator
from utils.external.ssd_tensorflow.utility import scaffolds

FLAGS = tf.app.flags.FLAGS

### REQUIRED ###
#tf.app.flags.DEFINE_integer('resnet_size', 20, '# of layers in the ResNet model')
#tf.app.flags.DEFINE_float('nb_epochs_rat', 1.0, '# of training epochs\'s ratio')
#tf.app.flags.DEFINE_float('lrn_rate_init', 1e-1, 'initial learning rate')
#tf.app.flags.DEFINE_float('batch_size_norm', 128, 'normalization factor of batch size')
#tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum coefficient')
#tf.app.flags.DEFINE_float('loss_w_dcy', 2e-4, 'weight decaying loss\'s coefficient')
### REQUIRED ###

# hardware related configuration
tf.app.flags.DEFINE_integer('num_readers', 8,
                            'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer('num_preprocessing_threads', 24,
                            'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer('num_cpu_threads', 0, 'The number of cpu cores used to train.')
tf.app.flags.DEFINE_float('gpu_memory_fraction', 1.0, 'GPU memory fraction to use.')

# scaffold related configuration
tf.app.flags.DEFINE_string('data_dir', './tfrecords',
                           'The directory where the dataset input data is stored.')
tf.app.flags.DEFINE_integer('num_classes', 21, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string('model_dir', './logs/', 'The directory where the model will be stored.')
tf.app.flags.DEFINE_integer('log_every_n_steps', 10, 'The frequency with which logs are printed.')
tf.app.flags.DEFINE_integer('save_summary_steps', 500,
                            'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer('save_checkpoints_secs', 7200,
                            'The frequency with which the model is saved, in seconds.')

# model related configuration
tf.app.flags.DEFINE_integer('train_image_size', 300,
                            'The size of the input image for the model to use.')
tf.app.flags.DEFINE_integer('train_epochs', None, 'The number of epochs to use for training.')
tf.app.flags.DEFINE_integer('max_number_of_steps', 120000,
                            'The max number of steps to use for training.')
tf.app.flags.DEFINE_string('data_format', 'channels_last', # 'channels_first' or 'channels_last'
                           'A flag to override the data format used in the model. channels_first '
                           'provides a performance boost on GPU but is not always compatible '
                           'with CPU. If left unspecified, the data format will be chosen '
                           'automatically based on whether TensorFlow was built for CPU or GPU.')
tf.app.flags.DEFINE_float('negative_ratio', 3.0, 'Negative ratio in the loss function.')
tf.app.flags.DEFINE_float('match_threshold', 0.5, 'Matching threshold in the loss function.')
tf.app.flags.DEFINE_float('neg_threshold', 0.5,
                          'Matching threshold for the negtive examples in the loss function.')

# optimizer related configuration
tf.app.flags.DEFINE_integer('tf_random_seed', 20190101, 'Random seed for TensorFlow initializers.')
tf.app.flags.DEFINE_float('weight_decay', 5e-4, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_float('momentum', 0.9,
                          'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')
tf.app.flags.DEFINE_float('end_learning_rate', 1e-6,
                          'The minimal end learning rate used by a polynomial decay learning rate.')

# for learning rate piecewise_constant decay
tf.app.flags.DEFINE_string('decay_boundaries', '500, 80000, 100000',
                           'Learning rate decay boundaries by global_step (comma-separated list).')
tf.app.flags.DEFINE_string('lr_decay_factors', '0.1, 1, 0.1, 0.01',
                           'The values of learning_rate decay factor for each segment between '
                           'boundaries (comma-separated list).')

# checkpoint related configuration
tf.app.flags.DEFINE_string('checkpoint_path', './model/',
                           'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string('checkpoint_model_scope', 'vgg_16',
                           'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string('model_scope', 'ssd300',
                           'Model scope name used to replace the name_scope in checkpoint.')
tf.app.flags.DEFINE_string('checkpoint_exclude_scopes',
                           'ssd300/multibox_head, ssd300/additional_layers, ssd300/conv4_3_scale',
                           'Comma-separated list of scopes of variables to exclude when restoring '
                           'from a checkpoint.')
tf.app.flags.DEFINE_boolean('ignore_missing_vars', True,
                            'When restoring a checkpoint would ignore missing variables.')
tf.app.flags.DEFINE_boolean('multi_gpu', False, 'Whether there is GPU to use for training.')

params = {}

def parse_comma_list(args):
  """Convert a comma-separated list to a list of floating-point numbers."""

  return [float(s.strip()) for s in args.split(',')]

def setup_params():
  """Setup hyper-parameters (from FLAGS to dict)."""

  global params
  params = {
    'num_gpus': 1,
    'max_number_of_steps': FLAGS.max_number_of_steps,
    'train_image_size': FLAGS.train_image_size,
    'data_format': FLAGS.data_format,
    'batch_size': FLAGS.batch_size,
    'model_scope': FLAGS.model_scope,
    'num_classes': FLAGS.num_classes,
    'negative_ratio': FLAGS.negative_ratio,
    'match_threshold': FLAGS.match_threshold,
    'neg_threshold': FLAGS.neg_threshold,
    'weight_decay': FLAGS.weight_decay,
    'momentum': FLAGS.momentum,
    'learning_rate': FLAGS.learning_rate,
    'end_learning_rate': FLAGS.end_learning_rate,
    'decay_boundaries': parse_comma_list(FLAGS.decay_boundaries),
    'lr_decay_factors': parse_comma_list(FLAGS.lr_decay_factors)
  }

def __setup_anchor_info():
  """Setup the anchor bounding boxes' information."""

  # get all anchor bounding boxes
  out_shape = [params['train_image_size']] * 2
  anchor_creator = anchor_manipulator.AnchorCreator(
    out_shape,
    layers_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
    anchor_scales=[(0.1,), (0.2,), (0.375,), (0.55,), (0.725,), (0.9,)],
    extra_anchor_scales=[(0.1414,), (0.2739,), (0.4541,), (0.6315,), (0.8078,), (0.9836,)],
    anchor_ratios=[(1., 2., .5), (1., 2., 3., .5, 0.3333), (1., 2., 3., .5, 0.3333),
                   (1., 2., 3., .5, 0.3333), (1., 2., .5), (1., 2., .5)],
    layer_steps=[8, 16, 32, 64, 100, 300])
  all_anchors, all_num_anchors_depth, all_num_anchors_spatial = anchor_creator.get_all_anchors()

  # construct the anchor bounding boxes' encoder & decoder
  num_anchors_per_layer = []
  for ind in range(len(all_anchors)):
    num_anchors_per_layer.append(all_num_anchors_depth[ind] * all_num_anchors_spatial[ind])
  anchor_encoder = anchor_manipulator.AnchorEncoder(
    allowed_borders=[1.0] * 6, positive_threshold=params['match_threshold'],
    ignore_threshold=params['neg_threshold'], prior_scaling=[0.1, 0.1, 0.2, 0.2])

  # pack all the information into one dictionary
  anchor_info = {
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

  all_num_anchors_depth = anchor_info['all_num_anchors_depth']
  with tf.variable_scope(params['model_scope'], values=[inputs], reuse=tf.AUTO_REUSE):
    # obtain predictions for localization & classification
    backbone = ssd_net.VGG16Backbone(data_format)
    feature_layers = backbone.forward(inputs, training=is_train)
    loc_pred, cls_pred = ssd_net.multibox_head(
      feature_layers, params['num_classes'], all_num_anchors_depth, data_format=data_format)
    if data_format == 'channels_first':
      cls_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in cls_pred]
      loc_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in loc_pred]

    # flatten predictions
    def reshape_fn(preds, nb_dims):
      preds = [tf.reshape(pred, [tf.shape(inputs)[0], -1, nb_dims]) for pred in preds]
      preds = tf.concat(preds, axis=1)
      preds = tf.reshape(preds, [-1, nb_dims])
      return preds
    cls_pred = reshape_fn(cls_pred, params['num_classes'])
    loc_pred = reshape_fn(loc_pred, 4)

  # pack all the output tensors together
  outputs = {'cls_pred': cls_pred, 'loc_pred': loc_pred}

  return outputs

def calc_loss_fn(objects, outputs, trainable_vars, anchor_info):
  """Calculate the loss function's value.

  Args:
  * objects: one tensor with all the annotations packed together
  * outputs: a dictionary of output tensors
  * trainable_vars: list of trainable variables
  * anchor_info: anchor bounding boxes' information

  Returns:
  * loss: loss function's value
  * metrics: dictionary of extra evaluation metrics
  """

  # extract output tensors
  batch_size = params['batch_size']
  cls_pred = outputs['cls_pred']
  loc_pred = outputs['loc_pred']

  # extract anchor bounding boxes' information
  encode_fn = anchor_info['encode_fn']
  decode_fn = anchor_info['decode_fn']
  num_anchors_per_layer = anchor_info['num_anchors_per_layer']
  all_num_anchors_depth = anchor_info['all_num_anchors_depth']

  # extract target localization & classification results from <objects>
  # TODO use tf.map_fn
  gt_locations_list = []
  gt_labels_list = []
  gt_scores_list = []
  b_flags, b_bboxes, b_labels = tf.split(objects, [1, 4, 1], -1)
  b_flags = tf.squeeze(tf.cast(b_flags, dtype=tf.int64), axis=-1)
  b_labels = tf.squeeze(tf.cast(b_labels, dtype=tf.int64), axis=-1)
  for batch_index in range(batch_size):
    index = tf.where(b_flags[batch_index] > 0)
    labels = tf.gather_nd(b_labels[batch_index], index)
    bboxes = tf.gather_nd(b_bboxes[batch_index], index)
    gt_locations, gt_labels, gt_scores = encode_fn(labels, bboxes)
    gt_locations_list += [tf.expand_dims(gt_locations, 0)]
    gt_labels_list += [tf.expand_dims(gt_labels, 0)]
    gt_scores_list += [tf.expand_dims(gt_scores, 0)]
  loc_targets = tf.concat(gt_locations_list, axis=0)
  cls_targets = tf.concat(gt_labels_list, axis=0)
  match_scores = tf.concat(gt_scores_list, axis=0)

  # post-forward operations
  with tf.control_dependencies([cls_pred, loc_pred]):
    with tf.name_scope('post_forward'):
      bboxes_pred = tf.map_fn(lambda _preds: decode_fn(_preds),
                              tf.reshape(loc_pred, [batch_size, -1, 4]),
                              dtype=[tf.float32] * len(num_anchors_per_layer), back_prop=False)
      bboxes_pred = [tf.reshape(preds, [-1, 4]) for preds in bboxes_pred]
      bboxes_pred = tf.concat(bboxes_pred, axis=0)

      flatten_loc_targets = tf.reshape(loc_targets, [-1, 4])
      flatten_cls_targets = tf.reshape(cls_targets, [-1])
      flatten_match_scores = tf.reshape(match_scores, [-1])

      # each positive examples has one label
      positive_mask = flatten_cls_targets > 0
      n_positives = tf.count_nonzero(positive_mask)
      batch_n_positives = tf.count_nonzero(cls_targets, -1)
      batch_negtive_mask = tf.equal(cls_targets, 0)
      batch_n_negtives = tf.count_nonzero(batch_negtive_mask, -1)
      batch_n_neg_select = tf.cast(
        params['negative_ratio'] * tf.cast(batch_n_positives, tf.float32), tf.int32)
      batch_n_neg_select = tf.minimum(batch_n_neg_select, tf.cast(batch_n_negtives, tf.int32))

      # hard negative mining for classification
      predictions_for_bg = tf.nn.softmax(
        tf.reshape(cls_pred, [batch_size, -1, params['num_classes']]))[:, :, 0]
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
        tf.clip_by_value(flatten_cls_targets, 0, params['num_classes']), final_mask)
      flatten_loc_targets = tf.stop_gradient(tf.boolean_mask(flatten_loc_targets, positive_mask))

      # final predictions & classification accuracy
      predictions = {
        'classes': tf.argmax(cls_pred, axis=-1),
        'probabilities': tf.reduce_max(tf.nn.softmax(cls_pred, name='softmax_tensor'), axis=-1),
        'loc_predict': bboxes_pred,
      }
      cls_accuracy = tf.metrics.accuracy(flatten_cls_targets, predictions['classes'])
      tf.identity(cls_accuracy[1], name='cls_accuracy')
      tf.summary.scalar('cls_accuracy', cls_accuracy[1])
      metrics = {'accuracy': cls_accuracy[1])

  # cross-entropy loss
  ce_loss = (params['negative_ratio'] + 1.) * \
    tf.losses.sparse_softmax_cross_entropy(flatten_cls_targets, cls_pred)
  tf.identity(ce_loss, name='cross_entropy_loss')
  tf.summary.scalar('cross_entropy_loss', ce_loss)

  # localization loss
  loc_loss = tf.reduce_mean(
    tf.reduce_sum(modified_smooth_l1(loc_pred, flatten_loc_targets, sigma=1.), axis=-1))
  tf.identity(loc_loss, name='localization_loss')
  tf.summary.scalar('localization_loss', loc_loss)

  # L2-regularization loss
  l2_loss_list = []
  for var in trainable_vars:
    if '_bn' not in var.name:
      if 'conv4_3_scale' not in var.name:
        l2_loss_list.append(tf.nn.l2_loss(var))
      else:
        l2_loss_list.append(tf.nn.l2_loss(var) * 0.1)
  l2_loss = tf.add_n(l2_loss_list)
  tf.identity(loc_loss, name='localization_loss')
  tf.summary.scalar('localization_loss', loc_loss)

  # overall loss
  loss = ce_loss + loc_loss + params['weight_decay'] * tf.add_n(l2_loss)

  return loss, metrics

class ModelHelper(AbstractModelHelper):
  """Model helper for creating a VGG model for the VOC dataset."""

  def __init__(self):
    """Constructor function."""

    # class-independent initialization
    super(ModelHelper, self).__init__()

    # initialize training & evaluation subsets
    self.dataset_train = PascalVocDataset(is_train=True)
    self.dataset_eval = PascalVocDataset(is_train=False)

    # setup hyper-parameters & anchor information
    setup_params()
    self.anchor_info = setup_anchor_info()

  def build_dataset_train(self, enbl_trn_val_split=False):
    """Build the data subset for training, usually with data augmentation."""

    return self.dataset_train.build()

  def build_dataset_eval(self):
    """Build the data subset for evaluation, usually without data augmentation."""

    return self.dataset_eval.build()

  def forward_train(self, inputs, data_format='channels_last'):
    """Forward computation at training."""

    return forward_fn(inputs, True, data_format, self.anchor_info)

  def forward_eval(self, inputs, data_format='channels_last'):
    """Forward computation at evaluation."""

    return forward_fn(inputs, False, data_format, self.anchor_info)

  def calc_loss(self, objects, outputs, trainable_vars):
    """Calculate loss (and some extra evaluation metrics)."""

    return calc_loss_fn(objects, outputs, trainable_vars, self.anchor_info)

  def setup_lrn_rate(self, global_step):
    """Setup the learning rate (and number of training iterations)."""

    bnds = [int(x) for x in params['decay_boundaries']]
    vals = [params['learning_rate'] * x for x in params['lr_decay_factors']]
    lrn_rate = tf.train.piecewise_constant(global_step, bnds, vals)
    lrn_rate = tf.maximum(lrn_rate, tf.constant(params['end_learning_rate'], dtype=lrn_rate.dtype))
    nb_iters = params['max_number_of_steps']

    return lrn_rate, nb_iters

  def warm_start(self, sess):
    """Initialize the model for warm-start.

    Description:
    * We use a pre-trained ImageNet classification model to initialize the backbone part of the SSD
      model for feature extraction. If the SSD model's checkpoint files already exist, then skip.
    """

    # early return if checkpoint files already exist
    if tf.train.latest_checkpoint(FLAGS.model_dir):
      tf.logging.info('checkpoint files already exist in ' + FLAGS.model_dir)
      return

    # obtain a list of scopes to be excluded from initialization
    excluded_scopes = []
    if FLAGS.checkpoint_exclude_scopes:
      excluded_scopes = [scope.strip() for scope in FLAGS.checkpoint_exclude_scopes.split(',')]
    tf.logging.info('excluded scopes: {}'.format(excluded_scopes))

    # obtain a list of variables to be initialized
    vars_list = []
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
      excluded = False
      for scope in excluded_scopes:
        if var.name.startswith(scope):
          excluded = True
          break
      if not excluded:
        vars_list.append(var)

    # rename variables to be initialized
    if FLAGS.checkpoint_model_scope is not None:
      # rename the variable scope
      if FLAGS.checkpoint_model_scope.strip() == '':
        vars_list = {var.op.name.replace(FLAGS.model_scope + '/', ''): var for var in vars_list}
      else:
        vars_list = {var.op.name.replace(
          FLAGS.model_scope, FLAGS.checkpoint_model_scope.strip()): var for var in vars_list}

      # re-map the variable's name
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
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      ckpt_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      ckpt_path = FLAGS.checkpoint_path
    tf.logging.info('restoring model weights from ' + ckpt_path)

    # remove missing variables from the list
    if FLAGS.ignore_missing_vars:
      reader = tf.train.NewCheckpointReader(ckpt_path)
      vars_list_avail = {}
      for var in var_list:
        if reader.has_tensor(var):
          vars_list_avail[var] = var_list[var]
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

  @property
  def model_name(self):
    """Model's name."""

    return 'ssd_vgg_300'

  @property
  def dataset_name(self):
    """Dataset's name."""

    return 'pascalvoc'