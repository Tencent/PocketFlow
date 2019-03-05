import os
import shutil
import numpy as np
import tensorflow as tf



from nets.abstract_model_helper import AbstractModelHelper
from datasets.pascalvoc_dataset import PascalVocDataset
from utils.misc_utils import is_primary_worker

import tensorflow.contrib.slim as slim

from utils.external.faster_rcnn_tensorflow.preprocessing.faster_rcnn_preprocessing import preprocess_image

from utils.external.faster_rcnn_tensorflow.net import resnet_faster_rcnn as resnet
from utils.external.faster_rcnn_tensorflow.net import mobilenet_v2_faster_rcnn as mobilenet_v2

from utils.external.faster_rcnn_tensorflow.utility import anchor_utils, encode_and_decode, boxes_utils
from utils.external.faster_rcnn_tensorflow.configs import cfgs
from utils.external.faster_rcnn_tensorflow.utility import loss_utils as losses
from utils.external.faster_rcnn_tensorflow.utility import show_box_in_tensor

from utils.external.faster_rcnn_tensorflow.utility.proposal_opr import postprocess_rpn_proposals
from utils.external.faster_rcnn_tensorflow.utility.anchor_target_layer_without_boxweight import anchor_target_layer
from utils.external.faster_rcnn_tensorflow.utility.proposal_target_layer import proposal_target_layer

from utils.external.ssd_tensorflow.voc_eval import do_python_eval

# model related configuration
tf.app.flags.DEFINE_integer('nb_iters_train', 200000, 'The number of training iterations.')
tf.app.flags.DEFINE_float('momentum', 0.9, 'momentum coefficient')
# evaluation related configuration
tf.app.flags.DEFINE_string('outputs_dump_dir', './f_rcnn_outputs/', 'outputs\'s dumping directory')
# checkpoint related configuration
tf.app.flags.DEFINE_string('backbone_ckpt_dir', './backbone_models/',
                           'The backbone model\'s (e.g. VGG-16) checkpoint directory')
FLAGS = tf.app.flags.FLAGS

def build_base_network(inputs, is_train):
  if cfgs.NET_NAME.startswith('resnet_v1'):
    return resnet.resnet_base(inputs, scope_name=cfgs.NET_NAME, is_training=is_train)
  elif cfgs.NET_NAME.startswith('MobilenetV2'):
    return mobilenet_v2.mobilenetv2_base(inputs, is_training=is_train)
  else:
    raise ValueError('Sry, we only support resnet or mobilenet_v2')

def build_fastrcnn(is_train, feature_to_cropped, rois, img_shape):
  with tf.variable_scope('Fast-RCNN'):
    # 5. ROI Pooling
    with tf.variable_scope('rois_pooling'):
      pooled_features = roi_pooling(feature_maps=feature_to_cropped, rois=rois, img_shape=img_shape)

    # 6. inferecne rois in Fast-RCNN to obtain fc_flatten features
    if cfgs.NET_NAME.startswith('resnet'):
      fc_flatten = resnet.restnet_head(input=pooled_features,
                                        is_training=is_train,
                                        scope_name=cfgs.NET_NAME)
    elif cfgs.NET_NAME.startswith('Mobile'):
      fc_flatten = mobilenet_v2.mobilenetv2_head(inputs=pooled_features,
                                                   is_training=is_train)
    else:
      raise NotImplementedError('only support resnet and mobilenet')

      # 7. cls and reg in Fast-RCNN
      # tf.variance_scaling_initializer()
      # tf.VarianceScaling()
    with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):
      cls_score = slim.fully_connected(fc_flatten,
                                       num_outputs=FLAGS.nb_classes,
                                       weights_initializer=slim.variance_scaling_initializer(factor=1.0,
                                                                                             mode='FAN_AVG',
                                                                                             uniform=True),
                                       activation_fn=None, trainable=is_train,
                                       scope='cls_fc')

      bbox_pred = slim.fully_connected(fc_flatten,
                                       num_outputs=(FLAGS.nb_classes) * 4,
                                       weights_initializer=slim.variance_scaling_initializer(factor=1.0,
                                                                                             mode='FAN_AVG',
                                                                                             uniform=True),
                                       activation_fn=None, trainable=is_train,
                                       scope='reg_fc')
      # for convient. It also produce (cls_num +1) bboxes

      cls_score = tf.reshape(cls_score, [-1, FLAGS.nb_classes])
      bbox_pred = tf.reshape(bbox_pred, [-1, 4 * (FLAGS.nb_classes)])

  return bbox_pred, cls_score

def postprocess_fastrcnn(is_train, rois, bbox_ppred, scores, img_shape):
  """
  :param rois:[-1, 4]
  :param bbox_ppred: [-1, (cfgs.Class_num+1) * 4]
  :param scores: [-1, FLAGS.nb_classes]
  :return:
  """

  with tf.name_scope('postprocess_fastrcnn'):
    rois = tf.stop_gradient(rois)
    scores = tf.stop_gradient(scores)
    bbox_ppred = tf.reshape(bbox_ppred, [-1, FLAGS.nb_classes, 4])
    bbox_ppred = tf.stop_gradient(bbox_ppred)

    bbox_pred_list = tf.unstack(bbox_ppred, axis=1)
    score_list = tf.unstack(scores, axis=1)

    allclasses_boxes = []
    allclasses_scores = []
    categories = []
    for i in range(1, cfgs.CLASS_NUM+1):
      # 1. decode boxes in each class
      tmp_encoded_box = bbox_pred_list[i]
      tmp_score = score_list[i]
      tmp_decoded_boxes = encode_and_decode.decode_boxes(encoded_boxes=tmp_encoded_box,
                                                         reference_boxes=rois,
                                                         scale_factors=cfgs.ROI_SCALE_FACTORS)
      # tmp_decoded_boxes = encode_and_decode.decode_boxes(boxes=rois,
      #                                                    deltas=tmp_encoded_box,
      #                                                    scale_factor=cfgs.ROI_SCALE_FACTORS)

      # 2. clip to img boundaries
      tmp_decoded_boxes = boxes_utils.clip_boxes_to_img_boundaries(decode_boxes=tmp_decoded_boxes,
                                                                   img_shape=img_shape)

      # 3. NMS
      keep = tf.image.non_max_suppression(
          boxes=tmp_decoded_boxes,
          scores=tmp_score,
          max_output_size=cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS,
          iou_threshold=cfgs.FAST_RCNN_NMS_IOU_THRESHOLD)

      perclass_boxes = tf.gather(tmp_decoded_boxes, keep)
      perclass_scores = tf.gather(tmp_score, keep)

      allclasses_boxes.append(perclass_boxes)
      allclasses_scores.append(perclass_scores)
      categories.append(tf.ones_like(perclass_scores) * i)

    final_boxes = tf.concat(allclasses_boxes, axis=0)
    final_scores = tf.concat(allclasses_scores, axis=0)
    final_category = tf.concat(categories, axis=0)

    if is_train:
      """
      in training. We should show the detecitons in the tensorboard. So we add this.
      """
      kept_indices = tf.reshape(tf.where(tf.greater_equal(final_scores, cfgs.SHOW_SCORE_THRSHOLD)), [-1])

      final_boxes = tf.gather(final_boxes, kept_indices)
      final_scores = tf.gather(final_scores, kept_indices)
      final_category = tf.gather(final_category, kept_indices)

  return final_boxes, final_scores, final_category

def roi_pooling(feature_maps, rois, img_shape):
  '''
  Here use roi warping as roi_pooling
  :param featuremaps_dict: feature map to crop
  :param rois: shape is [-1, 4]. [x1, y1, x2, y2]
  :return:
  '''
  with tf.variable_scope('ROI_Warping'):
    img_h, img_w = tf.cast(img_shape[1], tf.float32), tf.cast(img_shape[2], tf.float32)
    N = tf.shape(rois)[0]
    x1, y1, x2, y2 = tf.unstack(rois, axis=1)

    normalized_x1 = x1 / img_w
    normalized_x2 = x2 / img_w
    normalized_y1 = y1 / img_h
    normalized_y2 = y2 / img_h

    normalized_rois = tf.transpose(
        tf.stack([normalized_y1, normalized_x1, normalized_y2, normalized_x2]), name='get_normalized_rois')

    normalized_rois = tf.stop_gradient(normalized_rois)

    cropped_roi_features = tf.image.crop_and_resize(feature_maps, normalized_rois,
                                                    box_ind=tf.zeros(shape=[N, ],
                                                                     dtype=tf.int32),
                                                    crop_size=[cfgs.ROI_SIZE, cfgs.ROI_SIZE],
                                                    name='CROP_AND_RESIZE'
                                                    )
    roi_features = slim.max_pool2d(cropped_roi_features,
                                  [cfgs.ROI_POOL_KERNEL_SIZE, cfgs.ROI_POOL_KERNEL_SIZE],
                                  stride=cfgs.ROI_POOL_KERNEL_SIZE)

  return roi_features

def add_roi_batch_img_smry(img, rois, labels):
  positive_roi_indices = tf.reshape(tf.where(tf.greater_equal(labels, 1)), [-1])
  negative_roi_indices = tf.reshape(tf.where(tf.equal(labels, 0)), [-1])

  pos_roi = tf.gather(rois, positive_roi_indices)
  neg_roi = tf.gather(rois, negative_roi_indices)

  pos_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                  boxes=pos_roi)
  neg_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                  boxes=neg_roi)
  tf.summary.image('pos_rois', pos_in_img)
  tf.summary.image('neg_rois', neg_in_img)

def add_anchor_img_smry(img, anchors, labels):
  positive_anchor_indices = tf.reshape(tf.where(tf.greater_equal(labels, 1)), [-1])
  negative_anchor_indices = tf.reshape(tf.where(tf.equal(labels, 0)), [-1])

  positive_anchor = tf.gather(anchors, positive_anchor_indices)
  negative_anchor = tf.gather(anchors, negative_anchor_indices)

  pos_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                  boxes=positive_anchor)
  neg_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                  boxes=negative_anchor)
  tf.summary.image('positive_anchor', pos_in_img)
  tf.summary.image('negative_anchors', neg_in_img)

def forward_fn(inputs_dict,is_train):
  """Forward pass function.

    Args:
    * inputs: input tensor to the network's forward pass
    * is_train: whether to use the forward pass with training operations inserted
    * data_format: data format ('channels_last' OR 'channels_first')
    * anchor_info: anchor bounding boxes' information

    Returns:
    * outputs: a dictionary of output tensors
    """
  inputs = inputs_dict['inputs']
  objects = inputs_dict['objects']

  images = inputs['image']
  filenames = inputs['filename']
  shapes = inputs['shape']

  if is_train:
    flags, gtboxes_batch = tf.split(objects, [1, 5], axis=-1)
    flags = tf.squeeze(tf.cast(flags, dtype=tf.int32), axis=-1)
    index = tf.where(flags > 0)
    gtboxes_batch = tf.gather_nd(gtboxes_batch, index)

  with slim.arg_scope(
      [slim.conv2d, slim.conv2d_in_plane, slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
      weights_regularizer=tf.contrib.layers.l2_regularizer(cfgs.WEIGHT_DECAY),
      biases_regularizer=tf.no_regularizer,
      biases_initializer=tf.constant_initializer(0.0)):
    img_shape = tf.shape(images)
    # 1. build base network
    feature_to_cropped = build_base_network(images, is_train)
    # 2. build rpn
    with tf.variable_scope('build_rpn',
                           regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):
      rpn_conv3x3 = slim.conv2d(
        feature_to_cropped, 512, [3, 3],
        trainable=is_train, weights_initializer=cfgs.INITIALIZER,
        activation_fn=tf.nn.relu,
        scope='rpn_conv/3x3')
      num_anchors_per_location = len(cfgs.ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS)
      rpn_cls_score = slim.conv2d(rpn_conv3x3, num_anchors_per_location * 2, [1, 1], stride=1,
                                  trainable=is_train, weights_initializer=cfgs.INITIALIZER,
                                  activation_fn=None,
                                  scope='rpn_cls_score')
      rpn_box_pred = slim.conv2d(rpn_conv3x3, num_anchors_per_location * 4, [1, 1], stride=1,
                                 trainable=is_train, weights_initializer=cfgs.BBOX_INITIALIZER,
                                 activation_fn=None,
                                 scope='rpn_bbox_pred')
      rpn_box_pred = tf.reshape(rpn_box_pred, [-1, 4])
      rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])
      rpn_cls_prob = slim.softmax(rpn_cls_score, scope='rpn_cls_prob')

    # 3. generate_anchors
    featuremap_height, featuremap_width = tf.shape(feature_to_cropped)[1], tf.shape(feature_to_cropped)[2]
    featuremap_height = tf.cast(featuremap_height, tf.float32)
    featuremap_width = tf.cast(featuremap_width, tf.float32)

    anchors = anchor_utils.make_anchors(base_anchor_size=cfgs.BASE_ANCHOR_SIZE_LIST[0],
                                        anchor_scales=cfgs.ANCHOR_SCALES, anchor_ratios=cfgs.ANCHOR_RATIOS,
                                        featuremap_height=featuremap_height,
                                        featuremap_width=featuremap_width,
                                        stride=cfgs.ANCHOR_STRIDE,
                                        name="make_anchors_forRPN")

    # 4. postprocess rpn proposals. such as: decode, clip, NMS
    with tf.variable_scope('postprocess_RPN'):
      # rpn_cls_prob = tf.reshape(rpn_cls_score, [-1, 2])
      # rpn_cls_prob = slim.softmax(rpn_cls_prob, scope='rpn_cls_prob')
      # rpn_box_pred = tf.reshape(rpn_box_pred, [-1, 4])
      rois, roi_scores = postprocess_rpn_proposals(rpn_bbox_pred=rpn_box_pred,
                                                   rpn_cls_prob=rpn_cls_prob,
                                                   img_shape=img_shape,
                                                   anchors=anchors,
                                                   is_training=is_train)
      # rois shape [-1, 4]
      # +++++++++++++++++++++++++++++++++++++add img smry+++++++++++++++++++++++++++++++++++++++++++++++++++++++
      if is_train:
        rois_in_img = show_box_in_tensor.draw_boxes_with_scores(img_batch=images,
                                                                boxes=rois,
                                                                scores=roi_scores)
        tf.summary.image('all_rpn_rois', rois_in_img)

        score_gre_05 = tf.reshape(tf.where(tf.greater_equal(roi_scores, 0.5)), [-1])
        score_gre_05_rois = tf.gather(rois, score_gre_05)
        score_gre_05_score = tf.gather(roi_scores, score_gre_05)
        score_gre_05_in_img = show_box_in_tensor.draw_boxes_with_scores(img_batch=images,
                                                                        boxes=score_gre_05_rois,
                                                                        scores=score_gre_05_score)
        tf.summary.image('score_greater_05_rois', score_gre_05_in_img)
      # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if is_train:
      with tf.variable_scope('sample_anchors_minibatch'):
        rpn_labels, rpn_bbox_targets = \
          tf.py_func(
            anchor_target_layer,
            [gtboxes_batch, img_shape, anchors],
            [tf.float32, tf.float32])
        rpn_bbox_targets = tf.reshape(rpn_bbox_targets, [-1, 4])
        rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
        rpn_labels = tf.reshape(rpn_labels, [-1])
        add_anchor_img_smry(images, anchors, rpn_labels)

      # --------------------------------------add smry----------------------------------------------------------------
      rpn_cls_category = tf.argmax(rpn_cls_prob, axis=1)
      kept_rpppn = tf.reshape(tf.where(tf.not_equal(rpn_labels, -1)), [-1])
      rpn_cls_category = tf.gather(rpn_cls_category, kept_rpppn)
      acc = tf.reduce_mean(tf.to_float(tf.equal(rpn_cls_category, tf.to_int64(tf.gather(rpn_labels, kept_rpppn)))))
      with tf.control_dependencies([rpn_labels]):
        with tf.variable_scope('sample_RCNN_minibatch'):
          rois, labels, bbox_targets = \
            tf.py_func(proposal_target_layer,
                       [rois, gtboxes_batch],
                       [tf.float32, tf.float32, tf.float32])
          rois = tf.reshape(rois, [-1, 4])
          labels = tf.to_int32(labels)
          labels = tf.reshape(labels, [-1])
          bbox_targets = tf.reshape(bbox_targets, [-1, 4 * (FLAGS.nb_classes)])
          add_roi_batch_img_smry(images, rois, labels)

    # -------------------------------------------------------------------------------------------------------------#
    #                                            Fast-RCNN                                                         #
    # -------------------------------------------------------------------------------------------------------------#
    # 5. build Fast-RCNN
    # rois = tf.Print(rois, [tf.shape(rois)], 'rois shape', summarize=10)
    bbox_pred, cls_score = build_fastrcnn(is_train=is_train, feature_to_cropped=feature_to_cropped, rois=rois,
                                          img_shape=img_shape)
    # bbox_pred shape: [-1, 4*(cls_num+1)].
    # cls_score shape： [-1, cls_num+1]
    cls_prob = slim.softmax(cls_score, 'cls_prob')

    # ----------------------------------------------add smry-------------------------------------------------------
    if is_train:
      cls_category = tf.argmax(cls_prob, axis=1)
      fast_acc = tf.reduce_mean(tf.to_float(tf.equal(cls_category, tf.to_int64(labels))))

    #  6. postprocess_fastrcnn
    final_bboxes, final_scores, final_categories = postprocess_fastrcnn(is_train=is_train, rois=rois, bbox_ppred=bbox_pred,
                                                                    scores=cls_prob, img_shape=img_shape)
    if is_train and cfgs.ADD_BOX_IN_TENSORBOARD:
      gtboxes_in_img = show_box_in_tensor.draw_boxes_with_categories(img_batch=images,
                                                                     boxes=gtboxes_batch[:, :-1],
                                                                     labels=gtboxes_batch[:, -1])
      detections_in_img = show_box_in_tensor.draw_boxes_with_categories_and_scores(img_batch=images,
                                                                                   boxes=final_bboxes,
                                                                                   labels=final_categories,
                                                                                   scores=final_scores)
      tf.summary.image('Compare/final_detection', detections_in_img)
      tf.summary.image('Compare/gtboxes', gtboxes_in_img)
  if is_train:
    predictions = None
    forward_dict = { "rpn_box_pred": rpn_box_pred,
                     "rpn_bbox_targets": rpn_bbox_targets,
                     "rpn_cls_score": rpn_cls_score,
                     "rpn_labels": rpn_labels,
                     "bbox_pred": bbox_pred,
                     "bbox_targets": bbox_targets,
                     "cls_score": cls_score,
                     "labels": labels }
    metrics = {'ACC/rpn_accuracy': acc, 'ACC/fast_acc': fast_acc}
  else:
    forward_dict = {}
    predictions = {'filename': filenames,
                   'shape': shapes,
                   'resized_shape':img_shape,
                   'detected_boxes':final_bboxes,
                   'detected_scores':final_scores,
                   'detected_categories':final_categories
                   }

    metrics = {}
  outputs = {'forward_dict': forward_dict, 'predictions': predictions, 'metrics': metrics}
  return outputs

def calc_loss_fn(objects, outputs, trainable_vars):
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
  rpn_box_pred = outputs['rpn_box_pred']
  rpn_bbox_targets = outputs['rpn_bbox_targets']
  rpn_cls_score = outputs['rpn_cls_score']
  rpn_labels = outputs['rpn_labels']
  bbox_pred = outputs['bbox_pred']
  bbox_targets = outputs['bbox_targets']
  cls_score = outputs['cls_score']
  labels = outputs['labels']
  with tf.variable_scope('build_loss') as sc:
    with tf.variable_scope('rpn_loss'):
      rpn_bbox_loss = losses.smooth_l1_loss_rpn(bbox_pred=rpn_box_pred,
                                                bbox_targets=rpn_bbox_targets,
                                                label=rpn_labels,
                                                sigma=cfgs.RPN_SIGMA)
      rpn_select = tf.reshape(tf.where(tf.not_equal(rpn_labels, -1)), [-1])
      rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
      rpn_labels = tf.reshape(tf.gather(rpn_labels, rpn_select), [-1])
      rpn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score,
                                                                                   labels=rpn_labels))
      rpn_cls_loss = rpn_cls_loss * cfgs.RPN_CLASSIFICATION_LOSS_WEIGHT
      rpn_loc_loss = rpn_bbox_loss * cfgs.RPN_LOCATION_LOSS_WEIGHT

    with tf.variable_scope('FastRCNN_loss'):
      if not cfgs.FAST_RCNN_MINIBATCH_SIZE == -1:
        bbox_loss = losses.smooth_l1_loss_rcnn(bbox_pred=bbox_pred,
                                               bbox_targets=bbox_targets,
                                               label=labels,
                                               num_classes=FLAGS.nb_classes,
                                               sigma=cfgs.FASTRCNN_SIGMA)
        cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=cls_score,
          labels=labels))  # beacause already sample before
      else:
        """
        applying OHEM here
        """
        print(20 * "@@")
        print("@@" + 10 * " " + "TRAIN WITH OHEM ...")
        print(20 * "@@")
        cls_loss, bbox_loss = losses.sum_ohem_loss(cls_score=cls_score,
                                                   label=labels,
                                                   bbox_targets=bbox_targets,
                                                   bbox_pred=bbox_pred,
                                                   num_ohem_samples=256,
                                                   num_classes=FLAGS.nb_classes)
      fastrcnn_cls_loss = cls_loss * cfgs.FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT
      fastrcnn_loc_loss = bbox_loss * cfgs.FAST_RCNN_LOCATION_LOSS_WEIGHT
  rpn_total_loss = rpn_bbox_loss + rpn_cls_loss
  fastrcnn_total_loss = cls_loss + bbox_loss
  total_loss = rpn_total_loss + fastrcnn_total_loss

  # ---------------------------------------------------------------------------------------------------add summary
  tf.summary.scalar('RPN_LOSS/cls_loss', rpn_cls_loss)
  tf.summary.scalar('RPN_LOSS/location_loss', rpn_loc_loss)
  tf.summary.scalar('RPN_LOSS/rpn_total_loss', rpn_total_loss)
  tf.summary.scalar('FAST_LOSS/fastrcnn_cls_loss', fastrcnn_cls_loss)
  tf.summary.scalar('FAST_LOSS/fastrcnn_location_loss', fastrcnn_loc_loss)
  tf.summary.scalar('FAST_LOSS/fastrcnn_total_loss', fastrcnn_total_loss)
  return total_loss

class ModelHelper(AbstractModelHelper):
  """Model helper for creating a VGG model for the VOC dataset."""

  def __init__(self, data_format='channels_last'):
    """Constructor function."""

    # class-independent initialization
    super(ModelHelper, self).__init__(data_format, forward_w_labels=True)

    # initialize training & evaluation subsets
    self.dataset_train = PascalVocDataset(preprocess_fn=preprocess_image, is_train=True)
    self.dataset_eval = PascalVocDataset(preprocess_fn=preprocess_image, is_train=False)

    # setup hyper-parameters
    self.batch_size = None  # track the most recently-used one
    self.model_scope = "model"

  def build_dataset_train(self, enbl_trn_val_split=False):
    """Build the data subset for training, usually with data augmentation."""
    return self.dataset_train.build()

  def build_dataset_eval(self):
    """Build the data subset for evaluation, usually without data augmentation."""
    return self.dataset_eval.build()

  def forward_train(self, inputs, objects, data_format='channels_last'):
    """Forward computation at training."""
    inputs_dict = {'inputs': inputs, 'objects': objects}
    outputs = forward_fn(inputs_dict, True)
    self.vars = slim.get_model_variables()
    return outputs

  def forward_eval(self, inputs, data_format='channels_last'):
    """Forward computation at evaluation."""
    inputs_dict = {'inputs': inputs, 'objects': None}
    outputs = forward_fn(inputs_dict, False)
    return outputs

  def calc_loss(self, objects, outputs, trainable_vars):
    """Calculate loss (and some extra evaluation metrics)."""
    forward_dict = outputs['forward_dict']
    metrics = outputs['metrics']
    loss = tf.constant(0,dtype=tf.float32)
    if forward_dict != {}:
      """only build loss at training"""
      loss = calc_loss_fn(objects, forward_dict, trainable_vars)
    return loss, metrics

  def setup_lrn_rate(self, global_step):
    """Setup the learning rate (and number of training iterations)."""

    lrn_rate = tf.train.piecewise_constant(global_step,
                                     boundaries=[np.int64(cfgs.DECAY_STEP[0]), np.int64(cfgs.DECAY_STEP[1])],
                                     values=[cfgs.LR, cfgs.LR / 10., cfgs.LR / 100.])
    nb_iters = FLAGS.nb_iters_train

    tf.summary.scalar('lrn_rate', lrn_rate)

    return lrn_rate, nb_iters

  def warm_start(self, sess):
    """Initialize the model for warm-start.

    Description:
    * We use a pre-trained ImageNet classification model to initialize the backbone part of the SSD
      model for feature extraction. If the SSD model's checkpoint files already exist, then skip.
    """
    # early return if checkpoint files already exist
    checkpoint_path = tf.train.latest_checkpoint(os.path.dirname(FLAGS.save_path))
    model_variables = self.vars
    if checkpoint_path != None:
      if cfgs.RESTORE_FROM_RPN:
        print('___restore from rpn___')

        restore_variables = [var for var in model_variables if not var.name.startswith(self.model_scope + 'FastRCNN_Head')] + \
                            [slim.get_or_create_global_step()]
        for var in restore_variables:
          print(var.name)
        saver = tf.train.Saver()
        saver.build()
        saver.restore(sess, checkpoint_path)
      else:
        print("___restore from trained model___")
        for var in model_variables:
          print(var.name)
        saver = tf.train.Saver(model_variables)
        saver.build()
        saver.restore(sess, checkpoint_path)
      print("model restore from :", checkpoint_path)
    else:
      if cfgs.NET_NAME.startswith("resnet"):
        weights_name = cfgs.NET_NAME
      elif cfgs.NET_NAME.startswith("MobilenetV2"):
        weights_name = "mobilenet/mobilenet_v2_1.0_224"
      else:
        raise Exception('net name must in [resnet_v1_101, resnet_v1_50, MobilenetV2]')
      checkpoint_path = os.path.join(FLAGS.backbone_ckpt_dir, weights_name + '.ckpt')
      print("model restore from pretrained mode, path is :", checkpoint_path)
      # for var in model_variables:
      #     print(var.name)
      # print(20*"__++__++__")

      def name_in_ckpt_rpn(var):
        '''
        model/resnet_v1_50/block4 -->resnet_v1_50/block4
        model/MobilenetV2/** -- > MobilenetV2 **
        :param var:
        :return:
        '''
        return '/'.join(var.op.name.split('/')[1:])

      def name_in_ckpt_fastrcnn_head(var):
        '''
        model/Fast-RCNN/resnet_v1_50/block4 -->resnet_v1_50/block4
        model/Fast-RCNN/MobilenetV2/** -- > MobilenetV2 **
        :param var:
        :return:
        '''
        return '/'.join(var.op.name.split('/')[2:])
      nameInCkpt_Var_dict = {}
      for var in model_variables:
        if var.name.startswith(self.model_scope + '/Fast-RCNN/' + cfgs.NET_NAME):  # +'/block4'
          var_name_in_ckpt = name_in_ckpt_fastrcnn_head(var)
          nameInCkpt_Var_dict[var_name_in_ckpt] = var
        else:
          if var.name.startswith(self.model_scope + '/' + cfgs.NET_NAME):
            var_name_in_ckpt = name_in_ckpt_rpn(var)
            nameInCkpt_Var_dict[var_name_in_ckpt] = var
          else:
            continue
      restore_variables = nameInCkpt_Var_dict
      if not restore_variables:
        tf.logging.warning('no variables to restore.')
        return
      for key, item in restore_variables.items():
        print("var_in_graph: ", item.name)
        print("var_in_ckpt: ", key)
        print(20 * "___")
      # restore variables from checkpoint files
      saver = tf.train.Saver(restore_variables, reshape=False)
      saver.build()
      saver.restore(sess, checkpoint_path)
      print(20 * "****")
      print("restore from pretrained_weighs in IMAGE_NET")
    print('model restored')


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
      raw_shape = outputs['predictions']['shape'][0]
      resized_shape= outputs['predictions']['resized_shape']

      detected_boxes = outputs['predictions']['detected_boxes']
      detected_scores = outputs['predictions']['detected_scores']
      detected_categories = outputs['predictions']['detected_categories']


      raw_h, raw_w = raw_shape[0], raw_shape[1]
      resized_h, resized_w = resized_shape[1], resized_shape[2]

      xmin, ymin, xmax, ymax = detected_boxes[:, 0], detected_boxes[:, 1], \
                               detected_boxes[:, 2], detected_boxes[:, 3]

      xmin = xmin * raw_w / resized_w
      xmax = xmax * raw_w / resized_w
      ymin = ymin * raw_h / resized_h
      ymax = ymax * raw_h / resized_h

      boxes = np.transpose(np.stack([xmin, ymin, xmax, ymax]))
      dets = np.hstack((detected_categories.reshape(-1, 1),
                        detected_scores.reshape(-1, 1),
                        boxes))

      for cls_id in range(1, FLAGS.nb_classes):
        with open(os.path.join(FLAGS.outputs_dump_dir, 'results_%d.txt' % cls_id), 'a') as o_file:
          this_cls_detections = dets[dets[:, 0] == cls_id]
          if this_cls_detections.shape[0] == 0:
            continue  # this cls has none detections in this img
          for a_det in this_cls_detections:
            o_file.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                    format(filename, a_det[1],
                           a_det[2], a_det[3],
                           a_det[4], a_det[5]))  # that is [img_name, score, xmin, ymin, xmax, ymax]

    elif action == 'eval':
      do_python_eval(os.path.join(self.dataset_eval.data_dir, 'test'), FLAGS.outputs_dump_dir)
    else:
      raise ValueError('unrecognized action in dump_n_eval(): ' + action)

  @property
  def model_name(self):
    """Model's name."""
    return cfgs.NET_NAME

  @property
  def dataset_name(self):
    """Dataset's name."""
    return 'pascalvoc'


