# encoding: utf-8
"""
@author: zeming li
@contact: zengarden2009@gmail.com
"""

from utils.external.faster_rcnn_tensorflow.configs import cfgs
from utils.external.faster_rcnn_tensorflow.utility import encode_and_decode
from utils.external.faster_rcnn_tensorflow.utility import boxes_utils
import tensorflow as tf
import numpy as np


def postprocess_rpn_proposals(rpn_bbox_pred, rpn_cls_prob, img_shape, anchors, is_training):
    '''

    :param rpn_bbox_pred: [-1, 4]
    :param rpn_cls_prob: [-1, 2]
    :param img_shape:
    :param anchors:[-1, 4]
    :param is_training:
    :return:
    '''

    if is_training:
        pre_nms_topN = cfgs.RPN_TOP_K_NMS_TRAIN
        post_nms_topN = cfgs.RPN_MAXIMUM_PROPOSAL_TARIN
        nms_thresh = cfgs.RPN_NMS_IOU_THRESHOLD
    else:
        pre_nms_topN = cfgs.RPN_TOP_K_NMS_TEST
        post_nms_topN = cfgs.RPN_MAXIMUM_PROPOSAL_TEST
        nms_thresh = cfgs.RPN_NMS_IOU_THRESHOLD

    cls_prob = rpn_cls_prob[:, 1]

    # 1. decode boxes
    decode_boxes = encode_and_decode.decode_boxes(encoded_boxes=rpn_bbox_pred,
                                                  reference_boxes=anchors,
                                                  scale_factors=cfgs.ANCHOR_SCALE_FACTORS)

    # decode_boxes = encode_and_decode.decode_boxes(boxes=anchors,
    #                                               deltas=rpn_bbox_pred,
    #                                               scale_factor=None)

    # 2. clip to img boundaries
    decode_boxes = boxes_utils.clip_boxes_to_img_boundaries(decode_boxes=decode_boxes,
                                                            img_shape=img_shape)

    # 3. get top N to NMS
    if pre_nms_topN > 0:
        pre_nms_topN = tf.minimum(pre_nms_topN, tf.shape(decode_boxes)[0], name='avoid_unenough_boxes')
        cls_prob, top_k_indices = tf.nn.top_k(cls_prob, k=pre_nms_topN)
        decode_boxes = tf.gather(decode_boxes, top_k_indices)

    # 4. NMS
    keep = tf.image.non_max_suppression(
        boxes=decode_boxes,
        scores=cls_prob,
        max_output_size=post_nms_topN,
        iou_threshold=nms_thresh)

    final_boxes = tf.gather(decode_boxes, keep)
    final_probs = tf.gather(cls_prob, keep)

    return final_boxes, final_probs

