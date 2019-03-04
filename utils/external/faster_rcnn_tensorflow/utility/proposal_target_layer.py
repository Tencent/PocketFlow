# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils.external.faster_rcnn_tensorflow.configs import cfgs
import numpy as np
import numpy.random as npr

from utils.external.faster_rcnn_tensorflow.utility import encode_and_decode

def bbox_overlaps(boxes, query_boxes):
  """
  Parameters
  ----------
  boxes: (N, 4) ndarray of float
  query_boxes: (K, 4) ndarray of float
  Returns
  -------
  overlaps: (N, K) ndarray of overlap between boxes and query_boxes
  """
  N = boxes.shape[0]
  K = query_boxes.shape[0]
  overlaps = np.zeros((N, K))
  for k in range(K):
    box_area = (
        (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
        (query_boxes[k, 3] - query_boxes[k, 1] + 1)
    )
    for n in range(N):
      iw = (
          min(boxes[n, 2], query_boxes[k, 2]) -
          max(boxes[n, 0], query_boxes[k, 0]) + 1
      )
      if iw > 0:
        ih = (
            min(boxes[n, 3], query_boxes[k, 3]) -
            max(boxes[n, 1], query_boxes[k, 1]) + 1
        )
        if ih > 0:
          ua = float(
            (boxes[n, 2] - boxes[n, 0] + 1) *
            (boxes[n, 3] - boxes[n, 1] + 1) +
            box_area - iw * ih
          )
          overlaps[n, k] = iw * ih / ua
  return overlaps

def proposal_target_layer(rpn_rois, gt_boxes):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """
    # Proposal ROIs (x1, y1, x2, y2) coming from RPN
    # gt_boxes (x1, y1, x2, y2, label)
    if cfgs.ADD_GTBOXES_TO_TRAIN:
        all_rois = np.vstack((rpn_rois, gt_boxes[:, :-1]))
    else:
        all_rois = rpn_rois
    # np.inf
    rois_per_image = np.inf if cfgs.FAST_RCNN_MINIBATCH_SIZE == -1 else cfgs.FAST_RCNN_MINIBATCH_SIZE

    fg_rois_per_image = np.round(cfgs.FAST_RCNN_POSITIVE_RATE * rois_per_image)

    # Sample rois with classification labels and bounding box regression
    labels, rois, bbox_targets = _sample_rois(all_rois, gt_boxes, fg_rois_per_image,
                                              rois_per_image, cfgs.CLASS_NUM+1)
    # print(labels.shape, rois.shape, bbox_targets.shape)
    rois = rois.reshape(-1, 4)
    labels = labels.reshape(-1)
    bbox_targets = bbox_targets.reshape(-1, (cfgs.CLASS_NUM+1) * 4)

    return rois, labels, bbox_targets


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = int(4 * cls)
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]

    return bbox_targets


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image.
    that is : [label, tx, ty, tw, th]
    """

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = encode_and_decode.encode_boxes(unencode_boxes=gt_rois,
                                             reference_boxes=ex_rois,
                                             scale_factors=cfgs.ROI_SCALE_FACTORS)
    # targets = encode_and_decode.encode_boxes(ex_rois=ex_rois,
    #                                          gt_rois=gt_rois,
    #                                          scale_factor=cfgs.ROI_SCALE_FACTORS)

    return np.hstack(
        (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def _sample_rois(all_rois, gt_boxes, fg_rois_per_image,
                 rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.

    all_rois shape is [-1, 4]
    gt_boxes shape is [-1, 5]. that is [x1, y1, x2, y2, label]
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(all_rois,gt_boxes)
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, -1]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfgs.FAST_RCNN_IOU_POSITIVE_THRESHOLD)[0]

    # Guard against the case when an image has fewer than fg_rois_per_image
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfgs.FAST_RCNN_IOU_POSITIVE_THRESHOLD) &
                       (max_overlaps >= cfgs.FAST_RCNN_IOU_NEGATIVE_THRESHOLD))[0]
    # print("first fileter, fg_size: {} || bg_size: {}".format(fg_inds.shape, bg_inds.shape))
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)

    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_this_image), replace=False)
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_this_image), replace=False)

    # print("second fileter, fg_size: {} || bg_size: {}".format(fg_inds.shape, bg_inds.shape))
    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)


    # Select sampled values from various arrays:
    labels = labels[keep_inds]

    # Clamp labels for the background RoIs to 0
    labels[int(fg_rois_per_this_image):] = 0
    rois = all_rois[keep_inds]

    bbox_target_data = _compute_targets(
        rois, gt_boxes[gt_assignment[keep_inds], :-1], labels)
    bbox_targets = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets
