# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------
"""
Fast R-CNN:
data =
    {'data': [num_images, c, h, w],
    'rois': [num_rois, 5]}
label =
    {'label': [num_rois],
    'bbox_target': [num_rois, 4 * num_classes],
    'bbox_weight': [num_rois, 4 * num_classes]}
roidb extended format [image_index]
    ['image', 'height', 'width', 'flipped',
     'boxes', 'gt_classes', 'gt_overlaps', 'max_classes', 'max_overlaps', 'bbox_targets']
"""

import numpy as np
import numpy.random as npr

from utils.image import get_image, tensor_vstack
from bbox.bbox_transform import bbox_overlaps, bbox_transform
from bbox.bbox_regression import expand_bbox_regression_targets
from config.config import config as cfg

import time ###

def sample_rois(rois, fg_rois_per_image, rois_per_image, num_classes,
                labels=None, overlaps=None, bbox_targets=None, gt_boxes=None, gt_kps=None):
    """
    generate random sample of ROIs comprising foreground and background examples
    :param rois: all_rois [n, 4]; e2e: [n, 5] with batch_index
    :param fg_rois_per_image: foreground roi number
    :param rois_per_image: total roi number
    :param num_classes: number of classes
    :param labels: maybe precomputed
    :param overlaps: maybe precomputed (max_overlaps)
    :param bbox_targets: maybe precomputed
    :param gt_boxes: optional for e2e [n, 5] (x1, y1, x2, y2, cls)
    :param gt_kps: optional for e2e [n, num_kps*3] (x1, y1, v1, ...)
    :return: (labels, rois, bbox_targets, bbox_weights)
    """
    if labels is None:
        overlaps = bbox_overlaps(rois[:, 1:].astype(np.float), gt_boxes[:, :4].astype(np.float))
        gt_assignment = overlaps.argmax(axis=1)
        overlaps = overlaps.max(axis=1)
        labels = gt_boxes[gt_assignment, 4]

    # foreground RoI with FG_THRESH overlap
    fg_indexes = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)
    # Sample foreground regions without replacement
    if len(fg_indexes) > fg_rois_per_this_image:
        fg_indexes = npr.choice(fg_indexes, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_indexes = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) & (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_indexes.size)
    # Sample foreground regions without replacement
    if len(bg_indexes) > bg_rois_per_this_image:
        bg_indexes = npr.choice(bg_indexes, size=bg_rois_per_this_image, replace=False)

    # indexes selected
    keep_indexes = np.append(fg_indexes, bg_indexes)

    # pad more to ensure a fixed minibatch size
    while keep_indexes.shape[0] < rois_per_image:
        gap = np.minimum(len(rois), rois_per_image - keep_indexes.shape[0])
        gap_indexes = npr.choice(range(len(rois)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, gap_indexes)

    # select labels
    labels = labels[keep_indexes]
    # set labels of bg_rois to be 0
    labels[fg_rois_per_this_image:] = 0
    rois = rois[keep_indexes]

    # load or compute bbox_target
    if bbox_targets is not None:
        bbox_target_data = bbox_targets[keep_indexes, :]
    else:
        targets = bbox_transform(rois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :4])
        if cfg.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
            targets = ((targets - np.array(cfg.TRAIN.BBOX_MEANS))
                       / np.array(cfg.TRAIN.BBOX_STDS))
        bbox_target_data = np.hstack((labels[:, np.newaxis], targets))

    bbox_targets, bbox_weights = \
        expand_bbox_regression_targets(bbox_target_data, num_classes, cfg)

    res = {'rois_output': rois,
           'label'      : labels,
           'bbox_target': bbox_targets,
           'bbox_weight': bbox_weights,
        }
    if gt_kps is not None:
        keep_kps = gt_kps[gt_assignment[keep_indexes]]
        n_keep = keep_kps.shape[0]
        K = cfg.dataset.NUM_KEYPOINTS
        assert gt_kps.shape[1] == K*3

        G = cfg.network.KEYPOINTS_POOLED_SIZE
        kps_labels = np.empty([n_keep, K], dtype=np.float32)
        kps_labels.fill(-1)
        kps_targets = np.zeros([n_keep, K, G, G, 2], dtype=np.float32)
        kps_weights = kps_targets.copy()
        num_fg = fg_indexes.size
        assert num_fg > 0, 'need at least one roi'

        # assgin kp targets
        fg_kps_label, fg_kps_target, fg_kps_weight =  assign_keypoints(rois[:num_fg, 1:], keep_kps[:num_fg], pooled_size=G)
        kps_labels[:num_fg]  = fg_kps_label
        kps_targets[:num_fg] = fg_kps_target
        normalizer = 1.0 / (num_fg + 1e-3)
        kps_weights[:num_fg] = fg_kps_weight * normalizer

        res['kps_label'] = kps_labels.reshape([-1])
        res['kps_target'] = kps_targets.transpose([0,1,4,2,3]).reshape([n_keep, -1, G, G])
        res['kps_weight'] = kps_weights.transpose([0,1,4,2,3]).reshape([n_keep, -1, G, G])

    return res


def assign_keypoints(rois, gt_kps, pooled_size=7):
    '''
    rois: [N, 4]
    gt_kps: [N, K*3]

    Returns:
    kps_label: [N, K]
    kps_target: [N, K, pooled_size, pooled_size, 2]
    kps_weight: [N, K, pooled_size, pooled_size, 2]
    '''
    assert rois.shape[0]  == gt_kps.shape[0], 'n_rois and n_gt do not match !'

    N = rois.shape[0]
    K = gt_kps.shape[1] / 3
    G = pooled_size

    # generate grid centers for all rois
    def generate_grid_centers(rois, G):
        roi_wh = rois[:,2:4] - rois[:,0:2]  # [N, 2]
        roi_origin = rois[:,0:2]            # [N, 2]
        gx, gy = np.meshgrid(np.arange(0.5/G, 1., 1./G), np.arange(0.5/G, 1., 1./G))
        g_ctrs = np.stack([gx.ravel(), gy.ravel()], axis=-1)       # [G*G, 2]
        ctrs = roi_wh[:, np.newaxis, :] * g_ctrs[np.newaxis, :, :] # [N, G*G, 2]
        ctrs = ctrs + roi_origin[:, np.newaxis, :]                 # [N, G*G, 2]
        ctrs = np.repeat(ctrs, K, axis=0)                          # [N*K, G*G, 2]
        return ctrs
    all_ctrs = generate_grid_centers(rois, G) # [N*K, G*G, 2]

    # compute offset for each grid of each roi
    kp_xyv = gt_kps.reshape([-1, 3])                # [N*K, 3]
    kp_v   = kp_xyv[:,2].astype(np.int)             # [N*K]
    kp_xy  = kp_xyv[:,:2]                           # [N*K, 2]
    kps_offset = kp_xy[:, np.newaxis, :] - all_ctrs # [N*K, G*G, 2]

    # assign each kp to one grid
    dist = np.sum(np.square(kps_offset), axis=-1) # [N*K, G*G]
    argmin_dist = dist.argmin(axis=1)             # [N*K]
    kps_label = argmin_dist.astype(np.float32)    # [N*K]
    kps_label[np.where(kp_v < 1)] = -1

    # normalize targets by grid width and height
    wh = rois[:,2:4] - rois[:,0:2] + 1.0            # [N, 2]
    gwh = wh * (1.0 / G)
    gwh = gwh.repeat(K, axis=0)                     # [N*K, 2]
    kps_target = kps_offset / gwh[:, np.newaxis, :] # [N*K, G*G, 2]

    # compute weights, and normalize by K
    kps_weight = np.zeros_like(kps_target) # [N*K, G*G, 2]
    kps_weight[np.arange(kps_weight.shape[0]), argmin_dist] = 1.0/K
    kps_weight[np.where(kp_v < 1)] = 0

    return kps_label.reshape([N, K]), kps_target.reshape([N, K, G, G, 2]), kps_weight.reshape([N, K, G, G, 2])

