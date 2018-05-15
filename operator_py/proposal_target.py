# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------

"""
Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to them.
"""

import mxnet as mx
import numpy as np
from distutils.util import strtobool
from easydict import EasyDict as edict
import cPickle


from core.rcnn import sample_rois
from config.config import config



class ProposalTargetOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, batch_images, batch_rois, fg_fraction):
        super(ProposalTargetOperator, self).__init__()
        self._num_classes = num_classes
        self._batch_images = batch_images
        self._batch_rois = batch_rois
        self._fg_fraction = fg_fraction

    def forward(self, is_train, req, in_data, out_data, aux):
        all_rois = in_data[0].asnumpy()
        all_boxes = in_data[1].asnumpy()
        all_kps = None
        if config.network.PREDICT_KEYPOINTS:
            all_kps = in_data[2].asnumpy()

        all_targets = dict()
        num_images = all_boxes.shape[0]
        for i in range(num_images):
            valid_inds = np.where(all_boxes[i,:,4] > 0)[0]
            gt_boxes = all_boxes[i, valid_inds]
            gt_kps   = all_kps[i, valid_inds] if all_kps is not None else None
            roi_inds = np.where(all_rois[:,0] == i)[0]
            if self._batch_rois == -1:
                rois_per_image = roi_inds.shape[0] # keep same number of rois
                fg_rois_per_image = rois_per_image
            else:
                assert self._batch_rois % self._batch_images == 0, \
                        'batchimages {} must devide batch_rois {}'.format(self._batch_images, self._batch_rois)

                rois_per_image = self._batch_rois / num_images
                fg_rois_per_image = np.round(self._fg_fraction * rois_per_image).astype(int)

            # Include ground-truth boxes in the set of candidate rois
            gt_rois = np.zeros((gt_boxes.shape[0], 5), dtype=gt_boxes.dtype)
            gt_rois[:, 0] = i
            gt_rois[:, 1:5] = gt_boxes[:,:4]
            rois = np.vstack((all_rois[roi_inds], gt_rois))
            targets = sample_rois(rois, fg_rois_per_image, rois_per_image, self._num_classes,
                                  gt_boxes=gt_boxes, gt_kps=gt_kps)
            DEBUG = False ###
            if DEBUG:
                labels= targets['label']
                print 'gt: {}\tfg: {}\tbg: {}\tignore: {}'.format(valid_inds.shape[0],
                        (labels > 0).sum(),
                        (labels == 0).sum(),
                        (labels == -1).sum())

            for ind, k in enumerate(outputs):
                if k not in all_targets:
                    all_targets[k] = []
                all_targets[k].append(targets[k])

        for ind, k in enumerate(outputs):
            target =  np.concatenate(all_targets[k], axis=0)
            self.assign(out_data[ind], req[ind], target)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        if config.network.PREDICT_KEYPOINTS:
            self.assign(in_grad[2], req[2], 0)


arguments = ['rois', 'gt_boxes']
outputs   = ['rois_output', 'label', 'bbox_target', 'bbox_weight']
if config.network.PREDICT_KEYPOINTS:
    arguments += ['gt_kps']
    outputs   += ['kps_label', 'kps_pos_target', 'kps_pos_weight']


@mx.operator.register('proposal_target')
class ProposalTargetProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes, batch_images, batch_rois, fg_fraction='0.25'):
        super(ProposalTargetProp, self).__init__(need_top_grad=False)
        self._num_classes = int(num_classes)
        self._batch_images = int(batch_images)
        self._batch_rois = int(batch_rois)
        self._fg_fraction = float(fg_fraction)

    def list_arguments(self):
        return arguments

    def list_outputs(self):
        return outputs

    def infer_shape(self, in_shape):
        rpn_rois_shape = in_shape[0]
        gt_boxes_shape = in_shape[1]

        rois = rpn_rois_shape[0] if self._batch_rois == -1 else self._batch_rois
        label_shape = (rois, )
        bbox_target_shape = (rois, self._num_classes * 4)
        bbox_weight_shape = (rois, self._num_classes * 4)
        output_shapes = {'rois_output': (rois, 5),
                         'label':       (rois, ),
                         'bbox_target': (rois, self._num_classes * 4),
                         'bbox_weight': (rois, self._num_classes * 4)
                        }

        if config.network.PREDICT_KEYPOINTS:
            G = config.network.KEYPOINTS_POOLED_SIZE
            K = config.dataset.NUM_KEYPOINTS
            output_shapes['kps_label']      = (rois*K, )
            output_shapes['kps_pos_target'] = (rois, 2*K, G, G)
            output_shapes['kps_pos_weight'] = (rois, 2*K, G, G)

        return in_shape, [output_shapes[k] for k in outputs]

    def create_operator(self, ctx, shapes, dtypes):
        return ProposalTargetOperator(self._num_classes, self._batch_images, self._batch_rois, self._fg_fraction)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
