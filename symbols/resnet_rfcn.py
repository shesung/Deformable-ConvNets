# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yuwen Xiong, Xizhou Zhu
# --------------------------------------------------------

import cPickle
import mxnet as mx
from symbol import Symbol
from operator_py.proposal import *
from operator_py.proposal_target import *
from operator_py.box_annotator_ohem import *

#from resnet import get_resnet
from resnet_v1 import get_resnet

class resnet_rfcn(Symbol):

    def get_backbone(self, data, cfg):
        strides = [1, 2, 2, 1]
        num_layers = cfg.network.NUM_LAYERS
        return get_resnet(data, num_layers, strides)

    def proposal(self, cls_prob, bbox_pred, im_info, feature_stride, scales, ratios,
                 rpn_pre_nms_top_n, rpn_post_nms_top_n, threshold, rpn_min_size, prefix='', cxx_proposal=True):
        if cxx_proposal:
            rois = mx.contrib.sym.MultiProposal(cls_prob=cls_prob,
                                                bbox_pred=bbox_pred,
                                                im_info=im_info,
                                                name=prefix+'rois',
                                                feature_stride=feature_stride,
                                                scales=scales,
                                                ratios=ratios,
                                                rpn_pre_nms_top_n=rpn_pre_nms_top_n,
                                                rpn_post_nms_top_n=rpn_post_nms_top_n,
                                                threshold=threshold,
                                                rpn_min_size=rpn_min_size)
        else:
            rois = mx.sym.Custom(cls_prob=cls_prob,
                                 bbox_pred=bbox_pred,
                                 im_info=im_info,
                                 name=prefix+'rois',
                                 op_type='proposal',
                                 feat_stride=feature_stride,
                                 scales=scales,
                                 ratios=ratios,
                                 rpn_pre_nms_top_n=rpn_pre_nms_top_n,
                                 rpn_post_nms_top_n=rpn_post_nms_top_n,
                                 threshold=threshold,
                                 rpn_min_size=rpn_min_size)
        return rois


    def get_proposals(self, data, im_info, cfg, rpn_filters=512, is_train=True):
        num_anchors = cfg.network.NUM_ANCHORS
        rpn_conv = mx.sym.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=rpn_filters,
                                      name="rpn_conv_3x3")
        rpn_relu = mx.sym.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
        rpn_cls_score = mx.sym.Convolution(data=rpn_relu, kernel=(1, 1), pad=(0, 0),
                                           num_filter=2 * num_anchors, name="rpn_cls_score")
        rpn_bbox_pred = mx.sym.Convolution(data=rpn_relu, kernel=(1, 1), pad=(0, 0),
                                           num_filter=4 * num_anchors, name="rpn_bbox_pred")

        # ROI proposal
        rpn_cls_score_reshape = mx.sym.Reshape(data=rpn_cls_score,
                                               shape=(0, 2, -1, 0),
                                               name="rpn_cls_score_reshape")
        rpn_cls_act = mx.sym.SoftmaxActivation(data=rpn_cls_score_reshape,
                                               mode="channel", name="rpn_cls_act")
        rpn_cls_act_reshape = mx.sym.Reshape(data=rpn_cls_act,
                                             shape=(0, 2 * num_anchors, -1, 0),
                                             name='rpn_cls_act_reshape')

        rpn_cfg = cfg.TRAIN if is_train else cfg.TEST
        feature_stride     = cfg.network.RPN_FEAT_STRIDE
        scales             = tuple(cfg.network.ANCHOR_SCALES)
        ratios             = tuple(cfg.network.ANCHOR_RATIOS)
        rpn_pre_nms_top_n  = rpn_cfg.RPN_PRE_NMS_TOP_N
        rpn_post_nms_top_n = rpn_cfg.RPN_POST_NMS_TOP_N
        threshold          = rpn_cfg.RPN_NMS_THRESH
        rpn_min_size       = rpn_cfg.RPN_MIN_SIZE
        cxx_proposal       = rpn_cfg.CXX_PROPOSAL
        rois = self.proposal(rpn_cls_act_reshape, rpn_bbox_pred, im_info,
                             feature_stride, scales, ratios,
                             rpn_pre_nms_top_n, rpn_post_nms_top_n, threshold, rpn_min_size,
                             cxx_proposal=cxx_proposal)

        ret_syms = []
        if is_train:
            rpn_label = mx.sym.Variable(name='label')
            rpn_bbox_target = mx.sym.Variable(name='bbox_target')
            rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')

            # rpn classification
            rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label,
                                                multi_output=True,normalization='valid', use_ignore=True,
                                                ignore_label=-1, name="rpn_cls_prob")
            # rpn bounding box regression
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=3.0,
                                                                data=(rpn_bbox_pred - rpn_bbox_target))
            rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_,
                                            grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)
            ret_syms = [rpn_cls_prob, rpn_bbox_loss]
        return rois, ret_syms


    def get_symbol(self, cfg, is_train=True):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        batch_size = cfg.TRAIN.IMAGES_PER_GPU  if is_train else cfg.TEST.IMAGES_PER_GPU
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS

        # input init
        data = mx.sym.Variable(name="data")
        im_info = mx.sym.Variable(name="im_info")

        # backbone
        c2, c3, c4, c5 = self.get_backbone(data, cfg)

        # rpn
        rois, rpn_syms = self.get_proposals(c4, im_info, cfg, 256, is_train)

        if is_train:
            gt_boxes = mx.sym.Variable(name="gt_boxes")
            gt_kps = mx.sym.Variable(name="gt_kps")

            # ROI proposal target
            rois, label, bbox_target, bbox_weight = mx.sym.Custom(rois=rois,
                                                                  gt_boxes=gt_boxes,
                                                                  op_type='proposal_target',
                                                                  num_classes=num_reg_classes,
                                                                  batch_images=batch_size,
                                                                  batch_rois=cfg.TRAIN.BATCH_ROIS,
                                                                  fg_fraction=cfg.TRAIN.FG_FRACTION)

        # rfcn head
        conv_feat = c5
        feat_stride = 16

        conv_new_1 = mx.sym.Convolution(data=conv_feat, kernel=(1, 1), num_filter=1024, name="conv_new_1", lr_mult=3.0)
        relu_new_1 = mx.sym.Activation(data=conv_new_1, act_type='relu', name='relu_new_1')

        rfcn_cls  = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=7*7*num_classes, name="rfcn_cls")
        rfcn_bbox = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=7*7*4*num_reg_classes, name="rfcn_bbox")
        psroipooled_cls_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois,
                                                           group_size=7, pooled_size=7, output_dim=num_classes,
                                                           spatial_scale=1.0/feat_stride)
        psroipooled_loc_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois,
                                                           group_size=7, pooled_size=7, output_dim=8,
                                                           spatial_scale=1.0/feat_stride)
        cls_score = mx.sym.Pooling(name='ave_cls_scors_rois', data=psroipooled_cls_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        bbox_pred = mx.sym.Pooling(name='ave_bbox_pred_rois', data=psroipooled_loc_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
        bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4*num_reg_classes))

        if is_train:
            if cfg.TRAIN.ENABLE_OHEM:
                ohem_batch_size = cfg.TRAIN.BATCH_ROIS_OHEM * batch_size
                labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM',
                                                               num_classes=num_classes,
                                                               num_reg_classes=num_reg_classes,
                                                               roi_per_img=ohem_batch_size,
                                                               cls_score=cls_score,
                                                               bbox_pred=bbox_pred,
                                                               labels=label,
                                                               bbox_targets=bbox_target,
                                                               bbox_weights=bbox_weight)
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem,
                                                normalization='valid', use_ignore=True, ignore_label=-1)
                bbox_loss_ = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0,
                                                                  data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_,
                                            grad_scale=1.0 / ohem_batch_size)
                rcnn_label = labels_ohem
            else:
                roi_batch_size = cfg.TRAIN.BATCH_ROIS * batch_size
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label,
                                                normalization='valid')
                bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0,
                                                            data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_,
                                            grad_scale=1.0 / roi_batch_size)
                rcnn_label = label

            # reshape output
            rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(batch_size, -1),
                                        name='label_reshape')
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(batch_size, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(batch_size, -1, 4 * num_reg_classes),
                                       name='bbox_loss_reshape')
            group = mx.sym.Group(rpn_syms + [cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)])
        else:
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(batch_size, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(batch_size, -1, 4 * num_reg_classes),
                                       name='bbox_pred_reshape')
            group = mx.sym.Group([rois, cls_prob, bbox_pred])

        self.sym = group
        return group


    def init_weight_rpn(self, cfg, arg_params, aux_params):
        arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_conv_3x3_weight'])
        arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_conv_3x3_bias'])
        arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_cls_score_weight'])
        arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score_bias'])
        arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_bbox_pred_weight'])
        arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred_bias'])


    def init_weight_rfcn(self, cfg, arg_params, aux_params):
        arg_params['conv_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv_new_1_weight'])
        arg_params['conv_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv_new_1_bias'])
        arg_params['rfcn_cls_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rfcn_cls_weight'])
        arg_params['rfcn_cls_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_cls_bias'])
        arg_params['rfcn_bbox_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rfcn_bbox_weight'])
        arg_params['rfcn_bbox_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_bbox_bias'])


    def init_weight(self, cfg, arg_params, aux_params):
        self.init_weight_rpn(cfg, arg_params, aux_params)
        self.init_weight_rfcn(cfg, arg_params, aux_params)
