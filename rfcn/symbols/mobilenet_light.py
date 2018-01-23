# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yuwen Xiong, Xizhou Zhu
# --------------------------------------------------------

import cPickle
import mxnet as mx
from utils.symbol import Symbol
from operator_py.proposal import *
from operator_py.proposal_target import *
from operator_py.box_annotator_ohem import *


class mobilenet_light(Symbol):

    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.use_global_stats = True

    def get_stage_1(self, data):
        use_global_stats = self.use_global_stats
        conv1 = mx.symbol.Convolution(name='conv1', data=data , num_filter=32, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True)
        conv1_bn = mx.symbol.BatchNorm(name='conv1_bn', data=conv1 , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
        conv1_scale = conv1_bn
        relu1 = mx.symbol.Activation(name='relu1', data=conv1_scale , act_type='relu')

        conv2_1_dw = mx.symbol.Convolution(name='conv2_1_dw', data=relu1 , num_filter=32, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=32)
        conv2_1_dw_bn = mx.symbol.BatchNorm(name='conv2_1_dw_bn', data=conv2_1_dw , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
        conv2_1_dw_scale = conv2_1_dw_bn
        relu2_1_dw = mx.symbol.Activation(name='relu2_1_dw', data=conv2_1_dw_scale , act_type='relu')

        conv2_1_sep = mx.symbol.Convolution(name='conv2_1_sep', data=relu2_1_dw , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
        conv2_1_sep_bn = mx.symbol.BatchNorm(name='conv2_1_sep_bn', data=conv2_1_sep , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
        conv2_1_sep_scale = conv2_1_sep_bn
        relu2_1_sep = mx.symbol.Activation(name='relu2_1_sep', data=conv2_1_sep_scale , act_type='relu')

        conv2_2_dw = mx.symbol.Convolution(name='conv2_2_dw', data=relu2_1_sep , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True, num_group=64)
        conv2_2_dw_bn = mx.symbol.BatchNorm(name='conv2_2_dw_bn', data=conv2_2_dw , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
        conv2_2_dw_scale = conv2_2_dw_bn
        relu2_2_dw = mx.symbol.Activation(name='relu2_2_dw', data=conv2_2_dw_scale , act_type='relu')

        conv2_2_sep = mx.symbol.Convolution(name='conv2_2_sep', data=relu2_2_dw , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
        conv2_2_sep_bn = mx.symbol.BatchNorm(name='conv2_2_sep_bn', data=conv2_2_sep , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
        conv2_2_sep_scale = conv2_2_sep_bn
        relu2_2_sep = mx.symbol.Activation(name='relu2_2_sep', data=conv2_2_sep_scale , act_type='relu')

        conv3_1_dw = mx.symbol.Convolution(name='conv3_1_dw', data=relu2_2_sep , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=128)
        conv3_1_dw_bn = mx.symbol.BatchNorm(name='conv3_1_dw_bn', data=conv3_1_dw , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
        conv3_1_dw_scale = conv3_1_dw_bn
        relu3_1_dw = mx.symbol.Activation(name='relu3_1_dw', data=conv3_1_dw_scale , act_type='relu')

        conv3_1_sep = mx.symbol.Convolution(name='conv3_1_sep', data=relu3_1_dw , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
        conv3_1_sep_bn = mx.symbol.BatchNorm(name='conv3_1_sep_bn', data=conv3_1_sep , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
        conv3_1_sep_scale = conv3_1_sep_bn
        relu3_1_sep = mx.symbol.Activation(name='relu3_1_sep', data=conv3_1_sep_scale , act_type='relu')

        conv3_2_dw = mx.symbol.Convolution(name='conv3_2_dw', data=relu3_1_sep , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True, num_group=128)
        conv3_2_dw_bn = mx.symbol.BatchNorm(name='conv3_2_dw_bn', data=conv3_2_dw , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
        conv3_2_dw_scale = conv3_2_dw_bn
        relu3_2_dw = mx.symbol.Activation(name='relu3_2_dw', data=conv3_2_dw_scale , act_type='relu')

        conv3_2_sep = mx.symbol.Convolution(name='conv3_2_sep', data=relu3_2_dw , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
        conv3_2_sep_bn = mx.symbol.BatchNorm(name='conv3_2_sep_bn', data=conv3_2_sep , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
        conv3_2_sep_scale = conv3_2_sep_bn
        relu3_2_sep = mx.symbol.Activation(name='relu3_2_sep', data=conv3_2_sep_scale , act_type='relu')

        conv4_1_dw = mx.symbol.Convolution(name='conv4_1_dw', data=relu3_2_sep , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=256)
        conv4_1_dw_bn = mx.symbol.BatchNorm(name='conv4_1_dw_bn', data=conv4_1_dw , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
        conv4_1_dw_scale = conv4_1_dw_bn
        relu4_1_dw = mx.symbol.Activation(name='relu4_1_dw', data=conv4_1_dw_scale , act_type='relu')

        conv4_1_sep = mx.symbol.Convolution(name='conv4_1_sep', data=relu4_1_dw , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
        conv4_1_sep_bn = mx.symbol.BatchNorm(name='conv4_1_sep_bn', data=conv4_1_sep , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
        conv4_1_sep_scale = conv4_1_sep_bn
        relu4_1_sep = mx.symbol.Activation(name='relu4_1_sep', data=conv4_1_sep_scale , act_type='relu')

        # 28x28
        conv4_2_dw = mx.symbol.Convolution(name='conv4_2_dw', data=relu4_1_sep , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True, num_group=256)
        conv4_2_dw_bn = mx.symbol.BatchNorm(name='conv4_2_dw_bn', data=conv4_2_dw , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
        conv4_2_dw_scale = conv4_2_dw_bn
        relu4_2_dw = mx.symbol.Activation(name='relu4_2_dw', data=conv4_2_dw_scale , act_type='relu')

        conv4_2_sep = mx.symbol.Convolution(name='conv4_2_sep', data=relu4_2_dw , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
        conv4_2_sep_bn = mx.symbol.BatchNorm(name='conv4_2_sep_bn', data=conv4_2_sep , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
        conv4_2_sep_scale = conv4_2_sep_bn
        relu4_2_sep = mx.symbol.Activation(name='relu4_2_sep', data=conv4_2_sep_scale , act_type='relu')

        return relu4_2_sep

    def get_stage_2(self, conv_feat):
        use_global_stats = self.use_global_stats
        conv5_1_dw = mx.symbol.Convolution(name='conv5_1_dw', data=conv_feat , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=512)
        conv5_1_dw_bn = mx.symbol.BatchNorm(name='conv5_1_dw_bn', data=conv5_1_dw , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
        conv5_1_dw_scale = conv5_1_dw_bn
        relu5_1_dw = mx.symbol.Activation(name='relu5_1_dw', data=conv5_1_dw_scale , act_type='relu')

        conv5_1_sep = mx.symbol.Convolution(name='conv5_1_sep', data=relu5_1_dw , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
        conv5_1_sep_bn = mx.symbol.BatchNorm(name='conv5_1_sep_bn', data=conv5_1_sep , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
        conv5_1_sep_scale = conv5_1_sep_bn
        relu5_1_sep = mx.symbol.Activation(name='relu5_1_sep', data=conv5_1_sep_scale , act_type='relu')

        conv5_2_dw = mx.symbol.Convolution(name='conv5_2_dw', data=relu5_1_sep , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=512)
        conv5_2_dw_bn = mx.symbol.BatchNorm(name='conv5_2_dw_bn', data=conv5_2_dw , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
        conv5_2_dw_scale = conv5_2_dw_bn
        relu5_2_dw = mx.symbol.Activation(name='relu5_2_dw', data=conv5_2_dw_scale , act_type='relu')

        conv5_2_sep = mx.symbol.Convolution(name='conv5_2_sep', data=relu5_2_dw , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
        conv5_2_sep_bn = mx.symbol.BatchNorm(name='conv5_2_sep_bn', data=conv5_2_sep , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
        conv5_2_sep_scale = conv5_2_sep_bn
        relu5_2_sep = mx.symbol.Activation(name='relu5_2_sep', data=conv5_2_sep_scale , act_type='relu')

        conv5_3_dw = mx.symbol.Convolution(name='conv5_3_dw', data=relu5_2_sep , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=512)
        conv5_3_dw_bn = mx.symbol.BatchNorm(name='conv5_3_dw_bn', data=conv5_3_dw , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
        conv5_3_dw_scale = conv5_3_dw_bn
        relu5_3_dw = mx.symbol.Activation(name='relu5_3_dw', data=conv5_3_dw_scale , act_type='relu')

        conv5_3_sep = mx.symbol.Convolution(name='conv5_3_sep', data=relu5_3_dw , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
        conv5_3_sep_bn = mx.symbol.BatchNorm(name='conv5_3_sep_bn', data=conv5_3_sep , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
        conv5_3_sep_scale = conv5_3_sep_bn
        relu5_3_sep = mx.symbol.Activation(name='relu5_3_sep', data=conv5_3_sep_scale , act_type='relu')

        conv5_4_dw = mx.symbol.Convolution(name='conv5_4_dw', data=relu5_3_sep , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=512)
        conv5_4_dw_bn = mx.symbol.BatchNorm(name='conv5_4_dw_bn', data=conv5_4_dw , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
        conv5_4_dw_scale = conv5_4_dw_bn
        relu5_4_dw = mx.symbol.Activation(name='relu5_4_dw', data=conv5_4_dw_scale , act_type='relu')

        conv5_4_sep = mx.symbol.Convolution(name='conv5_4_sep', data=relu5_4_dw , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
        conv5_4_sep_bn = mx.symbol.BatchNorm(name='conv5_4_sep_bn', data=conv5_4_sep , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
        conv5_4_sep_scale = conv5_4_sep_bn
        relu5_4_sep = mx.symbol.Activation(name='relu5_4_sep', data=conv5_4_sep_scale , act_type='relu')

        conv5_5_dw = mx.symbol.Convolution(name='conv5_5_dw', data=relu5_4_sep , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=512)
        conv5_5_dw_bn = mx.symbol.BatchNorm(name='conv5_5_dw_bn', data=conv5_5_dw , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
        conv5_5_dw_scale = conv5_5_dw_bn
        relu5_5_dw = mx.symbol.Activation(name='relu5_5_dw', data=conv5_5_dw_scale , act_type='relu')

        conv5_5_sep = mx.symbol.Convolution(name='conv5_5_sep', data=relu5_5_dw , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
        conv5_5_sep_bn = mx.symbol.BatchNorm(name='conv5_5_sep_bn', data=conv5_5_sep , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
        conv5_5_sep_scale = conv5_5_sep_bn
        relu5_5_sep = mx.symbol.Activation(name='relu5_5_sep', data=conv5_5_sep_scale , act_type='relu')

        #conv5_6_dw = mx.symbol.Convolution(name='conv5_6_dw', data=relu5_5_sep , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True, num_group=512)
        conv5_6_dw = mx.symbol.Convolution(name='conv5_6_dw', data=relu5_5_sep , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=512)
        conv5_6_dw_bn = mx.symbol.BatchNorm(name='conv5_6_dw_bn', data=conv5_6_dw , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
        conv5_6_dw_scale = conv5_6_dw_bn
        relu5_6_dw = mx.symbol.Activation(name='relu5_6_dw', data=conv5_6_dw_scale , act_type='relu')

        conv5_6_sep = mx.symbol.Convolution(name='conv5_6_sep', data=relu5_6_dw , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
        conv5_6_sep_bn = mx.symbol.BatchNorm(name='conv5_6_sep_bn', data=conv5_6_sep , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
        conv5_6_sep_scale = conv5_6_sep_bn
        relu5_6_sep = mx.symbol.Activation(name='relu5_6_sep', data=conv5_6_sep_scale , act_type='relu')

        conv6_dw = mx.symbol.Convolution(name='conv6_dw', data=relu5_6_sep , num_filter=1024, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True, num_group=1024)
        conv6_dw_bn = mx.symbol.BatchNorm(name='conv6_dw_bn', data=conv6_dw , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
        conv6_dw_scale = conv6_dw_bn
        relu6_dw = mx.symbol.Activation(name='relu6_dw', data=conv6_dw_scale , act_type='relu')

        conv6_sep = mx.symbol.Convolution(name='conv6_sep', data=relu6_dw , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
        conv6_sep_bn = mx.symbol.BatchNorm(name='conv6_sep_bn', data=conv6_sep , use_global_stats=use_global_stats, fix_gamma=False, eps=0.000100)
        conv6_sep_scale = conv6_sep_bn
        relu6_sep = mx.symbol.Activation(name='relu6_sep', data=conv6_sep_scale , act_type='relu')
        return relu6_sep

    def get_rpn(self, conv_feat, num_anchors):
        rpn_conv = mx.sym.Convolution(
            data=conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=256, name="rpn_conv_3x3")
        rpn_relu = mx.sym.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")
        return rpn_cls_score, rpn_bbox_pred

    def get_symbol(self, cfg, is_train=True):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS

        # input init
        if is_train:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")
            gt_boxes = mx.sym.Variable(name="gt_boxes")
            rpn_label = mx.sym.Variable(name='label')
            rpn_bbox_target = mx.sym.Variable(name='bbox_target')
            rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')
        else:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")

        # shared convolutional layers
        conv_feat = self.get_stage_1(data)
        # res5
        relu1 = self.get_stage_2(conv_feat)

        rpn_cls_score, rpn_bbox_pred = self.get_rpn(conv_feat, num_anchors)

        if is_train:
            # prepare rpn data
            rpn_cls_score_reshape = mx.sym.Reshape(
                data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

            # classification
            rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                                   normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob")
            # bounding box regression
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))
            rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)

            # ROI proposal
            rpn_cls_act = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
            rpn_cls_act_reshape = mx.sym.Reshape(
                data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')
            if cfg.TRAIN.CXX_PROPOSAL:
                rois = mx.contrib.sym.Proposal(
                    cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
            else:
                rois = mx.sym.Custom(
                    cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                    scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
            # ROI proposal target
            gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
            rois, label, bbox_target, bbox_weight = mx.sym.Custom(rois=rois, gt_boxes=gt_boxes_reshape,
                                                                  op_type='proposal_target',
                                                                  num_classes=num_reg_classes,
                                                                  batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                                  batch_rois=cfg.TRAIN.BATCH_ROIS,
                                                                  cfg=cPickle.dumps(cfg),
                                                                  fg_fraction=cfg.TRAIN.FG_FRACTION)
        else:
            # ROI Proposal
            rpn_cls_score_reshape = mx.sym.Reshape(
                data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
            rpn_cls_prob = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
            rpn_cls_prob_reshape = mx.sym.Reshape(
                data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
            if cfg.TEST.CXX_PROPOSAL:
                rois = mx.contrib.sym.Proposal(
                    cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                    ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
            else:
                rois = mx.sym.Custom(
                    cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                    scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)



        # light head
        conv_new_1 = mx.sym.Convolution(data=relu1, kernel=(15, 1), pad=(7, 0), num_filter=64, name="conv_new_1", lr_mult=3.0)
        relu_new_1 = mx.sym.Activation(data=conv_new_1, act_type='relu', name='relu1')
        conv_new_2 = mx.sym.Convolution(data=relu_new_1, kernel=(1, 15), pad=(0, 7), num_filter=10*7*7, name="conv_new_2", lr_mult=3.0)
        relu_new_2 = mx.sym.Activation(data=conv_new_2, act_type='relu', name='relu2')
        conv_new_3 = mx.sym.Convolution(data=relu1, kernel=(1, 15), pad=(0, 7), num_filter=64, name="conv_new_3", lr_mult=3.0)
        relu_new_3 = mx.sym.Activation(data=conv_new_3, act_type='relu', name='relu3')
        conv_new_4 = mx.sym.Convolution(data=relu_new_3, kernel=(15, 1), pad=(7, 0), num_filter=10*7*7, name="conv_new_4", lr_mult=3.0)
        relu_new_4 = mx.sym.Activation(data=conv_new_4, act_type='relu', name='relu4')
        light_head = mx.symbol.broadcast_add(name='light_head', *[relu_new_2, relu_new_4])
        roi_pool = mx.contrib.sym.PSROIPooling(name='roi_pool', data=light_head, rois=rois, group_size=7, pooled_size=7, output_dim=10, spatial_scale=0.0625)
        fc_new_1 = mx.symbol.FullyConnected(name='fc_new_1', data=roi_pool, num_hidden=2048)
        fc_new_1_relu = mx.sym.Activation(data=fc_new_1, act_type='relu', name='fc_new_1_relu')
        cls_score = mx.symbol.FullyConnected(name='cls_score', data=fc_new_1_relu, num_hidden=num_classes)
        bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=fc_new_1_relu, num_hidden=num_reg_classes * 4)

        if is_train:
            if cfg.TRAIN.ENABLE_OHEM:
                labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                               num_reg_classes=num_reg_classes, roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                               cls_score=cls_score, bbox_pred=bbox_pred, labels=label,
                                                               bbox_targets=bbox_target, bbox_weights=bbox_weight)
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem, normalization='valid', use_ignore=True, ignore_label=-1)
                bbox_loss_ = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_', scalar=2.0, data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS_OHEM)
                rcnn_label = labels_ohem
            else:
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid')
                bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=2.0, data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS)
                rcnn_label = label

            # reshape output
            rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
            bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_loss_reshape')
            group = mx.sym.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)])
        else:
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_pred_reshape')
            group = mx.sym.Group([rois, cls_prob, bbox_pred])

        self.sym = group
        return group

    def get_symbol_rpn(self, cfg, is_train=True):
        # config alias for convenient
        num_anchors = cfg.network.NUM_ANCHORS

        # input init
        if is_train:
            data = mx.sym.Variable(name="data")
            rpn_label = mx.sym.Variable(name='label')
            rpn_bbox_target = mx.sym.Variable(name='bbox_target')
            rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')
        else:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")

        # shared convolutional layers
        conv_feat = self.get_stage_1(data)
        rpn_cls_score, rpn_bbox_pred = self.get_rpn(conv_feat, num_anchors)
        if is_train:
            # prepare rpn data
            rpn_cls_score_reshape = mx.sym.Reshape(
                data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

            # classification
            rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                                normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob",
                                                grad_scale=1.0)
            # bounding box regression
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))
            rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)
            group = mx.symbol.Group([rpn_cls_prob, rpn_bbox_loss])
        else:
            # ROI Proposal
            rpn_cls_score_reshape = mx.sym.Reshape(
                data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
            rpn_cls_prob = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
            rpn_cls_prob_reshape = mx.sym.Reshape(
                data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
            if cfg.TEST.CXX_PROPOSAL:
                rois, score = mx.contrib.sym.Proposal(
                    cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois', output_score=True,
                    feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                    ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
            else:
                rois, score = mx.sym.Custom(
                    cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois', output_score=True,
                    op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                    scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
                group = mx.symbol.Group([rois, score])
        self.sym = group
        return group

    def get_symbol_rfcn(self, cfg, is_train=True):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)

        # input init
        if is_train:
            data = mx.symbol.Variable(name="data")
            rois = mx.symbol.Variable(name='rois')
            label = mx.symbol.Variable(name='label')
            bbox_target = mx.symbol.Variable(name='bbox_target')
            bbox_weight = mx.symbol.Variable(name='bbox_weight')
            # reshape input
            rois = mx.symbol.Reshape(data=rois, shape=(-1, 5), name='rois_reshape')
            label = mx.symbol.Reshape(data=label, shape=(-1,), name='label_reshape')
            bbox_target = mx.symbol.Reshape(data=bbox_target, shape=(-1, 4 * num_reg_classes), name='bbox_target_reshape')
            bbox_weight = mx.symbol.Reshape(data=bbox_weight, shape=(-1, 4 * num_reg_classes), name='bbox_weight_reshape')
        else:
            data = mx.sym.Variable(name="data")
            rois = mx.symbol.Variable(name='rois')
            # reshape input
            rois = mx.symbol.Reshape(data=rois, shape=(-1, 5), name='rois_reshape')

        # shared convolutional layers
        conv_feat = self.get_stage_1(data)
        relu1 = self.get_stage_2(conv_feat)

        # conv_new_1
        conv_new_1 = mx.sym.Convolution(data=relu1, kernel=(1, 1), num_filter=1024, name="conv_new_1", lr_mult=3.0)
        relu_new_1 = mx.sym.Activation(data=conv_new_1, act_type='relu', name='relu1')

        # rfcn_cls/rfcn_bbox
        rfcn_cls = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=7*7*num_classes, name="rfcn_cls")
        rfcn_bbox = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=7*7*4*num_reg_classes, name="rfcn_bbox")
        psroipooled_cls_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois, group_size=7, pooled_size=7,
                                                   output_dim=num_classes, spatial_scale=0.0625)
        psroipooled_loc_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois, group_size=7, pooled_size=7,
                                                   output_dim=8, spatial_scale=0.0625)
        cls_score = mx.sym.Pooling(name='ave_cls_scors_rois', data=psroipooled_cls_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        bbox_pred = mx.sym.Pooling(name='ave_bbox_pred_rois', data=psroipooled_loc_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
        bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_reg_classes))

        if is_train:
            if cfg.TRAIN.ENABLE_OHEM:
                labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                               num_reg_classes=num_reg_classes, roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                               cls_score=cls_score, bbox_pred=bbox_pred, labels=label,
                                                               bbox_targets=bbox_target, bbox_weights=bbox_weight)
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem, normalization='valid', use_ignore=True, ignore_label=-1, grad_scale=1.0)
                bbox_loss_ = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS_OHEM)
                label = labels_ohem
            else:
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid', grad_scale=1.0)
                bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS)

            # reshape output
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
            bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_loss_reshape')
            group = mx.sym.Group([cls_prob, bbox_loss, mx.sym.BlockGrad(label)]) if cfg.TRAIN.ENABLE_OHEM else mx.sym.Group([cls_prob, bbox_loss])
        else:
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_pred_reshape')
            group = mx.sym.Group([cls_prob, bbox_pred])

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
        arg_params['conv_new_2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv_new_2_weight'])
        arg_params['conv_new_2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv_new_2_bias'])
        arg_params['conv_new_3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv_new_3_weight'])
        arg_params['conv_new_3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv_new_3_bias'])
        arg_params['conv_new_4_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv_new_4_weight'])
        arg_params['conv_new_4_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv_new_4_bias'])
        arg_params['fc_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_1_weight'])
        arg_params['fc_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_1_bias'])
        arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['cls_score_weight'])
        arg_params['cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['cls_score_bias'])
        arg_params['bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bbox_pred_weight'])
        arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['bbox_pred_bias'])

        #arg_params['rfcn_cls_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rfcn_cls_weight'])
        #arg_params['rfcn_cls_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_cls_bias'])
        #arg_params['rfcn_bbox_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rfcn_bbox_weight'])
        #arg_params['rfcn_bbox_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_bbox_bias'])

    def init_weight(self, cfg, arg_params, aux_params):
        self.init_weight_rpn(cfg, arg_params, aux_params)
        self.init_weight_rfcn(cfg, arg_params, aux_params)
