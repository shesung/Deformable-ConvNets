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


def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=256,
                  memonger=False, use_global_stats=True, use_dilated=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    dilate = (2,2) if use_dilated else ()
    pad_dilate = (2,2) if use_dilated else (1,1)
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch
        # notes, a bit difference with origin paper
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, use_global_stats=use_global_stats, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, use_global_stats=use_global_stats, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=pad_dilate, dilate=dilate,
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, use_global_stats=use_global_stats, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, use_global_stats=use_global_stats, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, use_global_stats=use_global_stats, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=pad_dilate, dilate=dilate,
                                      no_bias=True, workspace=workspace, name=name + '_conv2', cudnn_off=use_dilated)
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut


class resnet_light(Symbol):

    def get_backbone(self, data, cfg):
        self.memonger = True
        self.bn_mom = 0.9

        self.workspace = 512
        self.use_dilated_in_stage5 = False
        self.strides = [1, 2, 2, 1]

        num_layers = cfg.network.NUM_LAYERS
        if num_layers >= 50:
            self.filter_list = [64, 256, 512, 1024, 2048]
            self.bottle_neck = True
        else:
            self.filter_list = [64, 64, 128, 256, 512]
            self.bottle_neck = False

        if num_layers == 18:
            self.units = [2, 2, 2, 2]
        elif num_layers == 34:
            self.units = [3, 4, 6, 3]
        elif num_layers == 50:
            self.units = [3, 4, 6, 3]
        elif num_layers == 101:
            self.units = [3, 4, 23, 3]
        elif num_layers == 152:
            self.units = [3, 8, 36, 3]
        elif num_layers == 200:
            self.units = [3, 24, 36, 3]
        elif num_layers == 269:
            self.units = [3, 30, 48, 8]
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

        data = mx.sym.BatchNorm(data=data, fix_gamma=True, use_global_stats=True, eps=2e-5,
                                momentum=self.bn_mom, name='bn_data')
        body = mx.sym.Convolution(data=data, num_filter=self.filter_list[0], kernel=(7, 7), stride=(2,2),
                                  pad=(3, 3),no_bias=True, name="conv0", workspace=self.workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, use_global_stats=True, eps=2e-5,
                                momentum=self.bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

        output_layers = []
        num_stages = 4
        for i in range(num_stages):
            use_dilated = (i == 3 and self.use_dilated_in_stage5)
            body = residual_unit(body,
                                 num_filter  = self.filter_list[i+1],
                                 stride      = (self.strides[i], self.strides[i]),
                                 dim_match   = False,
                                 use_dilated = use_dilated,
                                 name        = 'stage%d_unit%d' % (i + 1, 1),
                                 bottle_neck = self.bottle_neck,
                                 workspace   = self.workspace,
                                 memonger    = self.memonger)
            for j in range(self.units[i]-1):
                body = residual_unit(body,
                                     num_filter  = self.filter_list[i+1],
                                     stride      = (1,1),
                                     dim_match   = True,
                                     use_dilated = use_dilated,
                                     name        = 'stage%d_unit%d' % (i + 1, j + 2),
                                     bottle_neck = self.bottle_neck,
                                     workspace   = self.workspace,
                                     memonger    = self.memonger)
            output_layers.append(body)
        return output_layers



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
            gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
            rois, label, bbox_target, bbox_weight = mx.sym.Custom(rois=rois,
                                                                  gt_boxes=gt_boxes_reshape,
                                                                  op_type='proposal_target',
                                                                  num_classes=num_reg_classes,
                                                                  batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                                  batch_rois=cfg.TRAIN.BATCH_ROIS,
                                                                  cfg=cPickle.dumps(cfg),
                                                                  fg_fraction=cfg.TRAIN.FG_FRACTION)

        # light head
        conv_feat = c5
        feat_stride = 16
        conv_new_1 = mx.sym.Convolution(data=conv_feat, kernel=(15, 1), pad=(7, 0), num_filter=64,
                                        name="conv_new_1", lr_mult=3.0)
        relu_new_1 = mx.sym.Activation(data=conv_new_1, act_type='relu', name='relu_new_1')
        conv_new_2 = mx.sym.Convolution(data=relu_new_1, kernel=(1, 15), pad=(0, 7), num_filter=10*7*7,
                                        name="conv_new_2", lr_mult=3.0)
        relu_new_2 = mx.sym.Activation(data=conv_new_2, act_type='relu', name='relu_new_2')
        conv_new_3 = mx.sym.Convolution(data=conv_feat, kernel=(1, 15), pad=(0, 7), num_filter=64,
                                        name="conv_new_3", lr_mult=3.0)
        relu_new_3 = mx.sym.Activation(data=conv_new_3, act_type='relu', name='relu_new_3')
        conv_new_4 = mx.sym.Convolution(data=relu_new_3, kernel=(15, 1), pad=(7, 0), num_filter=10*7*7,
                                        name="conv_new_4", lr_mult=3.0)
        relu_new_4 = mx.sym.Activation(data=conv_new_4, act_type='relu', name='relu_new_4')
        light_head = mx.symbol.broadcast_add(name='light_head', *[relu_new_2, relu_new_4])

        roi_pool = mx.contrib.sym.PSROIPooling(name='roi_pool', data=light_head, rois=rois, group_size=7,
                                               pooled_size=7, output_dim=10, spatial_scale=1.0/feat_stride)

        fc_new_1 = mx.symbol.FullyConnected(name='fc_new_1', data=roi_pool, num_hidden=2048)
        fc_new_1_relu = mx.sym.Activation(data=fc_new_1, act_type='relu', name='fc_new_1_relu')
        cls_score = mx.symbol.FullyConnected(name='cls_score', data=fc_new_1_relu, num_hidden=num_classes)
        bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=fc_new_1_relu, num_hidden=num_reg_classes*4)

        if is_train:
            if cfg.TRAIN.ENABLE_OHEM:
                labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM',
                                                               num_classes=num_classes,
                                                               num_reg_classes=num_reg_classes,
                                                               roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
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
                                            grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS_OHEM)
                rcnn_label = labels_ohem
            else:
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label,
                                                normalization='valid')
                bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0,
                                                            data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_,
                                            grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS)
                rcnn_label = label

            # reshape output
            rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1),
                                        name='label_reshape')
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_loss_reshape')
            group = mx.sym.Group(rpn_syms + [cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)])
        else:
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes),
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


    def init_weight_light(self, cfg, arg_params, aux_params):
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


    def init_weight(self, cfg, arg_params, aux_params):
        self.init_weight_rpn(cfg, arg_params, aux_params)
        self.init_weight_light(cfg, arg_params, aux_params)
