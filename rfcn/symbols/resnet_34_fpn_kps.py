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


def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False, use_global_stats=True, use_dilated=False):
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
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
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


class resnet_34_fpn_kps(Symbol):

    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.use_global_stats = True
        self.workspace = 512
        self.units = [3, 4, 6, 3]
        self.filter_list = [64, 64, 128, 256, 512]
        self.bottle_neck = False
        self.use_dilated = False

    def get_backbone(self, data):
        memonger = True
        bn_mom = 0.9

        data = mx.sym.BatchNorm(data=data, fix_gamma=True, use_global_stats=True, eps=2e-5, momentum=bn_mom, name='bn_data')
        body = mx.sym.Convolution(data=data, num_filter=self.filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=self.workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, use_global_stats=True, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

        output_layers = []
        for i in [0, 1, 2, 3]:
            use_dilated = self.use_dilated if i == 3 else False
            first_stride = (1, 1) if i == 0 else (2, 2)
            body = residual_unit(body,
                                 num_filter=self.filter_list[i+1],
                                 stride=first_stride,
                                 dim_match=False,
                                 use_dilated=use_dilated,
                                 name='stage%d_unit%d' % (i + 1, 1),
                                 bottle_neck=self.bottle_neck,
                                 workspace=self.workspace,
                                 memonger=memonger)
            for j in range(self.units[i]-1):
                body = residual_unit(body,
                                     num_filter=self.filter_list[i+1],
                                     stride=(1,1),
                                     dim_match=True,
                                     use_dilated=use_dilated,
                                     name='stage%d_unit%d' % (i + 1, j + 2),
                                     bottle_neck=self.bottle_neck,
                                     workspace=self.workspace,
                                     memonger=memonger)
            output_layers.append(body)
        return output_layers

    def get_fpn_feature(self, feat_list, feature_dim=256):
        c2, c3, c4, c5 = feat_list
        # lateral connection
        fpn_p5_1x1 = mx.symbol.Convolution(data=c5, kernel=(1, 1), pad=(0, 0), stride=(1, 1), num_filter=feature_dim, name='fpn_p5_1x1')
        fpn_p4_1x1 = mx.symbol.Convolution(data=c4, kernel=(1, 1), pad=(0, 0), stride=(1, 1), num_filter=feature_dim, name='fpn_p4_1x1')
        fpn_p3_1x1 = mx.symbol.Convolution(data=c3, kernel=(1, 1), pad=(0, 0), stride=(1, 1), num_filter=feature_dim, name='fpn_p3_1x1')
        fpn_p2_1x1 = mx.symbol.Convolution(data=c2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), num_filter=feature_dim, name='fpn_p2_1x1')
        # top-down connection
        fpn_p5_upsample = mx.symbol.UpSampling(fpn_p5_1x1, scale=2, sample_type='nearest', name='fpn_p5_upsample')
        fpn_p4_plus = mx.sym.ElementWiseSum(*[fpn_p5_upsample, fpn_p4_1x1], name='fpn_p4_sum')
        fpn_p4_upsample = mx.symbol.UpSampling(fpn_p4_plus, scale=2, sample_type='nearest', name='fpn_p4_upsample')
        fpn_p3_plus = mx.sym.ElementWiseSum(*[fpn_p4_upsample, fpn_p3_1x1], name='fpn_p3_sum')
        fpn_p3_upsample = mx.symbol.UpSampling(fpn_p3_plus, scale=2, sample_type='nearest', name='fpn_p3_upsample')
        fpn_p2_plus = mx.sym.ElementWiseSum(*[fpn_p3_upsample, fpn_p2_1x1], name='fpn_p2_sum')
        # FPN feature
        fpn_p6 = mx.sym.Convolution(data=c5, kernel=(3, 3), pad=(1, 1), stride=(2, 2), num_filter=feature_dim, name='fpn_p6')
        fpn_p5 = mx.symbol.Convolution(data=fpn_p5_1x1, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=feature_dim, name='fpn_p5')
        fpn_p4 = mx.symbol.Convolution(data=fpn_p4_plus, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=feature_dim, name='fpn_p4')
        fpn_p3 = mx.symbol.Convolution(data=fpn_p3_plus, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=feature_dim, name='fpn_p3')
        fpn_p2 = mx.symbol.Convolution(data=fpn_p2_plus, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=feature_dim, name='fpn_p2')
        return fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6

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

    def get_proposals(self, data, im_info, cfg, is_train=True):
        num_anchors = cfg.network.NUM_ANCHORS
        rpn_conv = mx.sym.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=256, name="rpn_conv_3x3")
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

    def get_multilayer_proposals(self, layers, im_info, cfg, is_train=True):
        num_layers = len(layers)
        all_rois = []
        all_rpn_cls_score = []
        all_rpn_bbox_pred = []
        for l in range(num_layers):
            prefix = "rpn%d" % l
            num_anchors = cfg.network.NUM_ANCHORS
            rpn_conv = mx.sym.Convolution(data=layers[l], kernel=(3, 3), pad=(1, 1), num_filter=256,
                                          name=prefix+"_conv_3x3")
            rpn_relu = mx.sym.Activation(data=rpn_conv, act_type="relu", name=prefix+"_relu")
            rpn_cls_score = mx.sym.Convolution(data=rpn_relu, kernel=(1, 1), pad=(0, 0),
                                               num_filter=2 * num_anchors, name=prefix+"_cls_score")
            rpn_cls_score_reshape = mx.sym.Reshape(data=rpn_cls_score,
                                                   shape=(0, 2, -1),
                                                   name=prefix+"_cls_score_reshape")
            rpn_bbox_pred = mx.sym.Convolution(data=rpn_relu, kernel=(1, 1), pad=(0, 0),
                                               num_filter=4 * num_anchors, name=prefix+"_bbox_pred")
            rpn_bbox_pred_reshape = mx.sym.Reshape(data=rpn_bbox_pred,
                                                   shape=(0, 4*num_anchors, -1),
                                                   name=prefix+"_bbox_pred_reshape")

            # ROI proposal
            act_cls_score_reshape = mx.sym.Reshape(data=rpn_cls_score,
                                                   shape=(0, 2, -1, 0),
                                                   name=prefix+"_act_score_reshape")
            rpn_cls_act = mx.sym.SoftmaxActivation(data=act_cls_score_reshape,
                                                   mode="channel", name=prefix+"_cls_act")
            rpn_cls_act_reshape = mx.sym.Reshape(data=rpn_cls_act,
                                                 shape=(0, 2 * num_anchors, -1, 0),
                                                 name=prefix+"_cls_act_reshape")

            rpn_cfg = cfg.TRAIN if is_train else cfg.TEST
            feature_stride     = cfg.network.MULTI_RPN_STRIDES[l]
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
                                 prefix, cxx_proposal)
            all_rois.append(rois)
            all_rpn_cls_score.append(rpn_cls_score_reshape)
            all_rpn_bbox_pred.append(rpn_bbox_pred_reshape)

        rois = mx.sym.Concat(*all_rois, dim=0, name='rois')
        ret_syms = []
        if is_train:
            rpn_label = mx.sym.Variable(name='label')
            rpn_bbox_target = mx.sym.Variable(name='bbox_target')
            rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')

            # rpn classification
            rpn_cls_score = mx.sym.Concat(*all_rpn_cls_score, dim=2, name='rpn_cls_score')
            rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score, label=rpn_label,
                                                multi_output=True,normalization='valid', use_ignore=True,
                                                ignore_label=-1, name="rpn_cls_prob")
            # rpn bounding box regression
            rpn_bbox_pred = mx.sym.Concat(*all_rpn_bbox_pred, dim=2, name='rpn_bbox_pred')
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=3.0,
                                                                data=(rpn_bbox_pred - rpn_bbox_target))
            rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_,
                                            grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)
            ret_syms = [rpn_cls_prob, rpn_bbox_loss]
        return rois, ret_syms


    def get_symbol(self, cfg, is_train=True):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_keypoints = cfg.dataset.NUM_KEYPOINTS
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)

        # input init
        data = mx.sym.Variable(name="data")
        im_info = mx.sym.Variable(name="im_info")

        c2, c3, c4, c5 = self.get_backbone(data)
        p2, p3, p4, p5, p6 = self.get_fpn_feature([c2,c3,c4,c5], feature_dim=256)
        rois, rpn_syms = self.get_multilayer_proposals([p2, p3, p4, p5, p6], im_info, cfg, is_train)

        if is_train:
            gt_boxes = mx.sym.Variable(name="gt_boxes")
            gt_kps = mx.sym.Variable(name="gt_kps")

            # ROI proposal target
            gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
            gt_kps_reshape = mx.sym.Reshape(data=gt_kps, shape=(-1, num_keypoints*3), name='gt_kps_reshape')
            rois, label, bbox_target, bbox_weight, \
                    kps_label, kps_pos_target, kps_pos_weight = mx.sym.Custom(rois=rois,
                                                                  gt_boxes=gt_boxes_reshape,
                                                                  gt_kps = gt_kps_reshape,
                                                                  op_type='proposal_target',
                                                                  num_classes=num_reg_classes,
                                                                  batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                                  batch_rois=cfg.TRAIN.BATCH_ROIS,
                                                                  cfg=cPickle.dumps(cfg),
                                                                  fg_fraction=cfg.TRAIN.FG_FRACTION)

        # conv_new_1
        spatial_scale = 1.0 / 4
        conv_new_1 = mx.sym.Convolution(data=p2, kernel=(1, 1), num_filter=256, name="conv_new_1", lr_mult=3.0)
        relu_new_1 = mx.sym.Activation(data=conv_new_1, act_type='relu', name='relu_new_1')

        # rfcn_cls/rfcn_bbox
        rfcn_cls = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=7*7*num_classes, name="rfcn_cls")
        rfcn_bbox = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=7*7*4*num_reg_classes, name="rfcn_bbox")
        psroipooled_cls_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois, group_size=7, pooled_size=7,
                                                   output_dim=num_classes, spatial_scale=spatial_scale)
        psroipooled_loc_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois, group_size=7, pooled_size=7,
                                                   output_dim=8, spatial_scale=spatial_scale)
        cls_score = mx.sym.Pooling(name='ave_cls_scors_rois', data=psroipooled_cls_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        bbox_pred = mx.sym.Pooling(name='ave_bbox_pred_rois', data=psroipooled_loc_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
        bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_reg_classes))

        # keypoints
        kp_spatial_scale = 1.0 / 4
        conv_kp_1 = mx.sym.Convolution(data=p2, kernel=(1, 1), num_filter=256, name="conv_kp_1", lr_mult=3.0)
        relu_kp_1 = mx.sym.Activation(data=conv_kp_1, act_type='relu', name='relu_kp_1')

        group_size = 1
        group_size2 = group_size * group_size
        pooled_size = cfg.network.KEYPOINTS_POOLED_SIZE
        pooled_size2 = pooled_size * pooled_size

        rfcn_kps_pos = mx.sym.Convolution(name="rfcn_kps_pos", data=relu_kp_1,
                                          kernel=(1, 1), num_filter=group_size2*2*num_keypoints,
                                          lr_mult=1.0)
        kps_pos_pred = mx.contrib.sym.PSROIPooling(name='psroipooled_kps_pos', data=rfcn_kps_pos, rois=rois,
                                                   group_size=group_size,
                                                   pooled_size=pooled_size,
                                                   output_dim=2*num_keypoints,
                                                   spatial_scale=kp_spatial_scale)
        kps_pos_pred = mx.sym.Reshape(name='kps_pos_pred_reshape', data=kps_pos_pred,
                                      shape=(-1,2*num_keypoints, pooled_size, pooled_size))
        # keypoints mask
        rfcn_kps_mask = mx.sym.Convolution(name="rfcn_kps_mask", data=relu_kp_1,
                                           kernel=(1, 1), num_filter=group_size2*num_keypoints,
                                           lr_mult=1.0)
        kps_mask_pred = mx.contrib.sym.PSROIPooling(name='psroipooled_kps_mask', data=rfcn_kps_mask, rois=rois,
                                                    group_size=group_size,
                                                    pooled_size=pooled_size,
                                                    output_dim=num_keypoints,
                                                    spatial_scale=kp_spatial_scale)
        kps_mask_pred = mx.sym.Reshape(name='kps_mask_pred_reshape', data=kps_mask_pred,
                                       shape=(-1, pooled_size2))

        if is_train:
            if cfg.TRAIN.ENABLE_OHEM:
                labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                               num_reg_classes=num_reg_classes, roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                               cls_score=cls_score, bbox_pred=bbox_pred, labels=label,
                                                               bbox_targets=bbox_target, bbox_weights=bbox_weight)
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem, normalization='valid', use_ignore=True, ignore_label=-1)
                bbox_loss_ = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS_OHEM)
                rcnn_label = labels_ohem
            else:
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid')
                bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS)
                rcnn_label = label

            # reshape output
            rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
            bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_loss_reshape')

            # keypoints loss
            kps_prob = mx.sym.SoftmaxOutput(name='kps_prob', data=kps_mask_pred, label=kps_label,
                                            normalization='valid',
                                            use_ignore=True,
                                            ignore_label=-1,
                                            grad_scale=cfg.TRAIN.KEYPOINT_LOSS_WEIGHTS[0])
            kps_pos_loss_ = kps_pos_weight * mx.sym.smooth_l1(name='kps_pos_loss_', scalar=1.0,
                                                              data=(kps_pos_pred - kps_pos_target))
            kps_pos_loss = mx.sym.MakeLoss(name='kps_pos_loss', data=kps_pos_loss_,
                                           grad_scale=cfg.TRAIN.KEYPOINT_LOSS_WEIGHTS[1])

            group = mx.sym.Group(rpn_syms + [cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label),
                                             kps_pos_loss, kps_prob, mx.sym.BlockGrad(kps_label)])

        else: # testing
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_pred_reshape')
            # keypoints
            kps_prob = mx.sym.SoftmaxActivation(name='kps_prob', data=kps_mask_pred)
            group = mx.sym.Group([rois, cls_prob, bbox_pred, kps_prob, kps_pos_pred])

        self.sym = group
        return group

    def init_weight_rpn(self, cfg, arg_params, aux_params):
        arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_conv_3x3_weight'])
        arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_conv_3x3_bias'])
        arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_cls_score_weight'])
        arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score_bias'])
        arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_bbox_pred_weight'])
        arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred_bias'])

    def init_weight_multi_rpn(self, cfg, arg_params, aux_params):
        num_layers = len(cfg.network.MULTI_RPN_STRIDES)
        for l in range(num_layers):
            prefix = 'rpn%d' % l
            arg_params[prefix+'_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[prefix+'_conv_3x3_weight'])
            arg_params[prefix+'_conv_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict[prefix+'_conv_3x3_bias'])
            arg_params[prefix+'_cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[prefix+'_cls_score_weight'])
            arg_params[prefix+'_cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict[prefix+'_cls_score_bias'])
            arg_params[prefix+'_bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[prefix+'_bbox_pred_weight'])
            arg_params[prefix+'_bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict[prefix+'_bbox_pred_bias'])

    def init_weight_rfcn(self, cfg, arg_params, aux_params):
        arg_params['conv_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv_new_1_weight'])
        arg_params['conv_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv_new_1_bias'])
        arg_params['rfcn_cls_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rfcn_cls_weight'])
        arg_params['rfcn_cls_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_cls_bias'])
        arg_params['rfcn_bbox_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rfcn_bbox_weight'])
        arg_params['rfcn_bbox_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_bbox_bias'])

        arg_params['conv_kp_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv_kp_1_weight'])
        arg_params['conv_kp_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv_kp_1_bias'])
        arg_params['rfcn_kps_pos_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rfcn_kps_pos_weight'])
        arg_params['rfcn_kps_pos_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_kps_pos_bias'])
        arg_params['rfcn_kps_mask_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rfcn_kps_mask_weight'])
        arg_params['rfcn_kps_mask_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_kps_mask_bias'])

        if 'kp_deconv_1_weight' in self.arg_shape_dict:
            print self.arg_shape_dict['kp_deconv_1_weight'] ###
            arg_params['kp_deconv_1_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['kp_deconv_1_weight'])
            self.init_upsampling(arg_params['kp_deconv_1_weight'])

    def init_weight_fpn(self, cfg, arg_params, aux_params):
        arg_params['fpn_p6_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p6_weight'])
        arg_params['fpn_p6_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p6_bias'])
        arg_params['fpn_p5_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p5_weight'])
        arg_params['fpn_p5_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p5_bias'])
        arg_params['fpn_p4_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p4_weight'])
        arg_params['fpn_p4_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p4_bias'])
        arg_params['fpn_p3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p3_weight'])
        arg_params['fpn_p3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p3_bias'])
        arg_params['fpn_p2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p2_weight'])
        arg_params['fpn_p2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p2_bias'])

        arg_params['fpn_p5_1x1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p5_1x1_weight'])
        arg_params['fpn_p5_1x1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p5_1x1_bias'])
        arg_params['fpn_p4_1x1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p4_1x1_weight'])
        arg_params['fpn_p4_1x1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p4_1x1_bias'])
        arg_params['fpn_p3_1x1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p3_1x1_weight'])
        arg_params['fpn_p3_1x1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p3_1x1_bias'])
        arg_params['fpn_p2_1x1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p2_1x1_weight'])
        arg_params['fpn_p2_1x1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p2_1x1_bias'])

    def init_upsampling(self, arr):
        weight = np.zeros(np.prod(arr.shape), dtype='float32')
        shape = arr.shape
        f = np.ceil(shape[3] / 2.)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(np.prod(shape)):
            x = i % shape[3]
            y = (i / shape[3]) % shape[2]
            weight[i] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        arr[:] = weight.reshape(shape)

    def init_weight(self, cfg, arg_params, aux_params):
        self.init_weight_multi_rpn(cfg, arg_params, aux_params)
        self.init_weight_rfcn(cfg, arg_params, aux_params)
        self.init_weight_fpn(cfg, arg_params, aux_params)
