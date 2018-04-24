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


class resnet_34_kps(Symbol):

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
            first_stride = (1, 1) if i in [0, 3] else (2, 2)
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
        #bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, use_global_stats=True, eps=2e-5, momentum=bn_mom, name='bn1')
        #relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
        #return relu1

    def proposal(self, cls_prob, bbox_pred, im_info, cfg, is_train):
        feature_stride = cfg.network.RPN_FEAT_STRIDE
        scales = tuple(cfg.network.ANCHOR_SCALES)
        ratios = tuple(cfg.network.ANCHOR_RATIOS)
        if is_train:
            rpn_pre_nms_top_n = cfg.TRAIN.RPN_PRE_NMS_TOP_N
            rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N
            threshold=cfg.TRAIN.RPN_NMS_THRESH
            rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE
        else:
            rpn_pre_nms_top_n = cfg.TEST.RPN_PRE_NMS_TOP_N
            rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N
            threshold=cfg.TEST.RPN_NMS_THRESH
            rpn_min_size=cfg.TEST.RPN_MIN_SIZE

        if cfg.TRAIN.CXX_PROPOSAL:
            rois = mx.contrib.sym.MultiProposal(cls_prob=cls_prob,
                                                bbox_pred=bbox_pred,
                                                im_info=im_info,
                                                name='rois',
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
                                 name='rois',
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
        rois = self.proposal(rpn_cls_act_reshape, rpn_bbox_pred, im_info, cfg, is_train)

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
        num_keypoints = cfg.dataset.NUM_KEYPOINTS
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)

        # input init
        data = mx.sym.Variable(name="data")
        im_info = mx.sym.Variable(name="im_info")

        c2, c3, c4, c5 = self.get_backbone(data)
        rois, rpn_syms = self.get_proposals(c4, im_info, cfg, is_train)

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
        conv_new_1 = mx.sym.Convolution(data=c5, kernel=(1, 1), num_filter=256, name="conv_new_1", lr_mult=3.0)
        relu_new_1 = mx.sym.Activation(data=conv_new_1, act_type='relu', name='relu_new_1')

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

        # keypoints
        conv_kp_1 = mx.sym.Convolution(data=c5, kernel=(1, 1), num_filter=256, name="conv_kp_1", lr_mult=3.0)
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
                                                   spatial_scale=0.0625)
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
                                                    spatial_scale=0.0625)
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

    def init_weight(self, cfg, arg_params, aux_params):
        self.init_weight_rpn(cfg, arg_params, aux_params)
        self.init_weight_rfcn(cfg, arg_params, aux_params)
