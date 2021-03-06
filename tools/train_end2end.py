# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------

import time
import argparse
import logging
import pprint
import sys
import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

import _init_paths
from config.config import config, update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train R-FCN network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent', help='frequency of logging', default=config.TRAIN.frequent, type=int)
    args = parser.parse_args()
    return args

args = parse_args()

import shutil
import numpy as np
import mxnet as mx

from symbols import *
from core import callback, metric
from core.loader import DummyAnchorLoader, AnchorLoader, PyramidAnchorLoader
from core.mutable_module import MutableModule
from utils.create_logger import create_logger
from utils.load_data import load_gt_roidb, merge_roidb, filter_roidb
from utils.load_model import load_param
from mxnet.io import PrefetchingIter
from utils.lr_scheduler import WarmupMultiFactorScheduler


def train_net(args, ctx, pretrained, epoch, prefix, begin_epoch, end_epoch, lr, lr_step):
    if config.dataset.dataset != 'JSONList':
        logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)
        prefix = os.path.join(final_output_path, prefix)
        shutil.copy2(args.cfg, prefix+'.yaml')
    else:
        import datetime
        import logging
        final_output_path = config.output_path
        prefix = prefix + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        prefix = os.path.join(final_output_path, prefix)
        shutil.copy2(args.cfg, prefix+'.yaml')
        log_file = prefix + '.log'
        head = '%(asctime)-15s %(message)s'
        logging.basicConfig(filename=log_file, format=head)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.info('prefix: %s' % prefix)
        print('prefix: %s' % prefix)

    # load symbol
    #shutil.copy2(os.path.join(curr_path, '..', 'symbols', config.symbol + '.py'), final_output_path)
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=True)

    # setup multi-gpu
    batch_size = config.TRAIN.IMAGES_PER_GPU * len(ctx)

    # print config
    pprint.pprint(config)
    logger.info('training config:{}\n'.format(pprint.pformat(config)))

    # load dataset and prepare imdb for training
    image_sets = [iset for iset in config.dataset.image_set.split('+')]
    roidbs = [load_gt_roidb(config.dataset.dataset, image_set, config.dataset.root_path, config.dataset.dataset_path,
                            flip=config.TRAIN.FLIP)
              for image_set in image_sets]
    roidb = merge_roidb(roidbs)
    roidb = filter_roidb(roidb, config)
    # load training data
    if config.network.MULTI_RPN:
        assert Fasle, 'still developing' ###
        num_layers = len(config.network.MULTI_RPN_STRIDES)
        rpn_syms = [sym.get_internals()['rpn%d_cls_score_output'% l] for l in range(num_layers)]
        train_data = PyramidAnchorLoader(rpn_syms, roidb, config, batch_size=batch_size, shuffle=config.TRAIN.SHUFFLE,
                                         ctx=ctx, feat_strides=config.network.MULTI_RPN_STRIDES, anchor_scales=config.network.ANCHOR_SCALES,
                                         anchor_ratios=config.network.ANCHOR_RATIOS, aspect_grouping=config.TRAIN.ASPECT_GROUPING,
                                         allowed_border=np.inf)
    else:
        feat_sym = sym.get_internals()['rpn_cls_score_output']
        train_data = AnchorLoader(feat_sym, roidb, config, batch_size=batch_size, shuffle=config.TRAIN.SHUFFLE, ctx=ctx,
                                  feat_stride=config.network.RPN_FEAT_STRIDE, anchor_scales=config.network.ANCHOR_SCALES,
                                  anchor_ratios=config.network.ANCHOR_RATIOS, aspect_grouping=config.TRAIN.ASPECT_GROUPING)

    # infer max shape
    data_shape_dict = dict(train_data.provide_data + train_data.provide_label)
    pprint.pprint(data_shape_dict)
    sym_instance.infer_shape(data_shape_dict)

    # load and initialize params
    if config.TRAIN.RESUME:
        print('continue training from ', begin_epoch)
        arg_params, aux_params = load_param(prefix, begin_epoch, convert=True)
    else:
        arg_params, aux_params = load_param(pretrained, epoch, convert=True)
        sym_instance.init_weight(config, arg_params, aux_params)

    # check parameter shapes
    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict)

    # create solver
    fixed_param_prefix = config.network.FIXED_PARAMS
    mod = MutableModule(sym,
                        train_data.data_names,
                        train_data.label_names,
                        context=ctx,
                        logger=logger,
                        fixed_param_prefix=fixed_param_prefix)

    if config.TRAIN.RESUME:
        mod._preload_opt_states = '%s-%04d.states'%(prefix, begin_epoch)

    # decide training params
    # metric
    rpn_eval_metric = metric.RPNAccMetric()
    rpn_cls_metric = metric.RPNLogLossMetric()
    rpn_bbox_metric = metric.RPNL1LossMetric()
    eval_metric = metric.RCNNAccMetric(config)
    cls_metric = metric.RCNNLogLossMetric(config)
    bbox_metric = metric.RCNNL1LossMetric(config)
    eval_metrics = mx.metric.CompositeEvalMetric()
    # rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, eval_metric, cls_metric, bbox_metric
    for child_metric in [rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, eval_metric, cls_metric, bbox_metric]:
        eval_metrics.add(child_metric)
    if config.network.PREDICT_KEYPOINTS:
        kps_cls_acc = metric.KeypointAccMetric(config)
        kps_cls_loss = metric.KeypointLogLossMetric(config)
        kps_pos_loss = metric.KeypointL1LossMetric(config)
        eval_metrics.add(kps_cls_acc)
        eval_metrics.add(kps_cls_loss)
        eval_metrics.add(kps_pos_loss)

    # callback
    batch_end_callback = callback.Speedometer(batch_size, frequent=args.frequent)
    means = np.tile(np.array(config.TRAIN.BBOX_MEANS), 2 if config.CLASS_AGNOSTIC else config.dataset.NUM_CLASSES)
    stds = np.tile(np.array(config.TRAIN.BBOX_STDS), 2 if config.CLASS_AGNOSTIC else config.dataset.NUM_CLASSES)
    epoch_end_callback = [mx.callback.do_checkpoint(prefix)]
    # decide learning rate
    base_lr = lr
    lr_factor = config.TRAIN.lr_factor
    lr_epoch = [float(epoch) for epoch in lr_step.split(',')]
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(roidb) / batch_size) for epoch in lr_epoch_diff]
    print('lr', lr, 'lr_epoch_diff', lr_epoch_diff, 'lr_iters', lr_iters)
    lr_scheduler = WarmupMultiFactorScheduler(lr_iters, lr_factor, config.TRAIN.warmup, config.TRAIN.warmup_lr, config.TRAIN.warmup_step)
    # optimizer
    optimizer_params = {'momentum': config.TRAIN.momentum,
                        'wd': config.TRAIN.wd,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': 1.0,
                        'clip_gradient': None}

    if not isinstance(train_data, PrefetchingIter):
        train_data = PrefetchingIter(train_data)

    # train
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore=config.TRAIN.kvstore,
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch)
    time.sleep(10)
    train_data.iters[0].terminate()

def main():
    print('Called with argument:', args)
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    train_net(args, ctx, config.network.pretrained, config.network.pretrained_epoch, config.TRAIN.model_prefix,
              config.TRAIN.begin_epoch, config.TRAIN.end_epoch, config.TRAIN.lr, config.TRAIN.lr_step)

if __name__ == '__main__':
    main()
