# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------

import cv2
import argparse
import os
import sys
import time
import logging
import pprint
import cPickle

import numpy as np

import mxnet as mx
from mxnet.io import PrefetchingIter

import _init_paths
from symbols import *
from dataset import *
from core.loader import TestLoader
from core.tester import Predictor, im_detect
from utils.load_model import load_param
from utils.create_logger import create_logger
from config.config import config, update_config
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper


def pred_eval(predictor, test_data, imdb, cfg, vis=False, thresh=1e-3, logger=None, ignore_cache=True):
    """
    wrapper for calculating offline validation for faster data analysis
    in this example, all threshold are set by hand
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffle
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: valid detection threshold
    :return:
    """

    det_file = os.path.join(imdb.result_path, imdb.name + '_detections.pkl')
    if os.path.exists(det_file) and not ignore_cache:
        with open(det_file, 'rb') as fid:
            cache_res = cPickle.load(fid)
            all_boxes = cache_res['all_boxes']
            all_keypoints = cache_res.get('all_keypoints')
        info_str = imdb.evaluate_detections(all_boxes, all_keypoints=all_keypoints)
        if logger:
            logger.info('evaluate detections: \n{}'.format(info_str))
        return

    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data]

    if not isinstance(test_data, PrefetchingIter):
        test_data = PrefetchingIter(test_data)

    nms = py_nms_wrapper(cfg.TEST.NMS)

    # limit detections to max_per_image over all classes
    max_per_image = cfg.TEST.max_per_image

    num_images = imdb.num_images
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[np.array([]) for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]
    all_keypoints = None
    if cfg.network.PREDICT_KEYPOINTS:
        all_keypoints = [[np.array([]) for _ in range(num_images)]
                         for _ in range(imdb.num_classes)]

    idx = 0
    data_time, net_time, post_time = 0.0, 0.0, 0.0
    t = time.time()
    for data_batch in test_data:
        t1 = time.time() - t
        t = time.time()

        rets = im_detect(predictor, data_batch, data_names, cfg)
        scores_all = rets[0]
        boxes_all = rets[1]
        if cfg.network.PREDICT_KEYPOINTS:
            pred_kps_all = rets[2]

        t2 = time.time() - t
        t = time.time()
        for delta, (scores, boxes) in enumerate(zip(scores_all, boxes_all)):
            if idx+delta >= num_images:
                break
            for j in range(1, imdb.num_classes):
                indexes = np.where(scores[:, j] > thresh)[0]
                cls_scores = scores[indexes, j, np.newaxis]
                cls_boxes = boxes[indexes, 4:8] if cfg.CLASS_AGNOSTIC else boxes[indexes, j * 4:(j + 1) * 4]
                cls_dets = np.hstack((cls_boxes, cls_scores))
                keep = nms(cls_dets)
                all_boxes[j][idx+delta] = cls_dets[keep, :]
                if cfg.network.PREDICT_KEYPOINTS:
                    all_keypoints[j][idx+delta] = pred_kps_all[delta][indexes, :][keep, :]

            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][idx+delta][:, -1]
                                          for j in range(1, imdb.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, imdb.num_classes):
                        keep = np.where(all_boxes[j][idx+delta][:, -1] >= image_thresh)[0]
                        all_boxes[j][idx+delta] = all_boxes[j][idx+delta][keep, :]
                        if cfg.network.PREDICT_KEYPOINTS:
                            all_keypoints[j][idx+delta] = all_keypoints[j][idx+delta][keep, :]

            if vis:
                boxes_this_image = [[]] + [all_boxes[j][idx+delta] for j in range(1, imdb.num_classes)]
                vis_all_detection(data_dict['data'].asnumpy(), boxes_this_image, imdb.classes, scales[delta], cfg)

        idx += test_data.batch_size
        t3 = time.time() - t
        t = time.time()
        msg = 'testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s'.format(idx, imdb.num_images, t1, t2, t3)
        print msg
        if logger:
            logger.info(msg)

    with open(det_file, 'wb') as f:
        cPickle.dump({'all_boxes':all_boxes, 'all_keypoints':all_keypoints}, f, protocol=cPickle.HIGHEST_PROTOCOL)

    info_str = imdb.evaluate_detections(all_boxes, all_keypoints=all_keypoints)
    if logger:
        logger.info('evaluate detections: \n{}'.format(info_str))



def test_rcnn(cfg, dataset, image_set, root_path, dataset_path,
              ctx, prefix, epoch,
              vis, ignore_cache, shuffle, has_rpn, proposal, thresh, logger=None, output_path=None):
    if not logger:
        assert False, 'require a logger'

    # print cfg
    pprint.pprint(cfg)
    logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))

    # load symbol and testing data
    if has_rpn:
        sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
        sym = sym_instance.get_symbol(cfg, is_train=False)
        imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
        roidb = imdb.gt_roidb()
    else:
        sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
        sym = sym_instance.get_symbol_rfcn(cfg, is_train=False)
        imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
        gt_roidb = imdb.gt_roidb()
        roidb = eval('imdb.' + proposal + '_roidb')(gt_roidb)

    # get test data iter
    test_data = TestLoader(roidb, cfg, batch_size=len(ctx)*cfg.TEST.IMAGES_PER_GPU, shuffle=shuffle, has_rpn=has_rpn)

    # load model
    arg_params, aux_params = load_param(prefix, epoch, process=True)

    # infer shape
    data_shape_dict = dict(test_data.provide_data)
    sym_instance.infer_shape(data_shape_dict)
    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

    # decide maximum shape
    data_names = test_data.data_names
    label_names = None

    # create predictor
    predictor = Predictor(sym, data_names, label_names, context=ctx,
                          provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    # start detection
    pred_eval(predictor, test_data, imdb, cfg, vis=vis, ignore_cache=ignore_cache, thresh=thresh, logger=logger)


def parse_args():
    parser = argparse.ArgumentParser(description='Test a R-FCN network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    # rcnn
    parser.add_argument('--vis', help='turn on visualization', action='store_true')
    parser.add_argument('--ignore_cache', help='ignore cached results boxes', action='store_true')
    parser.add_argument('--thresh', help='valid detection threshold', default=1e-3, type=float)
    parser.add_argument('--shuffle', help='shuffle data on visualization', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print args

    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]

    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.test_image_set)

    test_rcnn(config, config.dataset.dataset, config.dataset.test_image_set, config.dataset.root_path, config.dataset.dataset_path,
              ctx, os.path.join(final_output_path, '..', '_'.join([iset for iset in config.dataset.image_set.split('+')]), config.TRAIN.model_prefix), config.TEST.test_epoch,
              args.vis, args.ignore_cache, args.shuffle, config.TEST.HAS_RPN, config.dataset.proposal, args.thresh, logger=logger, output_path=final_output_path)


if __name__ == '__main__':
    main()
