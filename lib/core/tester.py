# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------

import cPickle
import os
import time
import mxnet as mx
import numpy as np

from mutable_module import MutableModule
from utils import image
from bbox.bbox_transform import bbox_pred, clip_boxes
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper


class Predictor(object):
    def __init__(self, symbol, data_names, label_names,
                 context=mx.cpu(), max_data_shapes=None,
                 provide_data=None, provide_label=None,
                 arg_params=None, aux_params=None):
        self._mod = MutableModule(symbol, data_names, label_names,
                                  context=context, max_data_shapes=max_data_shapes)
        self._mod.bind(provide_data, provide_label, for_training=False)
        self._mod.init_params(arg_params=arg_params, aux_params=aux_params)

        # warm up
        shape_dict  = dict(provide_data)
        max_size = max(shape_dict['data'])
        shape_dict['data'] = (shape_dict['data'][0], shape_dict['data'][1], max_size, max_size)
        data = [mx.nd.zeros(shape_dict[k]) for k in data_names]
        for j in xrange(2):
            data_batch = mx.io.DataBatch(data=data, label=[], pad=0, index=0,
                                         provide_data=[(k, shape_dict[k]) for k in data_names],
                                         provide_label=[])
            self._mod.forward(data_batch)

    def predict(self, data_batch):
        self._mod.forward(data_batch)
        num_ctx = len(self._mod._context)
        outputs = self._mod.get_outputs(merge_multi_context=False)
        return [dict(zip(self._mod.output_names, [out[i] for out in outputs])) for i in range(num_ctx)]


def im_proposal(predictor, data_batch, data_names, scales):
    output_all = predictor.predict(data_batch)

    data_dict_all = [dict(zip(data_names, data_batch.data[i])) for i in xrange(len(data_batch.data))]
    scores_all = []
    boxes_all = []

    for output, data_dict, scale in zip(output_all, data_dict_all, scales):
        # drop the batch index
        boxes = output['rois_output'].asnumpy()[:, 1:]
        scores = output['rois_score'].asnumpy()

        # transform to original scale
        boxes = boxes / scale
        scores_all.append(scores)
        boxes_all.append(boxes)

    return scores_all, boxes_all, data_dict_all


def generate_proposals(predictor, test_data, imdb, cfg, vis=False, thresh=0.):
    """
    Generate detections results using RPN.
    :param predictor: Predictor
    :param test_data: data iterator, must be non-shuffled
    :param imdb: image database
    :param vis: controls visualization
    :param thresh: thresh for valid detections
    :return: list of detected boxes
    """
    assert vis or not test_data.shuffle
    data_names = [k[0] for k in test_data.provide_data[0]]

    if not isinstance(test_data, PrefetchingIter):
        test_data = PrefetchingIter(test_data)

    idx = 0
    t = time.time()
    imdb_boxes = list()
    original_boxes = list()
    for im_info, data_batch in test_data:
        t1 = time.time() - t
        t = time.time()

        scales = [iim_info[0, 2] for iim_info in im_info]
        scores_all, boxes_all, data_dict_all = im_proposal(predictor, data_batch, data_names, scales)
        t2 = time.time() - t
        t = time.time()
        for delta, (scores, boxes, data_dict) in enumerate(zip(scores_all, boxes_all, data_dict_all)):
            # assemble proposals
            dets = np.hstack((boxes, scores))
            original_boxes.append(dets)

            # filter proposals
            keep = np.where(dets[:, 4:] > thresh)[0]
            dets = dets[keep, :]
            imdb_boxes.append(dets)

            if vis:
                vis_all_detection(data_dict['data'].asnumpy(), [dets], ['obj'], scale, cfg)

            print 'generating %d/%d' % (idx + 1, imdb.num_images), 'proposal %d' % (dets.shape[0]), \
                'data %.4fs net %.4fs' % (t1, t2 / test_data.batch_size)
            idx += 1


    assert len(imdb_boxes) == imdb.num_images, 'calculations not complete'

    # save results
    rpn_folder = os.path.join(imdb.result_path, 'rpn_data')
    if not os.path.exists(rpn_folder):
        os.mkdir(rpn_folder)

    rpn_file = os.path.join(rpn_folder, imdb.name + '_rpn.pkl')
    with open(rpn_file, 'wb') as f:
        cPickle.dump(imdb_boxes, f, cPickle.HIGHEST_PROTOCOL)

    if thresh > 0:
        full_rpn_file = os.path.join(rpn_folder, imdb.name + '_full_rpn.pkl')
        with open(full_rpn_file, 'wb') as f:
            cPickle.dump(original_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'wrote rpn proposals to {}'.format(rpn_file)
    return imdb_boxes


def im_detect(predictor, data_batch, data_names, cfg):
    output_all = predictor.predict(data_batch)

    data_dict = dict(zip(data_names, data_batch.data))
    batched_im_info = data_dict['im_info'].asnumpy()
    scores_all = []
    pred_boxes_all = []
    pred_kps_all = []
    for i, output in enumerate(output_all):
        if cfg.TEST.HAS_RPN:
            batch_rois = output['rois_output'].asnumpy()
        else:
            rois = data_dict['rois'].asnumpy().reshape((-1, 5))[:, 1:]

        # save output
        N = cfg.TEST.IMAGES_PER_GPU
        batch_scores = output['cls_prob_reshape_output'].asnumpy()
        batch_scores = batch_scores.reshape((N, -1, batch_scores.shape[2]))
        batch_bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()
        batch_bbox_deltas = batch_bbox_deltas.reshape((N, -1, batch_bbox_deltas.shape[2]))
        if cfg.network.PREDICT_KEYPOINTS:
            kps_deltas = output['kps_pos_pred_reshape_output'].asnumpy()           # [N*R, 2*K, G, G]
            kps_deltas = kps_deltas.reshape([N, -1] + list(kps_deltas.shape[1:])) # [N, R, 2*K, G, G]
            kps_probs = output['kps_prob_output'].asnumpy()            # [N*R*K, G*G]
            kps_probs = kps_probs.reshape((N, -1, kps_probs.shape[1])) # [N, R*K, G*G]

        for j in range(N):
            im_info  = batched_im_info[i*N + j]
            im_shape = im_info[:2]
            scale    = im_info[2]
            if scale < 1e-6:
                continue
            indices = np.where(batch_rois[:,0] == j)[0]
            rois = batch_rois[indices, 1:]
            scores = batch_scores[j]
            bbox_deltas = batch_bbox_deltas[j]

            # post processing
            means = None
            stds  = None
            if cfg.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
                means = np.array(cfg.TRAIN.BBOX_MEANS)
                stds  = np.array(cfg.TRAIN.BBOX_STDS)
            pred_boxes = bbox_pred(rois, bbox_deltas, means=means, stds=stds)
            pred_boxes = clip_boxes(pred_boxes, im_shape)

            # we used scaled image & roi to train, so it is necessary to transform them back
            pred_boxes = pred_boxes / scale

            scores_all.append(scores)
            pred_boxes_all.append(pred_boxes)

            if cfg.network.PREDICT_KEYPOINTS:
                pred_kps = predict_keypoints(rois, kps_probs[j], kps_deltas[j], scale=scale)
                pred_kps_all.append(pred_kps)

    if cfg.network.PREDICT_KEYPOINTS:
        return scores_all, pred_boxes_all, pred_kps_all
    return scores_all, pred_boxes_all


def predict_keypoints(rois, kps_probs, kps_deltas, scale=1.0):
    '''
    rois      : [N, 4]
    kps_probs : [N*K, G*G]
    kps_deltas: [N, 2*K, G, G]

    Return
    pred_kps  : [N, K*3]
    '''
    N = rois.shape[0]         # number of rois
    G = kps_deltas.shape[-1]  # roipooled size
    K = kps_deltas.shape[1]/2 # types of keypoint


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

    argmax = kps_probs.argmax(axis=-1)                           # [N*K]
    argmax_probs = kps_probs.max(axis=-1).reshape([-1,1])        # [N*K, 1]

    wh = rois[:,2:4] - rois[:,0:2]   # [N, 2]
    gwh = wh * (1.0 / G)             # [N, 2]
    gwh = gwh.repeat(K, axis=0)      # [N*K, 2]

    offset = kps_deltas.reshape([-1, 2, G*G]).transpose([0, 2, 1]) # [N*K, G*G, 2]
    argmax_offset = offset[np.arange(offset.shape[0]), argmax]     # [N*K, 2]
    argmax_ctrs = all_ctrs[np.arange(all_ctrs.shape[0]), argmax]   # [N*K, 2]
    pred_xy = argmax_offset * gwh + argmax_ctrs                    # [N*K, 2]
    pred_xy *= 1.0 / scale
    pred_kps = np.concatenate([pred_xy, argmax_probs], axis=1)     # [N*K, 3]

    return pred_kps.reshape([N, K*3])


def vis_all_detection(im_array, detections, class_names, scale, cfg, threshold=1e-3):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import matplotlib.pyplot as plt
    import random
    im = image.transform_inverse(im_array, cfg.network.PIXEL_MEANS)
    plt.imshow(im)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.random(), random.random(), random.random())  # generate a random color
        dets = detections[j]
        for det in dets:
            bbox = det[:4] * scale
            score = det[-1]
            if score < threshold:
                continue
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=3.5)
            plt.gca().add_patch(rect)
            plt.gca().text(bbox[0], bbox[1] - 2,
                           '{:s} {:.3f}'.format(name, score),
                           bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')
    plt.show()


def draw_all_detection(im_array, detections, class_names, scale, cfg, threshold=1e-1):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :param scale: visualize the scaled image
    :return:
    """
    import cv2
    import random
    color_white = (255, 255, 255)
    im = image.transform_inverse(im_array, cfg.network.PIXEL_MEANS)
    # change to bgr
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))  # generate a random color
        dets = detections[j]
        for det in dets:
            bbox = det[:4] * scale
            score = det[-1]
            if score < threshold:
                continue
            bbox = map(int, bbox)
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2)
            cv2.putText(im, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                        color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return im
