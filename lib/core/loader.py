# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------

import random
import time
import multiprocessing as mp
from multiprocessing import sharedctypes

import numpy as np
import mxnet as mx
from mxnet.executor_manager import _split_input_slice

from utils.image import get_image, tensor_vstack
from rpn.rpn import assign_anchor, assign_pyramid_anchor
from core.rcnn import sample_rois

def as_mx(arr):
    if isinstance(arr, mx.nd.NDArray):
        return arr
    else:
        return mx.nd.array(arr)


def get_rpn_testbatch(roidb, batch_size, cfg, data_buf):
    """
    """
    assert len(roidb) > 0 , 'empty list !'

    # get images
    t0 = time.time() ###
    target_size, max_size = cfg.TEST.SCALES[0]
    imgs, roidb = get_image(roidb, target_size, max_size, stride=cfg.network.IMAGE_STRIDE)
    max_h = max([img.shape[0] for img in imgs])
    max_w = max([img.shape[1] for img in imgs])
    t1 = time.time() ###

    # shapes
    shapes = {'data'    : (batch_size, 3, max_h, max_w),
              'im_info' : (batch_size, 3),
             }

    # reshape buffers
    batch = dict()
    for k in data_buf:
        s = shapes[k]
        c = np.prod(s)
        batch[k] = np.frombuffer(data_buf[k], dtype=np.float32, count=c).reshape(s)
        batch[k].fill(0)

    # transform image data
    bgr_means = cfg.network.PIXEL_MEANS
    for i, img in enumerate(imgs):
        h, w = img.shape[:2]
        for j in range(3):
            batch['data'][i, j, :h, :w] = img[:,:,2-j] - bgr_means[2-j]
        batch['im_info'][i, :] = [max_h, max_w, roidb[i]['im_info'][2]]

    t2 = time.time() ###
    #print 't_image:%.3f\tt_trans:%.3f' % (t1-t0, t2-t1) ###
    return shapes


def get_rpn_batch(roidb, sym, cfg, data_buf, allowed_border=0, max_gts=100, kps_dim=0):
    """
    allowed_border:

    max_gts:
        max number of groundtruths

    kps_dim:
        when trainning with keypoints, set kps_dim >0
    """
    num_images = len(roidb)
    assert num_images > 0 , 'empty list !'

    # get images
    t0 = time.time() ###
    target_size, max_size = random.choice(cfg.TRAIN.SCALES)
    imgs, roidb = get_image(roidb, target_size, max_size, stride=cfg.network.IMAGE_STRIDE)
    max_h = max([img.shape[0] for img in imgs])
    max_w = max([img.shape[1] for img in imgs])
    t1 = time.time() ###

    # assign anchor labels
    anchor_labels = []
    _, feat_shape, _ = sym.infer_shape(data=(num_images, 3, max_h, max_w))
    for i in range(num_images):
        if roidb[i]['gt_classes'].size > 0:
            assert np.sum(roidb[i]['gt_classes'] == 0) == 0, 'should not have background boxes!'
        gt_boxes = roidb[i]['boxes']
        im_info = [max_h, max_w, roidb[i]['im_info'][2]]
        # assign anchors
        anchor_labels.append(assign_anchor(feat_shape[0], gt_boxes, im_info, cfg, allowed_border))
    t2 = time.time() ###

    # shapes
    shapes = {'data'    : (num_images, 3, max_h, max_w),
              'im_info' : (num_images, 3),
              'gt_boxes': (num_images, max_gts, 5),
              'label'      : tuple([num_images] + list(anchor_labels[0]['label'].shape[1:])),
              'bbox_target': tuple([num_images] + list(anchor_labels[0]['bbox_target'].shape[1:])),
              'bbox_weight': tuple([num_images] + list(anchor_labels[0]['bbox_weight'].shape[1:])),
            }
    if kps_dim >0:
        shapes['gt_kps'] = ((num_images, max_gts, kps_dim))

    # reshape buffers
    batch = dict()
    for k in data_buf:
        s = shapes[k]
        c = np.prod(s)
        batch[k] = np.frombuffer(data_buf[k], dtype=np.float32, count=c).reshape(s)
        batch[k].fill(0)

    # transform image data and gt labels
    bgr_means = cfg.network.PIXEL_MEANS
    for i, img in enumerate(imgs):
        h, w = img.shape[:2]
        for j in range(3):
            batch['data'][i, j, :h, :w] = img[:,:,2-j] - bgr_means[2-j]
        batch['im_info'][i, :] = [max_h, max_w, roidb[i]['im_info'][2]]
        num_gt = roidb[i]['boxes'].shape[0]
        batch['gt_boxes'][i, :num_gt, :4] = roidb[i]['boxes']
        batch['gt_boxes'][i, :num_gt, 4] = roidb[i]['gt_classes']
        if kps_dim >0:
            batch['gt_kps'][i, :num_gt] = roidb[i]['keypoints']
        batch['label'][i]       = anchor_labels[i]['label']
        batch['bbox_target'][i] = anchor_labels[i]['bbox_target']
        batch['bbox_weight'][i] = anchor_labels[i]['bbox_weight']
    t3 = time.time() ###

    #print 't_image=%.3f\tt_assign=%.3f\tt_trans=%.3f\tt_all=%.3f' % (t1-t0, t2-t1, t3-t2, t3-t0) ###
    return shapes



class TestLoader(mx.io.DataIter):
    def __init__(self, roidb, cfg, batch_size=1, shuffle=False,
                 has_rpn=False):
        super(TestLoader, self).__init__()

        # save parameters as properties
        self.cfg = cfg
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.has_rpn = has_rpn

        # infer properties from roidb
        self.size = len(self.roidb)
        self.index = np.arange(self.size)

        # decide data and label names
        self.max_data_shapes = dict()
        self.max_size =  max([max(s) for s in cfg.TEST.SCALES])
        self.max_data_shapes['data'] = (self.batch_size, 3, self.max_size, self.max_size)
        self.max_data_shapes['im_info'] = (self.batch_size, 3)
        self.data_names = self.max_data_shapes.keys()
        self._provide_data = [(k, self.max_data_shapes[k]) for k in self.data_names]

        # init bufs
        self.i_buf = 0
        self.buf_size = 1
        self.buf_list = []
        self.init_buffer()

        self.reset()

    def init_buffer(self):
        self.buf_list = []
        all_shapes = self.provide_data + self.provide_label
        for i in range(self.buf_size):
            buf = dict()
            for k, s in all_shapes:
                size = np.prod(s)
                buf[k] = sharedctypes.RawArray('f', size)
            self.buf_list.append(buf)

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return []

    def reset(self):
        self.cur = 0

    def iter_next(self):
        return self.cur < self.size

    def next(self):
        if self.iter_next():
            batch = self.get_batch()
            self.cur += self.batch_size
            return batch
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        buf = self.buf_list[self.i_buf]
        shapes = get_rpn_testbatch(roidb, self.batch_size, self.cfg, buf)
        self.i_buf = (self.i_buf + 1) % self.buf_size

        data = dict()
        for k in shapes:
            s = shapes[k]
            c = np.prod(s)
            data[k] = np.frombuffer(buf[k], dtype=np.float32, count=c).reshape(s)
            data[k] = as_mx(data[k])
        self._provide_data  = [(k, data[k].shape) for k in self.data_names]
        return mx.io.DataBatch(data  = [data[k] for k in self.data_names],
                               label = [],
                               pad   = self.getpad(),
                               index = self.getindex(),
                               provide_data  = self.provide_data,
                               provide_label = self.provide_label)



class AnchorLoader(mx.io.DataIter):

    def __init__(self, feat_sym, roidb, cfg, batch_size=1, shuffle=False, ctx=None, work_load_list=None,
                 feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2), allowed_border=0,
                 aspect_grouping=False):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :return: AnchorLoader
        """
        super(AnchorLoader, self).__init__()

        # save parameters as properties
        self.feat_sym = feat_sym
        self.roidb = roidb
        self.cfg = cfg
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.feat_stride = feat_stride
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.allowed_border = allowed_border
        self.aspect_grouping = aspect_grouping
        self.max_gts = cfg.dataset.MAX_BOXES_PER_IMAGE
        self.max_size =  max([max(s) for s in cfg.TRAIN.SCALES])

        # infer properties from roidb
        self.size = len(roidb)
        self.size -= self.size % self.batch_size # make size can be divided by batch_size
        self.index = np.arange(self.size)

        # decide data and label names
        self.max_data_shapes = dict()
        self.max_data_shapes['data'] = (self.batch_size, 3, self.max_size, self.max_size)
        self.max_data_shapes['im_info'] = (self.batch_size, 3)
        self.max_data_shapes['gt_boxes'] = (self.batch_size, self.max_gts, 5)
        if cfg.network.PREDICT_KEYPOINTS:
            self.kps_dim = 3*cfg.dataset.NUM_KEYPOINTS
            self.max_data_shapes['gt_kps'] = (self.batch_size, self.max_gts, self.kps_dim)
        else:
            self.kps_dim = 0
        self.data_names = self.max_data_shapes.keys()
        self.label_names = ['label', 'bbox_target', 'bbox_weight']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self._provide_data = None
        self._provide_label = None

        # multiprocess for loading batch
        self.buf_size = 4
        self.buf_list = []
        self.buf_queue = None
        self.num_worker = 4
        self.batch_queue = None
        self.batch_process = None

        self.infer_shape()
        self.init_buffer()
        self.reset()


    def infer_shape(self):
        _, feat_shape, _ = self.feat_sym.infer_shape(data=self.max_data_shapes['data'])
        im_info = [self.max_size, self.max_size, 1.0]
        label = assign_anchor(feat_shape[0], np.zeros((0, 5)), im_info, self.cfg, self.allowed_border)
        label = [label[k] for k in self.label_names]
        label_shapes = [tuple([self.batch_size] + list(v.shape[1:])) for v in label]
        self._provide_label = zip(self.label_names, label_shapes)
        self._provide_data = [(k, self.max_data_shapes[k]) for k in self.data_names]

    def init_buffer(self):
        self.buf_queue = mp.Queue()
        self.buf_list = []
        all_shapes = self.provide_data + self.provide_label
        for i in range(self.buf_size):
            buf = dict()
            for k, s in all_shapes:
                size = np.prod(s)
                buf[k] = sharedctypes.RawArray('f', size)
            self.buf_list.append(buf)
            self.buf_queue.put(i)

    def reset(self):
        self.cur = 0
        # shuffle indexes
        if self.shuffle:
            if self.aspect_grouping:
                self.index = self.index.reshape((-1, self.batch_size))
            np.random.shuffle(self.index)
            self.index = self.index.flatten()

        # terminate all worker process
        if self.batch_process is not None:
            self.terminate()
        # slice roidb
        roidb_slices = [self.roidb[i:i+self.batch_size] for i in range(0,self.size,self.batch_size)]
        print 'total roidb_slices:', len(roidb_slices) ###
        # reset batch_queue
        self.batch_queue = mp.Queue(self.num_worker)

        # start worker process
        self.batch_process = []
        def worker(roidbs, q_buf, q_batch):
            for r in roidbs:
                i_buf = q_buf.get()
                #print i_buf, 'used' ###
                buf = self.buf_list[i_buf]
                shapes = get_rpn_batch(r, self.feat_sym, self.cfg, buf,
                                       allowed_border=self.allowed_border,
                                       max_gts=self.max_gts,
                                       kps_dim=self.kps_dim)
                q_batch.put((i_buf, shapes))
        for i in range(self.num_worker):
            roidb_slice = roidb_slices[i::self.num_worker]
            print 'worker roidb_slices:', len(roidb_slice) ###
            p = mp.Process(target=worker, args=(roidb_slice, self.buf_queue, self.batch_queue))
            self.batch_process.append(p)
            p.start()

    def terminate(self):
        if self.batch_process is not None:
            for w in self.batch_process:
                w.terminate()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def iter_next(self):
        return self.cur < self.size

    def next(self):
        if self.iter_next():
            batch = self.get_batch()
            self.cur += self.batch_size
            return batch
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0


    def get_batch(self):
        '''
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        buf = self.buf_list[0]
        shapes = get_rpn_batch(roidb, self.feat_sym, self.cfg, buf,
                               allowed_border=self.allowed_border,
                               max_gts=self.max_gts,
                               kps_dim=self.kps_dim)
        '''
        i_buf, shapes = self.batch_queue.get()
        buf = self.buf_list[i_buf]

        t0 = time.time() ###
        data = dict()
        for k in shapes:
            s = shapes[k]
            c = np.prod(s)
            data[k] = np.frombuffer(buf[k], dtype=np.float32, count=c).reshape(s)
            data[k] = as_mx(data[k])
        self.buf_queue.put(i_buf)
        #print i_buf, 'free', 'to_mx:', time.time() - t0 ###
        self._provide_data  = [(k, data[k].shape) for k in self.data_names]
        self._provide_label = [(k, data[k].shape) for k in self.label_names]
        return mx.io.DataBatch(data  = [data[k] for k in self.data_names],
                               label = [data[k] for k in self.label_names],
                               pad   = self.getpad(),
                               index = self.getindex(),
                               provide_data  = self.provide_data,
                               provide_label = self.provide_label)



def par_assign_anchor_wrapper(cfg, iroidb, feat_sym, feat_strides, anchor_scales, anchor_ratios, allowed_border):
    # get testing data for multigpu
    data, rpn_label = get_rpn_batch(iroidb, cfg)
    data_shape = {k: v.shape for k, v in data.items()}
    del data_shape['im_info']

    # add gt_boxes to data for e2e
    data['gt_boxes'] = rpn_label['gt_boxes'][np.newaxis, :, :]

    # add gt_kps to data for e2e
    if 'gt_kps' in rpn_label:
        data['gt_kps'] = rpn_label['gt_kps'][np.newaxis, :, :]


    feat_shape = [y[1] for y in [x.infer_shape(**data_shape) for x in feat_sym]]
    label = assign_pyramid_anchor(feat_shape, rpn_label['gt_boxes'], data['im_info'], cfg,
                                  feat_strides, anchor_scales, anchor_ratios, allowed_border)
    return {'data': data, 'label': label}


class PyramidAnchorLoader(mx.io.DataIter):

    # pool = Pool(processes=4)
    def __init__(self, feat_sym, roidb, cfg, batch_size=1, shuffle=False, ctx=None, work_load_list=None,
                 feat_strides=(4, 8, 16, 32, 64), anchor_scales=(8, ), anchor_ratios=(0.5, 1, 2), allowed_border=0,
                 aspect_grouping=False):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects
        :return: AnchorLoader
        """
        super(PyramidAnchorLoader, self).__init__()

        # save parameters as properties
        self.feat_sym = feat_sym
        self.roidb = roidb
        self.cfg = cfg
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.feat_strides = feat_strides
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.allowed_border = allowed_border
        self.aspect_grouping = aspect_grouping

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names
        if self.cfg.TRAIN.END2END:
            self.data_name = ['data', 'im_info', 'gt_boxes']
            if cfg.network.PREDICT_KEYPOINTS:
                self.data_name += ['gt_kps']
        else:
            self.data_name = ['data']
        #self.feat_pyramid_level = np.log2(self.cfg.network.RPN_FEAT_STRIDE).astype(int)
        # self.label_name = ['label_p' + str(x) for x in self.feat_pyramid_level] +\
        #                   ['bbox_target_p' + str(x) for x in self.feat_pyramid_level] +\
        #                   ['bbox_weight_p' + str(x) for x in self.feat_pyramid_level]

        self.label_name = ['label', 'bbox_target', 'bbox_weight']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch_parallel()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, self.data[i])] for i in xrange(len(self.data))]

    @property
    def provide_label(self):
        return [[(k, v.shape) for k, v in zip(self.label_name, self.label[i])] for i in xrange(len(self.data))]

    @property
    def provide_data_single(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label[0])]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            if self.aspect_grouping:
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))
                extra = inds.shape[0] % self.batch_size
                inds_ = np.reshape(inds[:-extra], (-1, self.batch_size))
                row_perm = np.random.permutation(np.arange(inds_.shape[0]))
                inds[:-extra] = np.reshape(inds_[row_perm, :], (-1,))
                self.index = inds
            else:
                np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch_parallel()
            # self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def infer_shape(self, max_data_shape=None, max_label_shape=None):
        """ Return maximum data and label shape for single gpu """
        if max_data_shape is None:
            max_data_shape = []
        if max_label_shape is None:
            max_label_shape = []
        max_shapes = dict(max_data_shape + max_label_shape)
        input_batch_size = max_shapes['data'][0]
        im_info = [[max_shapes['data'][2], max_shapes['data'][3], 1.0]]

        feat_shape = [y[1] for y in [x.infer_shape(**max_shapes) for x in self.feat_sym]]
        label = assign_pyramid_anchor(feat_shape, np.zeros((0, 5)), im_info, self.cfg,
                                      self.feat_strides, self.anchor_scales, self.anchor_ratios, self.allowed_border)
        label = [label[k] for k in self.label_name]
        label_shape = [(k, tuple([input_batch_size] + list(v.shape[1:]))) for k, v in zip(self.label_name, label)]

        return max_data_shape, label_shape

    def get_batch_parallel(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        rst = []
        for idx, islice in enumerate(slices):
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            rst.append(par_assign_anchor_wrapper(self.cfg, iroidb, self.feat_sym, self.feat_strides, self.anchor_scales,
                                                 self.anchor_ratios, self.allowed_border))

        all_data = [_['data'] for _ in rst]
        all_label = [_['label'] for _ in rst]
        self.data = [[mx.nd.array(data[key]) for key in self.data_name] for data in all_data]
        self.label = [[mx.nd.array(label[key]) for key in self.label_name] for label in all_label]


def get_rcnn_batch(roidb, cfg):
    """
    return a dict of multiple images
    :param roidb: a list of dict, whose length controls batch size
    ['images', 'flipped'] + ['gt_boxes', 'boxes', 'gt_overlap'] => ['bbox_targets']
    :return: data, label
    """
    num_images = len(roidb)
    imgs, roidb = get_image(roidb, cfg)
    im_array = tensor_vstack(imgs)

    assert cfg.TRAIN.BATCH_ROIS == -1 or cfg.TRAIN.BATCH_ROIS % cfg.TRAIN.BATCH_IMAGES == 0, \
        'BATCHIMAGES {} must divide BATCH_ROIS {}'.format(cfg.TRAIN.BATCH_IMAGES, cfg.TRAIN.BATCH_ROIS)

    if cfg.TRAIN.BATCH_ROIS == -1:
        rois_per_image = np.sum([iroidb['boxes'].shape[0] for iroidb in roidb])
        fg_rois_per_image = rois_per_image
    else:
        rois_per_image = cfg.TRAIN.BATCH_ROIS / cfg.TRAIN.BATCH_IMAGES
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image).astype(int)

    rois_array = list()
    labels_array = list()
    bbox_targets_array = list()
    bbox_weights_array = list()

    for im_i in range(num_images):
        roi_rec = roidb[im_i]

        # infer num_classes from gt_overlaps
        num_classes = roi_rec['gt_overlaps'].shape[1]

        # label = class RoI has max overlap with
        rois = roi_rec['boxes']
        labels = roi_rec['max_classes']
        overlaps = roi_rec['max_overlaps']
        bbox_targets = roi_rec['bbox_targets']

        im_rois, labels, bbox_targets, bbox_weights = \
            sample_rois(rois, fg_rois_per_image, rois_per_image, num_classes, cfg,
                        labels, overlaps, bbox_targets)

        # project im_rois
        # do not round roi
        rois = im_rois
        batch_index = im_i * np.ones((rois.shape[0], 1))
        rois_array_this_image = np.hstack((batch_index, rois))
        rois_array.append(rois_array_this_image)

        # add labels
        labels_array.append(labels)
        bbox_targets_array.append(bbox_targets)
        bbox_weights_array.append(bbox_weights)

    rois_array = np.array(rois_array)
    labels_array = np.array(labels_array)
    bbox_targets_array = np.array(bbox_targets_array)
    bbox_weights_array = np.array(bbox_weights_array)

    data = {'data': im_array,
            'rois': rois_array}
    label = {'label': labels_array,
             'bbox_target': bbox_targets_array,
             'bbox_weight': bbox_weights_array}

    return data, label


class ROIIter(mx.io.DataIter):
    def __init__(self, roidb, cfg, batch_size=2, shuffle=False, ctx=None, work_load_list=None, aspect_grouping=False):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects
        :return: ROIIter
        """
        super(ROIIter, self).__init__()

        # save parameters as properties
        self.roidb = roidb
        self.cfg = cfg
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.aspect_grouping = aspect_grouping

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names (only for training)
        self.data_name = ['data', 'rois']
        self.label_name = ['label', 'bbox_target', 'bbox_weight']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch_individual()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, self.data[i])] for i in xrange(len(self.data))]

    @property
    def provide_label(self):
        return [[(k, v.shape) for k, v in zip(self.label_name, self.label[i])] for i in xrange(len(self.data))]

    @property
    def provide_data_single(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label[0])]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            if self.aspect_grouping:
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))
                extra = inds.shape[0] % self.batch_size
                inds_ = np.reshape(inds[:-extra], (-1, self.batch_size))
                row_perm = np.random.permutation(np.arange(inds_.shape[0]))
                inds[:-extra] = np.reshape(inds_[row_perm, :], (-1,))
                self.index = inds
            else:
                np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch_individual()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        # slice roidb
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]

        # decide multi device slices
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        # get each device
        data_list = []
        label_list = []
        for islice in slices:
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            data, label = get_rcnn_batch(iroidb, self.cfg)
            data_list.append(data)
            label_list.append(label)

        all_data = dict()
        for key in data_list[0].keys():
            all_data[key] = tensor_vstack([batch[key] for batch in data_list])

        all_label = dict()
        for key in label_list[0].keys():
            all_label[key] = tensor_vstack([batch[key] for batch in label_list])

        self.data = [mx.nd.array(all_data[name]) for name in self.data_name]
        self.label = [mx.nd.array(all_label[name]) for name in self.label_name]

    def get_batch_individual(self):
        # slice roidb
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]

        # decide multi device slices
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        rst = []
        for idx, islice in enumerate(slices):
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            rst.append(self.parfetch(iroidb))

        all_data = [_['data'] for _ in rst]
        all_label = [_['label'] for _ in rst]
        self.data = [[mx.nd.array(data[key]) for key in self.data_name] for data in all_data]
        self.label = [[mx.nd.array(label[key]) for key in self.label_name] for label in all_label]

    def parfetch(self, iroidb):
        data, label = get_rcnn_batch(iroidb, self.cfg)
        return {'data': data, 'label': label}
