
"""
JSON List database
"""

import cPickle
import cv2
import os
import numpy as np
import PIL

from imdb import IMDB
from pascal_voc_eval import voc_eval, voc_eval_sds
from ds_utils import unique_boxes, filter_small_boxes
import json
import time

class JSONList(IMDB):
    def __init__(self, image_set, root_path, data_path, result_path=None, mask_size=-1, binary_thresh=None):
        """
        fill basic information to initialize imdb
        :param image_set: 15105475915610.8803405387930849.json
        :param root_path: /world/data-c9/dl-data/58b64b589db629d273ac457a/5a0920879b104a97e787032f
        :param devkit_path: data and results
        :return: imdb object
        """
        json_file = os.path.join(root_path, image_set)
        assert os.path.exists(json_file), 'file does not exist: {}'.format(json_file)
        self.json_config = None
        with open(json_file) as f:
            self.json_config = json.loads(f.read())

        image_set = image_set[:-5]
        super(JSONList, self).__init__('JSON', image_set, root_path, data_path, result_path)  # set self.name

        self.root_path = str(self.json_config['working_dir'])
        self.data_path = str(self.json_config['batches_dir'])
        self.json_list = os.path.join(self.data_path, 'inflated_list.json')
        #self.json_list = os.path.join(self.data_path, 'test_list.json')

        self.classes = ['__background__'] + [str(i) for i in range(int(self.json_config['num_labels']))]
        self.num_classes = len(self.classes)

        self.json_lines = []
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        print 'num_images', self.num_images
        print 'num_labels', self.num_classes
        self.mask_size = mask_size
        self.binary_thresh = binary_thresh

        self.config = {'comp_id': 'comp4',
                       'use_diff': False,
                       'min_size': 2}

    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        assert os.path.exists(self.json_list), 'Path does not exist: {}'.format(self.json_list)
        image_set_index = []
        with open(self.json_list) as f:
            for i, line in enumerate(f.readlines()):
                if i % 2 == 0: ### for debug
                    continue ###
                items = line.strip().split('\t')
                if len(items) < 2 : # path, json_str, ..
                    continue
                image_set_index.append(items[0])
                self.json_lines.append(items[1])
        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        return index

    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = []
        t0 = time.time()
        for i, line in enumerate(self.json_lines):
            if i % 1000 == 0:
                print 'loaded images:', i, time.time() - t0 ###
            index = self.image_set_index[i]
            gt_roidb.append(self.load_json_annotation(index, line))

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def load_json_annotation(self, index, anno_str):
        """
        for a given index, load image and bounding boxes info from XML file
        :param index: index of a specific image
        :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        roi_rec = dict()
        roi_rec['image'] = self.image_path_from_index(index)

        im_size = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|128).shape
        roi_rec['height'] = float(im_size[0])
        roi_rec['width'] = float(im_size[1])

        objs = json.loads(anno_str)
        # filter invalid objs
        raw_len = len(objs) ###
        objs = [obj for obj in objs if int(obj['name']) >= 0]
        if len(objs) != raw_len: ###
            print 'invalid objs found:', index ###
        # filter difficult objs
        if not self.config['use_diff']:
            objs = [obj for obj in objs if int(obj.get('difficult')) == 0]
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        class_to_index = dict(zip(self.classes, range(self.num_classes)))
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            x1 = float(obj['xmin'])
            y1 = float(obj['ymin'])
            x2 = min(float(obj['xmax']), float(im_size[1]-1))
            y2 = min(float(obj['ymax']), float(im_size[0]-1))
            boxes[ix, :] = [x1, y1, x2, y2]
            cls = class_to_index[str(obj['name'])]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        roi_rec.update({'boxes': boxes,
                        'gt_classes': gt_classes,
                        'gt_overlaps': overlaps,
                        'max_classes': overlaps.argmax(axis=1),
                        'max_overlaps': overlaps.max(axis=1),
                        'flipped': False})
        return roi_rec


    def evaluate_detections(self, detections):
        """
        top level evaluations
        :param detections: result matrix, [bbox, confidence]
        :return: None
        """
        # make all these folders for results
        result_dir = os.path.join(self.result_path, 'results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        year_folder = os.path.join(self.result_path, 'results', 'VOC' + self.year)
        if not os.path.exists(year_folder):
            os.mkdir(year_folder)
        res_file_folder = os.path.join(self.result_path, 'results', 'VOC' + self.year, 'Main')
        if not os.path.exists(res_file_folder):
            os.mkdir(res_file_folder)

        self.write_pascal_results(detections)
        info = self.do_python_eval()
        return info


    def write_pascal_results(self, all_boxes):
        """
        write results files in pascal devkit path
        :param all_boxes: boxes to be processed [bbox, confidence]
        :return: None
        """
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self.get_result_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_set_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if len(dets) == 0:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1))


    def do_python_eval(self):
        """
        python evaluation wrapper
        :return: info_str
        """
        info_str = ''
        annopath = os.path.join(self.data_path, 'Annotations', '{0!s}.xml')
        imageset_file = os.path.join(self.data_path, 'ImageSets', 'Main', self.image_set + '.txt')
        annocache = os.path.join(self.cache_path, self.name + '_annotations.pkl')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if self.year == 'SDS' or int(self.year) < 2010 else False
        print 'VOC07 metric? ' + ('Y' if use_07_metric else 'No')
        info_str += 'VOC07 metric? ' + ('Y' if use_07_metric else 'No')
        info_str += '\n'
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            filename = self.get_result_file_template().format(cls)
            rec, prec, ap = voc_eval(filename, annopath, imageset_file, cls, annocache,
                                     ovthresh=0.5, use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            info_str += 'AP for {} = {:.4f}\n'.format(cls, ap)
        print('Mean AP@0.5 = {:.4f}'.format(np.mean(aps)))
        info_str += 'Mean AP@0.5 = {:.4f}\n\n'.format(np.mean(aps))
        # @0.7
        aps = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            filename = self.get_result_file_template().format(cls)
            rec, prec, ap = voc_eval(filename, annopath, imageset_file, cls, annocache,
                                     ovthresh=0.7, use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            info_str += 'AP for {} = {:.4f}\n'.format(cls, ap)
        print('Mean AP@0.7 = {:.4f}'.format(np.mean(aps)))
        info_str += 'Mean AP@0.7 = {:.4f}'.format(np.mean(aps))
        return info_str
