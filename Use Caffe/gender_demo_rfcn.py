#!/usr/bin/env python

# --------------------------------------------------------
# R-FCN
# Copyright (c) 2016 Yuwen Xiong
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

#import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
import numpy as np
import caffe, os, cv2
import argparse

CLASSES = ('__background__', 'f', 'm')

NETS = {'ResNet-101': ('models/pascal_voc/ResNet-101/rfcn_end2end/test_agnostic.prototxt', 
	'output/rfcn_end2end/voc_0712_trainval/resnet101_rfcn_iter_110000.caffemodel')}


def vis_detections(im, image_name, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return im
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        if class_name == 'f':
        	colour = (255, 0, 0)
        else:
        	colour = (0, 0, 255)
        cv2.rectangle(im, (bbox[0], bbox[1]),(bbox[2],bbox[3]), colour, 2)
    return im


def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'gender_demo', image_name)
    im = cv2.imread(im_file)
    m = im.astype(np.uint8).copy()
    # Detect all object classes and regress object bounds
    scores, boxes = im_detect(net, im)
    # Visualize detections for each class
    CONF_THRESH = 0.49
    NMS_THRESH = 0.45
    print image_name
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4:8]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        m = vis_detections(m, image_name, cls, dets, thresh=CONF_THRESH)
    cv2.imwrite(os.path.join(results_dir, im_name), m)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]', default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode', help='Use CPU mode (overrides --gpu)', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()
    prototxt = NETS['ResNet-101'][0]
    caffemodel = NETS['ResNet-101'][1]
    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)
    im_names = ['005920.jpg', '005921.jpg', '005922.jpg', '005926.jpg', '005928.jpg', '005929.jpg', 
    '011081.jpg', '011082.jpg', '011083.jpg', '011085.jpg', '011086.jpg']
    for im_name in im_names:
        demo(net, im_name)
