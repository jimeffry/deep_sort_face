#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
from easydict import EasyDict
import numpy as np


cfg = EasyDict()

# anchor config
cfg.FEATURE_MAPS = [160, 80, 40, 20, 10, 5]
cfg.STEPS = [4, 8, 16, 32, 64, 128]
cfg.ANCHOR_SIZES = [16, 32, 64, 128, 256, 512]
cfg.CLIP = False
cfg.VARIANCE = [0.1, 0.2]
# detection config
cfg.NMS_THRESH = 0.5
cfg.NMS_TOP_K = 300
cfg.TOP_K = 750
cfg.CONF_THRESH = 0.3
cfg.resize_width = 640
cfg.resize_height = 640
cfg.NUM_CLASSES = 2
cfg.USE_NMS = True

# head config
cfg.HEAD = EasyDict()
cfg.HEAD.DIR = '/data/detect/Scut_Head/'
cfg.HEAD.OVERLAP_THRESH = [0.1, 0.35, 0.5]
# crowedhuman
cfg.crowedhuman_train_file = 'crowedhuman_train.txt'
cfg.crowedhuman_val_file = 'crowedhuman_val.txt'
cfg.crowedhuman_dir = '/data/detect/head/imgs'
