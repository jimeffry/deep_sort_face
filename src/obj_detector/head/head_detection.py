#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
from box_utils import decode, nms,nms_py,nms_t
import numpy as np


class Detect(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, cfg):
        self.num_classes = cfg.NUM_CLASSES
        self.top_k = cfg.TOP_K
        self.nms_thresh = cfg.NMS_THRESH
        self.conf_thresh = cfg.CONF_THRESH
        self.variance = cfg.VARIANCE
        self.nms_top_k = cfg.NMS_TOP_K
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4] 
        """
        num = loc_data.size(0)
        num_priors = prior_data.size(0)
        conf_data = self.softmax(conf_data)

        conf_preds = conf_data.view(
            num, num_priors, self.num_classes).transpose(2, 1)
        batch_priors = prior_data.view(-1, num_priors,
                                       4).expand(num, num_priors, 4)
        batch_priors = batch_priors.contiguous().view(-1, 4)

        decoded_boxes = decode(loc_data.view(-1, 4),
                               batch_priors, self.variance)
        decoded_boxes = decoded_boxes.view(num, num_priors, 4)

        # output = torch.zeros(num, self.num_classes, self.top_k, 5)
        output = list()
        for i in range(num):
            boxes = decoded_boxes[i].clone()
            conf_scores = conf_preds[i].clone()
            c_mask = conf_scores[1].gt(self.conf_thresh)
            scores = conf_scores[1][c_mask]
            if scores.numel() == 0:
                continue
            l_mask = c_mask.unsqueeze(1).expand_as(boxes)
            boxes_ = boxes[l_mask].view(-1, 4)
            ids, count = nms(boxes_, scores, self.nms_thresh, self.nms_top_k)
            # ids, count = nms_py(boxes_, scores, self.nms_thresh, self.nms_top_k)
            count = count if count < self.top_k else self.top_k
            #ids = torch.tensor(ids)
            if count >0:
                box_score = [boxes_[ids[:count]].detach().numpy(),scores[ids[:count]].detach().numpy()]
                output.append(box_score)
        return output
