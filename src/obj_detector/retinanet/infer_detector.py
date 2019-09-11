#######################################################
#author: lxy
#time: 14:30 2019.7.24
#tool: python3
#version: 0.1
#project: pedestrian detection
#modify:
#####################################################
import os
import sys
import torch
from torch import nn
from torch.autograd import Function
from utils import  BBoxTransform, ClipBoxes
from pth_nms import pth_nms
sys.path.append(os.path.join(os.path.dirname(__file__),'../../configs'))
from retinanet_config import cfgs

class InferDetector(Function):
    def __init__(self):
        #super(RetinanetDetector,self).__init__()
        self.top_k = cfgs.top_k
        self.score_threshold = cfgs.score_threshold
        self.nms_threshold = cfgs.nms_threshold
        self.num_classes = cfgs.ClsNum
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes(cfgs.ImgSize,cfgs.ImgSize)

    def forward(self,anchors,regression,classification):
        transformed_anchors = self.regressBoxes(anchors, regression)
        transformed_anchors = self.clipBoxes(transformed_anchors)
        scores = torch.max(classification, dim=2, keepdim=True)[0]
        scores_over_thresh = (scores>self.score_threshold)[0, :, 0]
        '''
        batch = transformed_anchors.size(0)  # batch size
        num_priors = anchors.size(1)
        output = torch.zeros(batch, self.num_classes, self.top_k, 5)
        conf_preds = classification.view(batch, num_priors,self.num_classes).transpose(2, 1)
        for i in range(batch):
            decoded_boxes = transformed_anchors[i]
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            for cl in range(self.num_classes):
                c_mask = conf_scores[cl].gt(self.score_threshold)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                #print(boxes.size(), scores.size())
                ids, count = pth_nms(boxes, scores,self.nms_threshold,self.top_k)
                ids = torch.tensor(ids,dtype=torch.long)
                if count ==0:
                    continue
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].view(-1,1),
                               boxes[ids[:count]].view(-1,4)), 1)
        '''
        if scores_over_thresh.sum() == 0:
            # no boxes to NMS, just return
            return (torch.zeros(0), torch.zeros(0), torch.zeros(0, 4))
        classification = classification[:, scores_over_thresh, :]
        transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
        scores = scores[:, scores_over_thresh, :]
        anchors_nms_idx,_ = pth_nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :],overlap=cfgs.nms_threshold)
        nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)
        return (nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :])
        
        #return output
