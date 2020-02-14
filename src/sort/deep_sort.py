import numpy as np
import sys
import os
import cv2
from preprocessing import non_max_suppression
from detection import Detection
from tracker import Tracker
from nn_matching import NearestNeighborDistanceMetric
sys.path.append(os.path.join(os.path.dirname(__file__),'../extract_feature/pedestrian'))
from feature_extractor import Extractor
sys.path.append(os.path.join(os.path.dirname(__file__),'../extract_feature/face'))
from face_model import mx_FaceRecognize
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs


class DeepSort(object):
    def __init__(self, model_path,epoch_num=0,max_dist=0.2,use_cuda=False,mot_type='face'):
        self.min_confidence = 0.3
        self.nms_max_overlap = 1.0
        self.img_size = [112,112]
        self.mot_type = mot_type
        self.line_inout = cfgs.inout_point
        if mot_type == 'face':
            self.extractor = mx_FaceRecognize(model_path,epoch_num,self.img_size)
        else:
            self.extractor = Extractor(model_path, use_cuda=use_cuda)
        max_cosine_distance = cfgs.max_cosine_distance #max_dist
        nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)
        self.feature_tmp = []
        self.history_inout = dict()
        # if xy is 0 ,it will calculate inout for the door is right or left with x-axis
        # if xy is 1, it will calculate inout for the door is up or down with y-axis
        self.xy = 1
        self.inout_type = cfgs.inout_type

    def update(self, bbox_xcycwh, confidences, ori_img,update_fg=True):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        if update_fg or not(len(bbox_xcycwh)==len(self.feature_tmp)):
            features = self._get_features(bbox_xcycwh, ori_img)
            self.feature_tmp = features
        else:
            features = self.feature_tmp
        detections = [Detection(bbox_xcycwh[i], conf, features[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]

        # run on non-maximum supression
        # boxes = np.array([d.tlwh for d in detections])
        # scores = np.array([d.confidence for d in detections])
        # indices = non_max_suppression( boxes, self.nms_max_overlap, scores)
        # detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)
        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            #print('time_s',track.time_since_update)
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            fg = self.check_in_out(track)
            outputs.append(np.array([x1,y1,x2,y2,track_id,fg], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)
        return outputs
    def check_in_out(self,track):
        '''
        according to the trajectories of a track, judge the moving direction and if move out or in
        1: in
        0: out
        2: keep state
        -1: init state
        '''
        tmp_centers = track.trajects
        keyname = track.track_id
        fg = self.history_inout.setdefault(keyname,-1)
        trj_num = len(tmp_centers)
        if trj_num >2:
            p_start = tmp_centers[0]
            p_end = tmp_centers[trj_num-1]
            diff = p_end[self.xy] - p_start[self.xy]
            if diff > 0 and p_start[self.xy] < self.line_inout[self.xy] and p_end[self.xy] > self.line_inout[self.xy]:
                if self.inout_type==0:
                    if fg == -1:
                        fg = 0
                    elif fg == 0:
                        fg = 2
                else:
                    if fg == -1:
                        fg = 1
                    elif fg == 1:
                        fg = 2
            elif diff < 0 and p_start[self.xy] > self.line_inout[self.xy] and p_end[self.xy] < self.line_inout[self.xy]:
                if self.inout_type==0:
                    if fg == -1 :
                        fg = 1
                    elif fg == 1:
                        fg = 2
                else:
                    if fg == -1:
                        fg = 0
                    elif fg == 0:
                        fg = 2
        self.history_inout[keyname] = fg 
        return fg


    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        bbox_xywh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        bbox_xywh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_xywh


    def _xcycwh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2
    
    def _get_features(self, bbox_xcycwh, ori_img):
        im_crops = []
        for box in bbox_xcycwh:
            _,_,w,h = box
            x1,y1,x2,y2 = self._xcycwh_to_xyxy(box)
            im = ori_img[y1:y2,x1:x2]
            #print('img',ori_img.shape)
            if self.mot_type=='face':
                #if w != self.img_size[1] or h != self.img_size[0]:
                im = cv2.resize(im,(self.img_size[1],self.img_size[0]))
            im_crops.append(im)
        if len(im_crops)>0:
            features = self.extractor.extractfeature(np.array(im_crops)) if self.mot_type=='face' else self.extractor(im_crops)
        else:
            features = np.array([])
        return features


