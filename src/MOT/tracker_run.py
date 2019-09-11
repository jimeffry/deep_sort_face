#######################################################
#author: lxy
#time: 14:30 2019.8.29
#tool: python3
#version: 0.1
#project: MOT
#modify:
#####################################################
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../tracker'))
from staple import Staple
from mosse import MOSSE
from dat import DAT
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from staple_config import StapleConfig

class TrackerRun(object):
    def __init__(self,tracker_type):
        self.tracker_type = tracker_type

    def init(self,frame,bboxes):
        self.tracks = []
        for tmp in bboxes:
            if self.tracker_type=='mosse':
                self.tracks.append(MOSSE())
            elif self.tracker_type=='staple':
                self.tracks.append(Staple(config=StapleConfig()))
            elif self.tracker_type=='dat':
                self.tracks.append(DAT())
        #self.tracks = [self.tracker for i in bboxes]
        for idx, track_tmp in enumerate(self.tracks):
            track_tmp.init(frame,bboxes[idx])

    def update(self,frame):
        boxes_out = []
        for track_tmp in self.tracks:
            boxes_out.append(track_tmp.update(frame))
        return boxes_out


class MoveTrackerRun(object):
    def __init__(self,tracker):
        self.tracker = tracker

    def track_init(self,bbox_xyah):
        means = []
        convariances = []
        for box in bbox_xyah:
            mean,convariance = self.tracker.initiate(box)
            #print(mean)
            means.append(mean)
            convariances.append(convariance)
        self.means_track = means
        self.convariances_track = convariances

    def track_predict(self):
        self.mean_update = []
        self.convariance_update = []
        for mean_tmp, conv_tmp in zip(self.means_track,self.convariances_track):
            mean, convariance = self.tracker.predict(mean_tmp,conv_tmp)
            self.mean_update.append(mean)
            self.convariance_update.append(convariance)
        #print('pred:',np.shape(self.mean_update))
        self.means_track = self.mean_update
        self.convariances_track = self.convariance_update
    
    def track_update(self,detect_box):
        self.means_track = []
        self.convariances_track = []
        idx = 0
        for mean_tmp,conv_tmp in zip(self.mean_update,self.convariance_update):
            #detect_tmp = mean_tmp[:4]
            #detect_tmp[:2] = detect_tmp[:2] +1
            detect_tmp = detect_box[idx]
            idx+=1
            mean, convariance = self.tracker.update(mean_tmp,conv_tmp,detect_tmp)
            #print('updata:',np.shape(mean))
            self.means_track.append(mean)
            self.convariances_track.append(convariance)
        #print('update:',np.shape(self.means_track))