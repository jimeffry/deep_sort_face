# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import sys
import os
import numpy as np
from nn_matching import NearestNeighborDistanceMetric
from kalman_filter import KalmanFilter
from linear_assignment import confirmedTracks_matching,min_cost_matching
from iou_matching import iou_cost
from track import Track
from utils_mot import gate_cost_matrix
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs

class Tracker(object):
    """
    This is the multi-target tracker.
    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.
    """
    def __init__(self):
        self.metric = NearestNeighborDistanceMetric("cosine", cfgs.max_cosine_distance, cfgs.feature_max_keep)
        self.max_age = cfgs.max_age
        self.n_init = cfgs.confirm_frame_cnt
        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.
        This function should be called once every time step, before `update`.
        """
        #print('tracker pred:',len(self.tracks))
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.
        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.
        """
        if cfgs.debug:
            print('updata_detect:',len(detections))
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(detections)
        if cfgs.debug:
            print('unmatch_track for delete:',len(unmatched_tracks))
            print('unmatch_det for init track:',len(unmatched_detections))
        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        if cfgs.debug:
            print('activate:',len(active_targets))
        features, targets = [], []
        for track in self.tracks:
            #print('track_feature',len(track.features))
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        '''
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)
            return cost_matrix
        '''
        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        if cfgs.debug:
            print('confirm and unconfirm:',len(confirmed_tracks),len(unconfirmed_tracks))
        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            confirmedTracks_matching(
                self.metric,self.kf, self.max_age,self.tracks, detections, confirmed_tracks)
        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        ## unconfirmed tracks and unmatched detections for last match
        cost_matrix = iou_cost(self.tracks,detections,iou_track_candidates,unmatched_detections)
        matches_b, unmatched_tracks_b, unmatched_detections = \
            min_cost_matching(cost_matrix,cfgs.max_iou_distance,self.tracks,detections, iou_track_candidates, unmatched_detections)
        if cfgs.debug:
            print('iou mach tracks:: unconfirm and unmatch_track:',iou_track_candidates)
            print("matcha and matchb",len(matches_a),len(matches_b))
            print('a:',matches_a)
            print('b:',matches_b)
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1
