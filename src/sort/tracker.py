# vim: expandtab:ts=4:sw=4
import sys
import os
import numpy as np
from nn_matching import NearestNeighborDistanceMetric
from kalman_filter import KalmanFilter
from linear_assignment import matching_cascade,min_cost_matching
from iou_matching import iou_cost
from track import Track
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

    def __init__(self, metric,max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1
        self.last_confirm_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        print('tracker pred:',len(self.tracks))
        for track in self.tracks:
            track.predict(self.kf)
            #print('run predict')

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        print('updata_detect:',len(detections))
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)
        print('unmatch_track for delete:',len(unmatched_tracks))
        print('unmatch_det for init track:',len(unmatched_detections))
        # Update track set.
        for track_idx, detection_idx in matches:
            #print("run match")
            self.last_confirm_id = self.tracks[track_idx].update(
                self.kf, detections[detection_idx],self.last_confirm_id)
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
            #print("run unmatch_track for delete")
        for detection_idx in unmatched_detections:
            #print("run unmatch_det for init track")
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        print('activate:',len(active_targets))
        features, targets = [], []
        for track in self.tracks:
            #print('track_feature',len(track.features))
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        print('confirm and unconfirm:',len(confirmed_tracks),len(unconfirmed_tracks))
        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            matching_cascade(self.metric,self.kf, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)
        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        cost_matrix = iou_cost(self.tracks,detections, iou_track_candidates, unmatched_detections)
        matches_b, unmatched_tracks_b, unmatched_detections = \
            min_cost_matching(
                cost_matrix, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)
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
