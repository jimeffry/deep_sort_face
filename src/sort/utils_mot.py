import numpy as np
from kalman_filter import chi2inv95
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs

def feature2Maha_metric(tracks,dets,track_indices,detection_indices,metric,kalman):
    '''
    func: 
        calculate the cosin distance of feature and Mahalanobis distance
    return: 
        the cosin distacne matrix with filtered by Mahalanobis distance threshold
    '''
    features = np.array([dets[i].feature for i in detection_indices])
    targets = np.array([tracks[i].track_id for i in track_indices])
    cost_matrix = metric.distance(features, targets)
    maha_distance = gate_cost_matrix(kalman, cost_matrix, tracks, dets, track_indices,detection_indices)
    return maha_distance

def gate_cost_matrix(
        kf, cost_matrix, tracks, detections, track_indices, detection_indices,only_position=False):
    """Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Parameters
    ----------
    kf : The Kalman filter.
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[track_indices[i]]` and
        `detections[detection_indices[j]]`.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.

    Returns
    -------
    ndarray
        Returns the modified cost matrix.

    """
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray(
        [detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = cfgs.INFTY_COST
    return cost_matrix