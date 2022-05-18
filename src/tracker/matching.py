import lap
import numpy as np
from cython_bbox import bbox_overlaps as bbox_ious
from scipy.spatial.distance import cdist

_CHI2INV95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


def linear_assignment(cost_matrix, thresh):
    """
    Linear assignment with threshold.
    """
    if cost_matrix.size == 0:
        out = (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )

        return out
    matches, unmatched_a, unmatched_b = [], [], []
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)

    return matches, unmatched_a, unmatched_b


def iou(atlbrs, btlbrs):
    """
    Compute cost based on IoU.
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU.
    """
    if (atracks and isinstance(atracks[0], np.ndarray)) or \
       (btracks and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    ious_val = iou(atlbrs, btlbrs)
    cost_matrix = 1 - ious_val

    return cost_matrix


def embedding_distance(tracks, detections):
    """
    Compute embedding distance.
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features))  # Nomalized features

    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    """
    Fuses motion objects.
    """
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = _CHI2INV95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean,
            track.covariance,
            measurements,
            only_position,
            metric='maha',
        )
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1-lambda_)* gating_distance

    return cost_matrix
