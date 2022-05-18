from collections import OrderedDict

import numpy as np


class TrackState:
    new = 0
    tracked = 1
    lost = 2
    removed = 3


class BaseTrack:
    """
    Track class template.
    """
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.new

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.lost

    def mark_removed(self):
        self.state = TrackState.removed
