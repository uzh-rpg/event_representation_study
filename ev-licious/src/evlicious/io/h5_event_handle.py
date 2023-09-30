from pathlib import Path
from .utils import event_handle
from .utils.events import Events

import hdf5plugin
import h5py
import numpy as np


def _find_index_from_timestamps(t_query, t_events):
    return np.searchsorted(t_events, t_query+1e-3)


class H5EventHandle(event_handle.EventHandle):
    def __init__(self, handle, height=None, width=None):
        assert "events" in handle.keys()
        assert "x" in handle["events"].keys()
        assert "y" in handle["events"].keys()
        assert "t" in handle["events"].keys()
        assert "p" in handle["events"].keys()
        assert "height" in handle["events"].keys()
        assert "width" in handle["events"].keys()
        assert "divider" in handle["events"].keys()

        self.height = height
        self.width = width
        self.height = handle["events"]["height"][()]
        self.width = handle["events"]["width"][()]
        self.divider = handle['events']['divider'][()]

        self.handle = handle
        self.index = None

    def get_time_limits(self):
        return self.handle["events"]['t'][0], self.handle["events"]['t'][-1]

    def prepare_time_to_index_lut(self, t0_us, t1_us):
        idx0 = _find_index_from_timestamps(t0_us, self.handle['events']['t'])
        idx1 = _find_index_from_timestamps(t1_us, self.handle['events']['t'])
        keys = list(zip(t0_us,t1_us))
        values = list(zip(idx0, idx1))
        self.index = dict(zip(keys, values))

    def find_index_from_timestamp(self, t_us):
        return _find_index_from_timestamps(t_us, self.handle['events']['t'])

    @classmethod
    def from_path(cls, path: Path, height=None, width=None):
        handle = h5py.File(str(path))
        return cls(handle, height=height, width=width)

    def get_between_idx(self, i0, i1):
        return Events(x=self.handle["events"]["x"][i0:i1],
                      y=self.handle["events"]["y"][i0:i1],
                      t=self.handle["events"]["t"][i0:i1],
                      p=self.handle["events"]["p"][i0:i1],
                      height=self.height,
                      width=self.width,
                      divider=self.divider)

    def get_between_time(self, t0_us: int, t1_us: int):
        if self.index is not None and (t0_us, t1_us) in self.index:
            i0, i1 = self.index[(t0_us, t1_us)]
        else:
            i0, i1 = _find_index_from_timestamps(np.array([t0_us, t1_us]), self.handle["events"]["t"])
        return self.get_between_idx(i0, i1)

    def __len__(self):
        return len(self.handle["events"]["t"])

    def compute_time_and_index_windows(self, step_size: int, window: int, step_size_unit: str, window_unit: str):
        assert window_unit in ['nr', 'us']
        assert step_size_unit in ['nr', 'us']
        assert window_unit in ['nr', 'us']
        assert step_size_unit in ['nr', 'us']
        t_handle = self.handle["events"]["t"]

        # first compute i1
        if window_unit == "nr":
            # +1 includes the end timestamp if it can be exactly divided by step_size
            i1 = np.arange(step_size, len(t_handle) + 1, step_size)
            i1_temp = np.clip(i1, 0, len(t_handle)-1)
            timestamps1 = t_handle[i1_temp]
        else:
            t0 = t_handle[0]
            t1 = t_handle[-1]
            # +1 includes the end timestamp if it can be exactly divided by step_size
            timestamps1 = np.arange(t0+step_size, t1+1, step_size)
            i1 = _find_index_from_timestamps(timestamps1, t_handle)

        # second compute i0
        if step_size_unit == "nr":
            i0 = i1 - window
            i0 = np.clip(i0, 0, len(t_handle) - 1)
            i0, inverse = np.unique(i0, return_inverse=True)
            timestamps0 = t_handle[i0]
            timestamps0 = timestamps0[inverse]
        else:
            timestamps0 = timestamps1 - window
            i0 = _find_index_from_timestamps(timestamps0, t_handle)
            i0 = np.clip(i0, 0, len(t_handle) - 1)

        return (timestamps0, timestamps1), (i0, i1)








