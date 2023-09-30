from pathlib import Path
import numpy as np

from .utils.event_handle import EventHandle
from .utils.events import Events
from .h5_event_handle import _find_index_from_timestamps


class DataEventHandle(EventHandle):
    def __init__(self, data, height, width, divider=1):
        self.height = height
        self.width = width

        self.divider = divider
        self.data = data

    def get_between_idx(self, i0, i1):
        return Events.from_array(self.data[i0:i1], height=self.height, width=self.width, divider=self.divider)

    def get_between_time(self, t0_us: int, t1_us: int):
        t = self.data[:,2]
        idx0 = _find_index_from_timestamps(t0_us, t)
        idx1 = _find_index_from_timestamps(t1_us, t)
        return self.get_between_idx(idx0, idx1)

    def __len__(self):
        return len(self.data)


class BinEventHandle(DataEventHandle):
    @classmethod
    def from_path(cls, path: Path, height, width, divider=1):
        data = load_bin(path)
        return DataEventHandle(data, height=height, width=width, divider=divider)
        

def load_bin(path: Path) -> np.array:
    f = open(str(path), 'rb')
    raw_data = np.fromfile(f, dtype=np.uint8)
    f.close()

    raw_data = np.uint32(raw_data)
    all_y = raw_data[1::5]
    all_x = raw_data[0::5]
    all_p = (raw_data[2::5] & 128) >> 7  # bit 7
    all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
    all_p = all_p.astype(np.float64)
    all_p[all_p == 0] = -1
    events = np.column_stack((all_x, all_y, all_ts, all_p))

    return events
