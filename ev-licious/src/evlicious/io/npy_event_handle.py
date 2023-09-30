from pathlib import Path
from .utils import event_handle
from .utils.events import Events
import glob

import numpy as np
import evlicious


def npz_len(npz):
    """
    Takes a path to an .npz file, which is a Zip archive of .npy files.
    Generates a sequence of (name, shape, np.dtype).
    """
    return len(np.load(npz))


class NPYEventHandle(event_handle.EventHandle):
    def __init__(self, files, height, width, divider=1, strict=True):
        self.height = height
        self.width = width
        self.files = files
        self.divider = divider
        self.strict = strict

        self.files = [f for f in self.files if npz_len(f) > 0]
        self.npz_lens = np.array([npz_len(f) for f in self.files])
        self.npz_times = np.array([np.load(f)[-1,2]//1e3 for f in self.files])
        self.npz_cumsum = np.cumsum(self.npz_lens)

    def _npy_to_events(self, file, height, width, divider=1):
        fh = np.load(file)
        x, y, t, p = fh.T
        p[p==0] = -1
        t  = (t // 1000).astype("int64")

        return Events(x=x.astype("uint16"),
                      y=y.astype("uint16"),
                      t=t,
                      p=p.astype("int8"),
                      width=width,
                      height=height,
                      divider=divider)

    def get_time_limits(self):
        return np.load(self.files[0])[0,2]//1e3, np.load(self.files[-1])[-1,2]//1e3

    @classmethod
    def from_path(cls, path: Path, height, width, **kwargs):
        files = sorted(glob.glob(str(path / "*.npy")))
        return cls(files, height=height, width=width, **kwargs)

    def get_between_idx(self, i0, i1):
        idx0 = np.searchsorted(self.npz_cumsum, i0)
        idx1 = np.searchsorted(self.npz_cumsum, i1)

        events = evlicious.tools.stack([self._npy_to_events(self.files[i], height=self.height,
                                                            width=self.width, divider=self.divider) for
                       i in range(idx0, idx1+1)])
        offset = (self.npz_cumsum[idx0-1] if idx0 > 0 else 0)
        i0 -= offset
        i1 -= offset

        return events[i0:i1]

    def get_between_time(self, t0_us: int, t1_us: int):
        idx0 = np.searchsorted(self.npz_times, t0_us)
        idx1 = np.searchsorted(self.npz_times, t0_us)
        events = evlicious.tools.stack([self._npy_to_events(self.files[i], height=self.height,
                                                            width=self.width, divider=self.divider) for
                                        i in range(idx0, idx1 + 1)])
        return events[(events.t <= t1_us) & (events.t >= t0_us)]

    def __len__(self):
        return self.npz_cumsum[-1]








