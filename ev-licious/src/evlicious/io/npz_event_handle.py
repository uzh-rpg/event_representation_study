from pathlib import Path
from .utils import event_handle
from .utils.events import Events
import glob

import numpy as np
import zipfile
import evlicious


def npz_len(npz):
    """
    Takes a path to an .npz file, which is a Zip archive of .npy files.
    Generates a sequence of (name, shape, np.dtype).
    """
    with zipfile.ZipFile(npz) as archive:
        for name in archive.namelist():
            if not name.endswith('.npy'):
                continue

            npy = archive.open(name)
            version = np.lib.format.read_magic(npy)
            shape, fortran, dtype = np.lib.format._read_array_header(npy, version)
            return shape[0]




class NPZEventHandle(event_handle.EventHandle):
    def __init__(self, files, height, width, divider=1, strict=False):
        self.height = height
        self.width = width
        self._p_key, self._t_key, self._x_key, self._y_key = sorted(list(np.load(files[0]).keys()))
        self.files = files
        self.divider = divider
        self.strict = strict

        self.files = [f for f in self.files if npz_len(f) > 0]
        self.npz_lens = np.array([npz_len(f) for f in self.files])
        self.npz_times = np.array([np.load(f)[self._t_key][-1] for f in self.files])
        self.npz_cumsum = np.cumsum(self.npz_lens)

    def _npz_to_events(self, file, height, width, divider=1):
        fh = np.load(file)
        p = fh[self._p_key].astype("int8")
        p[p==0] = -1
        x = fh[self._x_key].astype("uint16")
        y = fh[self._y_key].astype("uint16")
        t = fh[self._t_key].astype("int64")

        if not self.strict:
            mask = (x >= 0) & (y >= 0) & (x <= (width-1) * divider) & (y <= (height-1) * divider)
            x = x[mask]
            y = y[mask]
            p = p[mask]
            t = t[mask]

        return Events(x=x, y=y, t=t, p=p, width=width, height=height, divider=divider)

    def get_time_limits(self):
        return np.load(self.files[0])[self._t_key][0], np.load(self.files[-1])[self._t_key][-1]

    @classmethod
    def from_path(cls, path: Path, height, width, **kwargs):
        files = sorted(glob.glob(str(path / "*.npz")))
        return cls(files, height=height, width=width, **kwargs)

    def get_between_idx(self, i0, i1):
        idx0 = np.searchsorted(self.npz_cumsum, i0)
        idx1 = np.searchsorted(self.npz_cumsum, i1)

        events = evlicious.tools.stack([self._npz_to_events(self.files[i], height=self.height,
                                                                 width=self.width, divider=self.divider) for
                       i in range(idx0, idx1+1)])
        offset = (self.npz_cumsum[idx0-1] if idx0 > 0 else 0)
        i0 -= offset
        i1 -= offset

        return events[i0:i1]

    def get_between_time(self, t0_us: int, t1_us: int):
        idx0 = np.searchsorted(self.npz_times, t0_us)
        idx1 = np.searchsorted(self.npz_times, t0_us)
        events = evlicious.tools.stack([self._npz_to_events(self.files[i], height=self.height,
                                                            width=self.width, divider=self.divider) for
                                        i in range(idx0, idx1 + 1)])
        return evlicious.tools.mask(events, (events.t <= t1_us) & (events.t >= t0_us))

    def __len__(self):
        return self.npz_cumsum[-1]








