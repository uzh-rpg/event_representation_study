from dataclasses import dataclass
import numpy as np

from .render import _render, RenderingType
from pathlib import Path

TYPES = dict(_x=np.uint16, _y=np.uint16, t=np.int64, p=np.int8,
             x=np.uint16, y=np.uint16)


@dataclass(frozen=False)
class Events:
    def __init__(self, x, y, t, p, width, height, divider=1):
        self._x = x
        self._y = y
        self.t = t
        self.p = p
        self.width = width
        self.height = height
        self.divider = divider

        for k, t in TYPES.items():
            if not k in ['x', 'y']:
                assert getattr(self, k).dtype == t, f"Field {k} does not have type {t}, but {getattr(self, k).dtype}."
                
        assert self.x.shape == self.y.shape == self.p.shape == self.t.shape
        assert self.x.ndim == 1

        if self._x.size > 0:
            assert np.max(self.p) <= 1
            self.p[self.p==0] = -1
            assert np.max(self.x) <= self.width-1, np.max(self.x)
            assert np.max(self.y) <= self.height-1, np.max(self.y)
            assert np.min(self.x) >= 0
            assert np.min(self.y) >= 0

    @property
    def x(self):
        if self.divider > 1:
            return self._x.astype("float32") / self.divider
        return self._x

    @property
    def y(self):
        if self.divider > 1:
            return self._y.astype("float32") / self.divider
        return self._y

    def __len__(self):
        return len(self.x)

    def to_dict(self, format="xytp"):
        return {k: getattr(self, k) for k in format}

    def to_array(self, format="xytp"):
        return np.stack([getattr(self, k) for k in format], axis=-1)

    def render(self, rendering=None, rendering_type=RenderingType.RED_BLUE_OVERLAP, cast=True):
        return _render(self, rendering, rendering_type, cast)

    def __getitem__(self, item):
        return Events(x=self._x[item].copy(),
                      y=self._y[item].copy(),
                      t=self.t[item].copy(),
                      p=self.p[item].copy(),
                      width=self.width,
                      height=self.height,
                      divider=self.divider)

    def iter_events(self, format="xytp"):
        for i in range(len(self)):
            yield [getattr(self, k)[i] for k in format]

    def to(self, path: Path):
        from .h5_writer import H5Writer
        writer = H5Writer(path)
        writer.add_data(self)

    @classmethod
    def from_dict(cls, dictionary, height=-1, width=-1, divider=-1, format="xytp"):
        array = np.stack([dictionary[k] for k in format], axis=-1)
        return cls.from_array(array, format=format, height=height, width=width, divider=divider)

    @classmethod
    def from_array(cls, array, format="xytp", height=-1, width=-1, divider=1):
        data_dict = {}
        for k, data in zip(format, array.T):
            data_dict[k] = data.astype(TYPES[k])

        if width < 0:
            width = int(data_dict['x'].max())+1
        if height < 0:
            height = int(data_dict['y'].max())+1

        return cls(**data_dict, height=height, width=width, divider=divider)



