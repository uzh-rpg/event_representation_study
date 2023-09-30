import abc
from pathlib import Path
import numpy as np
import tqdm

from .h5_writer import H5Writer
from .visualization import Visualizer


class EventHandle:
    @classmethod
    @abc.abstractmethod
    def from_path(cls, path: Path, height=None, width=None):
        raise NotImplemented

    @abc.abstractmethod
    def get_between_time(self, t0_us: int, t1_us: int):
        raise NotImplemented

    @abc.abstractmethod
    def get_between_idx(self, i0, i1):
        raise NotImplemented

    @abc.abstractmethod
    def compute_time_and_index_windows(self, step_size: int, window: int, step_size_unit: str, window_unit: str):
        raise NotImplemented

    @abc.abstractmethod
    def __len__(self):
        raise NotImplemented

    def to(self, path: Path):
        window_size = 100000
        writer = H5Writer(path)

        idx = np.arange(0, len(self), window_size).tolist() + [len(self)]
        idx0, idx1 = idx[:-1], idx[1:]
        for (i0_, i1_) in zip(idx0, idx1):
            events = self.get_between_idx(i0_, i1_)
            writer.add_data(events)

    def __getitem__(self, item: slice):
        assert type(item) is slice
        assert item.step is None
        return self.get_between_idx(item.start, item.stop)

    def interactive_viz_loop(self):
        visualizer = Visualizer(self)
        visualizer.visualizationLoop()

    def iterator(self, step_size: int, window: int, step_size_unit: str, window_unit: str, pbar=False):
        _, (idx0, idx1) = self.compute_time_and_index_windows(step_size, window, step_size_unit, window_unit)
        if pbar:
            pbar = tqdm.tqdm(total=len(idx0))
        for (i0_, i1_) in zip(idx0, idx1):
            if pbar:
                pbar.update(1)
            yield self.get_between_idx(i0_, i1_)

    def load(self):
        return self.get_between_idx(0, len(self))



