from matplotlib.pyplot import hot
import numpy as np
import enum
from evlicious import Events
from .utils import _background_activity_filter, _contrast_threshold_control, _refractory_period


class Filtering_Type(enum.IntEnum):
    BackgroundActivity = enum.auto()
    Random = enum.auto()
    ContrastThresholdIncrease = enum.auto()
    RefractoryPeriod = enum.auto()
    HotPixel = enum.auto()

    @classmethod
    def summary(cls):
        summary = ""
        for name, t in cls.__members__.items():
            summary += f" {name}={int(t)} "
        return summary


class HotPixel:
    def __init__(self):
        self.hot_pixel_mask = None

    def calibrate(self, events, debug=False, threshold=0.6):
        count = np.zeros(shape=(events.height, events.width))
        np.add.at(count, (events.y, events.x), np.ones_like(events.p))
        if debug:
            import matplotlib.pyplot as plt
            sorted_counts = np.sort(count.ravel(),)
            sorted_counts /= sorted_counts[-1]
            fig, ax = plt.subplots(ncols=2)
            ax[0].imshow(count)
            ax[1].scatter(np.arange(len(sorted_counts)), sorted_counts)
            ax[1].plot([0,len(sorted_counts)], [threshold, threshold])
            plt.show()

        mask = count / np.max(count) < threshold
        min_count_hotpixel = np.min(count[~mask])
        max_count_non_hotpixel = np.max(count[mask])
        if float(min_count_hotpixel) / max_count_non_hotpixel > 2:
            return mask
        else:
            return np.ones(shape=(events.height, events.width)) > 0

        return hot_pixel_mask

    def insert(self, events):
        if self.hot_pixel_mask is None:
            self.hot_pixel_mask = self.calibrate(events)
        mask = self.hot_pixel_mask[events.y, events.x]
        return events[mask]   


class BackgroundActivity:
    def __init__(self, depth_us, radius):
        self.radius = radius
        self.depth_us = depth_us
        self.timestamps = None

    def insert(self, events: Events):
        if self.timestamps is None:
            self.timestamps = np.full(shape=(events.height, events.width), fill_value=-np.inf)
        mask = np.ones_like(events.x) > 0
        event_mask = _background_activity_filter(mask, self.timestamps, events.x, events.y, events.t, self.depth_us, self.radius)
        return events[event_mask]


class Random:
    def __init__(self, random_downsampling_factor):
        self.random_downsampling_factor = random_downsampling_factor

    def insert(self, events: Events):
        remaining = len(events) // self.random_downsampling_factor
        indices = np.random.choice(len(events), remaining, replace=False)
        return events[indices]


class ContrastThresholdIncrease:
    def __init__(self, contrast_threshold_multiplier):
        self.contrast_threshold_multiplier = contrast_threshold_multiplier
        self.counter_map = None

    def insert(self, events: Events):
        if self.counter_map is None:
            self.counter_map = np.zeros(shape=(events.height, events.width), dtype="int32")

        mask = np.ones_like(events.x) < 0
        mask = _contrast_threshold_control(self.counter_map, mask, events.x, events.y, events.p,
                                           self.contrast_threshold_multiplier)

        return events[mask]


class RefractoryPeriod:
    def __init__(self, depth_us):
        self.depth_us = depth_us
        self.timestamps = None

    def insert(self, events: Events):
        if self.timestamps is None:
            self.timestamps = np.full(shape=(events.height, events.width), fill_value=-np.inf)

        mask = np.ones_like(events.x) > 0
        mask = _refractory_period(mask, events.x, events.y, events.t, self.depth_us, self.timestamps)

        return events[mask]


def from_flags(flags):
    if flags.filter_type == int(Filtering_Type.BackgroundActivity):
        assert flags.depth_us > 0
        assert flags.radius > 0
        return BackgroundActivity(depth_us=flags.depth_us, radius=flags.radius)
    elif flags.filter_type == int(Filtering_Type.Random):
        assert flags.random_downsampling_factor > 0
        return Random(random_downsampling_factor=flags.random_downsampling_factor)
    elif flags.filter_type == int(Filtering_Type.ContrastThresholdIncrease):
        assert flags.contrast_threshold_multiplier > 0
        return ContrastThresholdIncrease(contrast_threshold_multiplier=flags.contrast_threshold_multiplier)
    elif flags.filter_type == int(Filtering_Type.RefractoryPeriod):
        assert flags.depth_us > 0
        return RefractoryPeriod(depth_us=flags.depth_us)
    elif flags.filter_type == int(Filtering_Type.HotPixel):
        return HotPixel()
    else:
        raise ValueError("Filter unknown")


def to_dict(cls):
    a = 2


if __name__ == '__main__':
    print(to_dict(Filtering_Type))

