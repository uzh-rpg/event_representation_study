from pathlib import Path
import sys
sys.path.append("/usr/lib/python3/dist-packages/")

from .utils.prophesee_utils import _cd_events_to_standard_format, EventDatReader
from .utils.event_handle import EventHandle


class DatEventHandle(EventHandle):
    def __init__(self, reader, height, width, divider=1, strict=True):
        self.height = height
        self.width = width

        self.divider = divider
        self.strict = strict
        self.reader = reader

    @classmethod
    def from_path(cls, path: Path, **kwargs):
        reader = EventDatReader(str(path))
        height, width = reader.get_size()
        return cls(reader, height=height, width=width, divider=1, **kwargs)

    def get_between_idx(self, i0, i1):
        self.reader.seek_event(i0)
        events = self.reader.load_n_events(i1 - i0)
        return _cd_events_to_standard_format(events, self.height, self.width)

    def get_between_time(self, t0_us: int, t1_us: int):
        self.reader.seek_time(t0_us)
        events = self.reader.load_delta_t(t1_us - t0_us)
        return _cd_events_to_standard_format(events, self.height, self.width)

    def __len__(self):
        return self.reader.event_count()
