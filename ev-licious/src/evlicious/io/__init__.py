from .h5_event_handle import H5EventHandle
from .npz_event_handle import NPZEventHandle
from .npy_event_handle import NPYEventHandle
from .dat_event_handle import DatEventHandle

try:
    from .rosbag_event_handle import RosbagEventHandle
    from .utils.rosbag import events_to_ros_message
except ImportError: 
    print("Cannot use ros api")

from .utils.h5_writer import H5Writer, H5_BLOSC_COMPRESSION_FLAGS
from .utils.fake_events import generate_fake_events
from .bin_event_handle import BinEventHandle
from pathlib import Path


def load_events_from_path(path: Path, height=None, width=None, divider=1):
    if path.suffix  == ".h5":
        return H5EventHandle.from_path(path)
    elif path.suffix == ".dat":
        return DatEventHandle.from_path(path)
    elif path.suffix == ".bag":
        return RosbagEventHandle.from_path(path)
    elif path.suffix == ".bin":
        return BinEventHandle.from_path(path, height, width, divider)
    elif path.is_dir():
        for f in path.iterdir():
            if f.is_file():
                if f.suffix == ".npz":
                    return NPZEventHandle.from_path(path, height=height, width=width, divider=divider)
                elif f.suffix == ".npy":
                    return NPYEventHandle.from_path(path, height=height, width=width, divider=divider)
    else:
        raise ValueError

def save_events(path, events):
    writer = H5Writer(path)
    writer.add_data(events)


