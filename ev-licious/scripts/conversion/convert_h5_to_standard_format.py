import numpy as np
import h5py
import argparse

import evlicious
from evlicious.io.utils.events import TYPES
from pathlib import Path

KEY_ALIASES = dict(x=["x"], y=["y"], t=["timestamp", "timestamps", "t"], p=["polarity", "p"])

def _recursive_load(file_handle: h5py.File, query: str):
    if type(file_handle) is h5py.Dataset:
        return None
    if query in file_handle.keys():
        return file_handle[query][()]
    else:
        for key in file_handle.keys():
            result = _recursive_load(file_handle[key], query)
            if result is not None:
                return result
        else:
            return None

def very_lenient_load(input_path: Path, height: int, width: int, divider: int):
    # This function tries its hardest to load the events and cast them to the correct type
    # * loads height, width and divider and if it does not find a value, it assign from cli.
    # * check if fh has format x,y,t,p or events/x, events/y, events/t, events/p
    # * check if other words like polarity, timestamp, timestamps was used
    file_handle = h5py.File(input_path, "r")
    if height < 0:
        height = _recursive_load(file_handle, "height")
        assert height is not None
    if width < 0:
        width = _recursive_load(file_handle, "width")
        assert width is not None
    if divider < 0:
        divider = _recursive_load(file_handle, "divider")
        assert divider is not None

    if "events" in file_handle.keys():
        file_handle = file_handle["events"]

    events = dict(height=int(height), width=int(width), divider=int(divider))
    for k, aliases in KEY_ALIASES.items():
        for alias in aliases:
            if alias in file_handle.keys():
                events[k] = file_handle[alias][()]
                break
        else:
            raise ValueError(f"The file you provided does not have a key for {k}. Searched for {aliases}.")

    return events

def _check_and_modify_coord(x, divider, limit):
    # for x,y check that they do not have floating point values,
    # otherwise, multiply them by 32,
    has_float = np.any(x > x.astype("int32"))
    if has_float:
        divider = 32
        x *= divider

    x_div = x.astype("float32") / divider
    mask = (x_div >= 0) & (x_div <= limit-1) & (x < 2**16)
    assert any(mask)

    return x, divider, mask

def _check_and_modify_timestamps(t):
    # Check in which range dt is
    dt = np.diff(t)
    dt_min = np.min(dt[dt>0])

    # if t is in seconds, this is usually below 0.1
    if dt_min < 0.1:
        t *= 1e6
    # these might be in nanoseconds
    elif dt_min > 100:
        t /= 1e3

    mask = t >= 0

    nz, = np.nonzero(dt < 0)
    if len(nz) > 0:
        mask[int(nz[0]):] = 0

    return t, mask


def check_and_modify_values(events):
    # check if x,y are in floating point
    # check if t is in s, us, ns
    # check if polarity is [0,1] or [-1,1]

    height, width, divider = events['height'], events['width'], events["divider"]

    events['x'], divider, mask_x = _check_and_modify_coord(events['x'], divider, width)
    events['y'], divider, mask_y = _check_and_modify_coord(events['y'], divider, height)
    events['divider'] = divider

    events['t'], mask_t = _check_and_modify_timestamps(events['t'])

    events['p'] = events['p'].astype("int8")
    events['p'][events['p']==0] = -1

    mask = mask_x & mask_y & mask_t

    return events, mask

def fix_file(input_path: Path, output_path: Path, height: int, width: int, divider: int):
    events = very_lenient_load(input_path, height, width, divider)
    events, mask = check_and_modify_values(events)

    opts = evlicious.io.H5_BLOSC_COMPRESSION_FLAGS
    with h5py.File(output_path, "a") as fh_out:
        # order matters, since t will be cast last and takes the most memory in RAM
        for k in ["width", "height", "divider", 'x', 'y', 'p', 't']:
            v = events[k]
            if k in TYPES:
                dtype = TYPES[k]
                v = v[mask].astype(dtype)

            if type(v) is int:
                fh_out.create_dataset(f"events/{k}", data=v)
            else:
                fh_out.create_dataset(f"events/{k}", data=v, **opts)

            # we need this otherwise we may go out of memory
            del events[k]


def FLAGS():
    parser = argparse.ArgumentParser("""Corrects all h5 files with specific pattern. 
                                        This script should be a catch-all for correcting 
                                        legacy datasets which were recorded in h5""")
    
    parser = evlicious.args.add_io_args(parser)
    parser.add_argument("--height", type=int, default=-1)
    parser.add_argument("--width", type=int, default=-1)
    parser.add_argument("--divider", type=int, default=-1)

    flags = parser.parse_args()
    assert flags.events.exists()

    if str(flags.output_path) == ".":
        flags.output_path = flags.output_path / flags.events.name

    return flags


if __name__ == '__main__':
    flags = FLAGS()
    if flags.events.is_dir():
        files = list(flags.events.rglob("*.h5"))
        with evlicious.tools.TaskManager(total=len(files)) as tm:
            for f in files:
                output_path = flags.output_path / f.relative_to(flags.events)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                tm.new_task(fix_file, f, output_path, flags.height, flags.width, flags.divider)
    else:
        assert flags.events.suffix == ".h5"
        fix_file(flags.events, flags.output_path, flags.height, flags.width, flags.divider)
