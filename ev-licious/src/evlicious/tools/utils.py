from evlicious import Events
import numpy as np
from typing import Optional
import numba


def events_to_voxel_grid_cuda(events: Events, num_bins: int, normalize: bool=True, t0_us: Optional[int]=None, t1_us: Optional[int]=None, device="cuda:0"):
    """
    Build a voxel grid with trilinear interpolation in the time and x,y domain from a set of events.
    """
    import torch

    voxel_grid = torch.zeros((num_bins, events.height, events.width), dtype=torch.float32, device=device)

    if len(events) < 2:
        return voxel_grid

    t_torch = torch.from_numpy(events.t).to(device)
    x_torch = torch.from_numpy(events.x.astype("int16")).to(device)
    y_torch = torch.from_numpy(events.y.astype("int16")).to(device)
    p_torch = torch.from_numpy(events.p).to(device)
    
    # normalize the event timestamps so that they lie between 0 and num_bins
    t0_us = t0_us if t0_us is not None else t_torch[0]
    t1_us = t1_us if t1_us is not None else t_torch[-1]
    deltaT = t1_us - t0_us

    if deltaT == 0:
        deltaT = 1.0

    t_norm = (num_bins - 1) * (t_torch - t0_us) / deltaT

    t_norm_int = t_norm.int()
    for tlim in [t_norm_int, t_norm_int+1]:
        mask = (tlim >= 0) & (tlim < num_bins)
        weight = _bil_w_tch(t_norm_int, tlim) * p_torch
        coords = x_torch.long() + events.width * y_torch.long() + events.width * events.height * tlim.long()
        voxel_grid.put_(coords[mask], weight[mask])

    if normalize:
        eps = 1e-5        
        mask_nonzero = voxel_grid.abs() > 0
        if len(mask_nonzero[0]) > 0:
                mean, stddev = voxel_grid[mask_nonzero].mean(), voxel_grid[mask_nonzero].std()
                if stddev > 0:
                    voxel_grid[mask_nonzero] = (voxel_grid[mask_nonzero] - mean) / (eps + stddev)

    return voxel_grid


def events_to_voxel_grid(events: Events, num_bins: int, normalize: bool=True, t0_us: Optional[int]=None, t1_us: Optional[int]=None):
    """
    Build a voxel grid with trilinear interpolation in the time and x,y domain from a set of events.
    """

    voxel_grid = np.zeros((num_bins, events.height, events.width), np.float32)

    if len(events) < 2:
        return voxel_grid
    
    # normalize the event timestamps so that they lie between 0 and num_bins
    t0_us = t0_us if t0_us is not None else events.t[0]
    t1_us = t1_us if t1_us is not None else events.t[-1]
    deltaT = t1_us - t0_us

    if deltaT == 0:
        deltaT = 1.0

    t_norm = (num_bins - 1) * (events.t - t0_us) / deltaT

    t_norm_int = t_norm.astype("int32")
    for tlim in [t_norm_int, t_norm_int+1]:
        mask = (tlim >= 0) & (tlim < num_bins)
        weight = _bil_w(t_norm_int, tlim) * events.p
        _draw_xy_to_voxel_grid(voxel_grid, events.x[mask], events.y[mask], tlim[mask], weight[mask])

    if normalize:
        eps = 1e-5        
        mask_nonzero = np.nonzero(voxel_grid)
        if mask_nonzero[0].size > 0:
                mean, stddev = voxel_grid[mask_nonzero].mean(), voxel_grid[mask_nonzero].std()
                if stddev > 0:
                    voxel_grid[mask_nonzero] = (voxel_grid[mask_nonzero] - mean) / (eps + stddev)

    return voxel_grid

def _bil_w(x, x_int):
    return 1 - np.abs(x_int- x)

def _bil_w_tch(x, x_int):
    return 1 - (x - x_int.float()).abs()

def _draw_xy_to_voxel_grid(voxel_grid, x, y, b, value):
    if x.dtype == np.uint16:
        _draw_xy_to_voxel_grid_int(voxel_grid, x, y, b, value)
        return

    x_int = x.astype("int32")
    y_int = y.astype("int32")
    for xlim in [x_int, x_int + 1]:
        for ylim in [y_int, y_int + 1]:
            weight = _bil_w(x, xlim) * _bil_w(y, ylim)
            _draw_xy_to_voxel_grid_int(voxel_grid, xlim, ylim, b, weight * value)

def _draw_xy_to_voxel_grid_int(voxel_grid, x, y, b, value):
    B, H, W = voxel_grid.shape
    mask = (x >= 0) & (y >= 0) & (x < W) & (y < H)
    np.add.at(voxel_grid, (b[mask], y[mask], x[mask]), value[mask])

def resize_to_resolution(events: Events, height: int, width: int, chunks:int =1, pbar=None):
    # this subsamples events if they were generated with cv2.INTER_AREA
    change_map = np.zeros((height, width), dtype="float32")
    fx = int(events.width / width)
    fy = int(events.height / height)

    rescaled_events_list = []
    for sub_events in _fixed_chunk_iterator(events, chunks, pbar):
        mask = np.zeros(shape=(len(sub_events.x),), dtype="bool")
        mask, change_map = _filter_events_resize(sub_events.x, sub_events.y, sub_events.p, mask, change_map, fx, fy)
        rescaled_events = rescale_coordinates(sub_events[mask], 1.0/fx, 1.0/fy)
        rescaled_events_list.append(rescaled_events)

    return stack(rescaled_events_list)

def _fixed_chunk_iterator(events: Events, num_chunks: int, pbar=None):
    num_events_per_chunk = len(events)//num_chunks
    for i in range(num_chunks):
        pbar is not None and pbar.update(1)
        yield events[i*num_events_per_chunk:(i+1)*num_events_per_chunk]
    if num_chunks*num_events_per_chunk < len(events):
        pbar is not None and pbar.update(1)
        yield events[num_chunks*num_events_per_chunk:len(events)]

def rescale_coordinates(events: Events, fx: float, fy: float) -> Events:
    return Events(x=(events._x * fx).astype("uint16"), 
                  y=(events._y * fy).astype("uint16"), 
                  p=events.p, t=events.t, 
                  width=int(events.width * fx), 
                  height=int(events.height * fy),
                  divider=events.divider)


@numba.jit(nopython=True, cache=True)
def _filter_events_resize(x, y, p, mask, change_map, fx, fy):
    # iterates through x,y,p of events, and increments cells of size fx x fy by 1/(fx*fy) 
    # if one of these cells reaches +-1, then reset the cell, and pass through that event.
    # for memory reasons, this only returns the True/False for every event, indicating if 
    # the event was skipped or passed through.
    for i in range(len(x)):
        x_l = x[i] // fx
        y_l = y[i] // fy
        change_map[y_l, x_l] += p[i] * 1.0 / (fx * fy)

        if np.abs(change_map[y_l, x_l]) >= 1:
            mask[i] = True
            change_map[y_l, x_l] -= p[i]

    return mask, change_map


def stack(events_list):
    return Events(x=np.concatenate([e._x for e in events_list], axis=0),
                  y=np.concatenate([e._y for e in events_list], axis=0),
                  t=np.concatenate([e.t for e in events_list], axis=0),
                  p=np.concatenate([e.p for e in events_list], axis=0),
                  width=events_list[0].width,
                  height=events_list[0].height,
                  divider=events_list[0].divider)

@numba.jit(nopython=True)
def _background_activity_filter(mask, timestamps, x, y, t, depth_us, radius=1):
    for i, (x_, y_, t_) in enumerate(zip(x, y, t)):
        t_last = timestamps[y_,x_]
        discard = t_last > 0 and t_ - t_last > depth_us
        mask[i] = not discard
        xmin = max([x_-radius,0])
        ymin = max([y_-radius,0])
        timestamps[ymin:y_+radius, xmin:x_+radius] = t_
    return mask

def split_into_positive_negative(events: Events):
    return events[events.p==1], events[events.p==-1]

@numba.jit(nopython=True)
def _contrast_threshold_control(activity, mask, x, y, p, factor):
    for i, (x_, y_, p_) in enumerate(zip(x, y, p)):
        activity[y_, x_] += p_
        if np.abs(activity[y_, x_]) >= factor:
            mask[i] = True
            activity[y_, x_] = 0
    return mask

@numba.jit(nopython=True)
def _refractory_period(mask, x, y, t, period, last_timestamp):
    for i in range(len(x)):
        if t[i] - last_timestamp[y[i],x[i]] < period:
            mask[i] = False
            continue
        last_timestamp[y[i],x[i]] = t[i]
    return mask
