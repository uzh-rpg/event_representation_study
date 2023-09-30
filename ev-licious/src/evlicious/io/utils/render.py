import numpy as np
import cv2
import numba
import matplotlib.pyplot as plt
import enum
import evlicious


class RenderingType(enum.IntEnum):
    RED_BLUE_OVERLAP = enum.auto()
    RED_BLUE_NO_OVERLAP = enum.auto()
    BLACK_WHITE_NO_OVERLAP = enum.auto()
    TIME_SURFACE = enum.auto()
    EVENT_FRAME = enum.auto()


def _render(events, rendering, rendering_type, cast=True):
    if rendering_type == RenderingType.RED_BLUE_OVERLAP:
        return _render_red_blue_overlap(events, rendering, cast=cast)
    elif rendering_type == RenderingType.RED_BLUE_NO_OVERLAP:
        return _render_no_overlap(events, rendering, color="red_blue", cast=cast)
    elif rendering_type == RenderingType.BLACK_WHITE_NO_OVERLAP:
        return _render_no_overlap(events, rendering, color="black_white", cast=cast)
    elif rendering_type == RenderingType.TIME_SURFACE:
        return _render_timesurface(events)
    elif rendering_type == RenderingType.EVENT_FRAME:
        return _render_event_frame(events)


def _render_event_frame(events):
    img = np.zeros(shape=(events.height, events.width), dtype="float32")
    img = _aggregate(img, events.x, events.y, events.p)
    img_rendered = 20 * img + 128
    return cv2.cvtColor(np.clip(img_rendered, 0, 255).astype("uint8"), cv2.COLOR_GRAY2BGR)

def _aggregate(img, x, y, v):
    if x.dtype == np.float32 or x.dtype == np.float64:
        _aggregate_float(img, x, y, v)
    else:
        _aggregate_int(img, x, y, v)
    return img

def _aggregate_float(img, x, y, v):
    H, W = img.shape
    x_ = x.astype("int32")
    y_ = y.astype("int32")
    for xlim in [x_, x_+1]:
        for ylim in [y_, y_+1]:
            mask = (xlim >= 0) & (ylim >= 0) & (xlim < W) & (ylim < H)
            weight = (1 - np.abs(xlim - x)) * (1 - np.abs(ylim - y))
            _aggregate_int(img, xlim[mask], ylim[mask], v[mask] * weight[mask])

def _aggregate_int(img, x, y, v):
    np.add.at(img, (y, x), v)

def _jitter_events(events):
    x = events.x.astype("uint16")
    y = events.y.astype("uint16")
    events_list = []
    for xlim in [x, x+1]:
        for ylim in [y, y+1]:
            mask = _is_in_rectangle(xlim, ylim, (events.height, events.width))
            events_list.append(
                evlicious.Events(x=xlim[mask], y=ylim[mask],
                                 p=events.p[mask],
                                 t=events.t[mask],
                                 divider=1,
                                 height=events.height,
                                 width=events.width)
            )
    return evlicious.tools.stack(events_list)

def _render_red_blue_overlap(events, rendering, cast=True):
    white_canvas = np.full(shape=(events.height, events.width, 3), fill_value=255, dtype="uint8")
    rendering = rendering.copy() if rendering is not None else white_canvas
    mask = _is_in_rectangle(events.x, events.y, rendering.shape[:2])
    events = events[mask]

    if not cast:
        events = _jitter_events(events)

    rendering[events.y, events.x, :] = 0
    rendering[events.y, events.x, events.p + 1] = 255

    return rendering

def _render_no_overlap(events, rendering, color="red_blue", cast=True):
    white_canvas = np.full(shape=(events.height, events.width, 3), fill_value=255, dtype="uint8")
    rendering = rendering.copy() if rendering is not None else white_canvas
    H, W = rendering.shape[:2]
    mask = (events.x >= 0) & (events.y >= 0) & (events.x <= W - 1) & (events.y <= H - 1)
    events = events[mask]

    if not cast:
        events = _jitter_events(events)

    red = np.array([255, 0, 0])
    blue = np.array([0,0,255])
    black = np.array([0,0,0])
    white = np.array([255,255,255])

    if color == "red_blue":
        pos = blue
        neg = red
    elif color == "black_white":
        pos = white
        neg = black

    visited_mask = np.ones(shape=(events.height, events.width)) == 0
    return _render_no_overlap_numba(rendering,
                                    visited_mask,
                                    events.x[::-1],
                                    events.y[::-1],
                                    events.p[::-1],
                                    pos, neg)

@numba.jit(nopython=True)
def _render_no_overlap_numba(rendering, mask, x, y, p, pos_color, neg_color):
    for x_, y_, p_ in zip(x, y, p):
        if not mask[y_, x_]:
            rendering[y_, x_] = pos_color if p_ > 0 else neg_color
            mask[y_, x_] = True
    return rendering

def _render_timesurface(events):
    image = np.zeros((events.height, events.width), dtype="float32")
    cm = plt.get_cmap("jet")
    tau = 3e4
    t = events.t.astype("int")

    if len(events.x) > 2:
        value = np.exp(-(t[-1]-t)/float(tau))
        _aggregate(image, events.x, events.y, value)
        image = cm(image)
    else:
        image = image.astype("uint8")

    return image

def _is_in_rectangle(x, y, shape):
    return (x >= 0) & (y >= 0) & (x <= shape[1]-1) & (y <= shape[0]-1)