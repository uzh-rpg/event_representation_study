import numpy as np
from .events import Events


def generate_fake_events(optical_flow=(10, 0), N=10000, circle_radius=5, starting_point=(10,10), resolution=(30,30)):
    """
    optical flow: 2 numbers, vx, vy denoting the flow in x and y direction.
    N: Number of events
    Example:
        events = optical_flow([5, 6], 10000)
    """
    vx, vy = optical_flow

    # sample t and angle from uniform distribution
    time = np.sort(np.random.random((N,))).reshape((N,1))
    angle = np.random.random((N,1)) * 2*np.pi
    polarity = 2*(np.random.random((N,1)) > .5)-1

    # compute coordinates
    u0, v0 = starting_point
    x = (u0 + time * vx + np.cos(angle) * circle_radius).astype(np.int64)
    y = (v0 + time * vy + np.sin(angle) * circle_radius).astype(np.int64)

    # compute mask for events that are within image
    H, W = resolution
    mask = (x[:,0] >= 0) & (y[:,0] >= 0) & (x[:,0] < W) & (y[:,0] < H)

    events = np.concatenate([x[mask, :], y[mask, :], 1e6*time[mask, :], polarity[mask, :]], 1)

    return Events.from_array(events, format="xytp")