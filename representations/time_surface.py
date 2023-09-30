from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
import numba


@dataclass(frozen=True)
class ToTimesurface:
    """Create global or local time surfaces for each event. Modeled after the paper Lagorce et al.
    2016, Hots: a hierarchy of event-based time-surfaces for pattern recognition
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7508476.
    Parameters:
        sensor_size: a 3-tuple of x,y,p for sensor_size
        surface_dimensions (int, int): width does not have to be equal to height, however both numbers have to be odd.
            if surface_dimensions is None: the time surface is defined globally, on the whole sensor grid.
        tau (float): time constant to decay events around occuring event with.
        decay (str): can be either 'lin' or 'exp', corresponding to linear or exponential decay.
    """

    sensor_size: Tuple[int, int, int]
    surface_dimensions: Union[None, Tuple[int, int]] = None
    tau: float = 5e3
    decay: str = "lin"

    def __call__(self, events, indices):
        timestamp_memory = np.zeros(
            (self.sensor_size[2], self.sensor_size[1], self.sensor_size[0])
        )
        timestamp_memory -= self.tau * 3 + 1
        all_surfaces = np.zeros(
            (
                len(indices),
                self.sensor_size[2],
                self.sensor_size[1],
                self.sensor_size[0],
            )
        )

        to_timesurface_numpy(
            events["x"],
            events["y"],
            events["t"],
            events["p"],
            indices,
            timestamp_memory,
            all_surfaces,
            tau=self.tau,
        )
        return all_surfaces


@numba.jit(nopython=True)
def to_timesurface_numpy(x, y, t, p, indices, timestamp_memory, all_surfaces, tau=5e3):
    """Representation that creates timesurfaces for each event in the recording. Modeled after the
    paper Lagorce et al. 2016, Hots: a hierarchy of event-based time-surfaces for pattern
    recognition https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7508476.
    Parameters:
        surface_dimensions (int, int): width does not have to be equal to height, however both numbers have to be odd.
            if surface_dimensions is None: the time surface is defined globally, on the whole sensor grid.
        tau (float): time constant to decay events around occuring event with.
        decay (str): can be either 'lin' or 'exp', corresponding to linear or exponential decay.
    Returns:
        array of timesurfaces with dimensions (w,h) or (p,w,h)
    """
    current_index_pos = 0
    for index in range(len(x)):
        timestamp_memory[p[index], y[index], x[index]] = t[index]

        if index == indices[current_index_pos]:
            timestamp_context = timestamp_memory - t[index]
            all_surfaces[current_index_pos, :, :, :] = np.exp(timestamp_context / tau)
            current_index_pos += 1
            if current_index_pos > len(indices) - 1:
                break
