# author: Nikola Zubic

from representations.event_stack import EventStack
import tonic.transforms as tonic_transforms
from representations.time_surface import ToTimesurface
import numpy as np
import numpy.lib.recfunctions as rfn
from representations.optimized_representation import get_optimized_representation
from representations.tore import events2ToreFeature


def get_item_transform(
    reshaped_return_data, representation_name, transform, height, width, num_events
):
    if "ToVoxelGrid" in representation_name:
        transformation = transform((width, height, 2), n_time_bins=12)
        rep = transformation(reshaped_return_data)
        rep = rep.transpose(0, 2, 3, 1)[..., 0]
        rep = rep.transpose(1, 2, 0) * 255

    elif "MixedDensityEventStack" in representation_name:
        rep = get_optimized_representation(
            reshaped_return_data, num_events, height, width
        )
        rep *= 255

    elif "EventStack" in representation_name:
        reshaped_return_data["p"] = (reshaped_return_data["p"] + 1) // 2
        stack_size = 12
        transformation = EventStack(stack_size, num_events, height, width)
        pre_stack = transformation.pre_stack(
            reshaped_return_data, reshaped_return_data[-1]["t"]
        )
        post_stack = transformation.post_stack(pre_stack)
        rep = post_stack.transpose(0, 1, 3, 2)[..., 0] * 255
        # rep = post_stack.reshape(height, width, stack_size) * 255

    elif "ToImage" in representation_name:
        transformation = transform((width, height, 2))
        reshaped_return_data["p"] = (reshaped_return_data["p"] + 1) // 2
        rep = transformation(reshaped_return_data)
        rep = rep.transpose(1, 2, 0)
        rep *= 255

    elif "TORE" in representation_name.upper():
        k = 6
        x, y, ts, pol = (
            reshaped_return_data["x"],
            reshaped_return_data["y"],
            reshaped_return_data["t"],
            reshaped_return_data["p"],
        )

        # Normalize the data for TORE representation conversion
        x = x - min(x) + 1
        y = y - min(y) + 1
        sampleTimes = ts[-1]
        frameSize = (max(y), max(x))

        rep = events2ToreFeature(x, y, ts, pol, sampleTimes, k, frameSize)
        rep *= 255

    elif "ToTimesurface" in representation_name:
        reshaped_return_data["p"] = ((reshaped_return_data["p"] + 1) / 2).astype(
            np.int8
        )
        transform = ToTimesurface(
            sensor_size=(width, height, 2),
            surface_dimensions=None,
            tau=50000,
            decay="exp",
        )
        t = reshaped_return_data["t"]
        t_norm = (t - t[0]) / (t[-1] - t[0]) * 6
        idx = np.searchsorted(t_norm, np.arange(6) + 1)

        rep = transform(reshaped_return_data, idx)
        rep = rep.reshape((-1, rep.shape[-2], rep.shape[-1]))
        rep = rep.transpose(1, 2, 0)

        rep *= 255

    return rep
