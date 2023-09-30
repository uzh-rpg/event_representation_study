import os
import sys
from pathlib import Path
import torch
from chosen_indexes import extract_indexes
import numpy as np
from compute_otmi import otmi
from copy import deepcopy
import argparse

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
sys.path.append(os.path.join(ROOT, "representations"))
sys.path.append(os.path.join(ROOT, "ev-YOLOv6/"))

from representations.representation_search.mixed_density_event_stack import (
    MixedDensityEventStack,
)
from representations.tore import events2ToreFeature
from representations.event_stack import EventStack
import tonic.transforms as tonic_transforms
from representations.time_surface import ToTimesurface

from yolov6.data.gen1_2yolo import Gen1H5
from representations.gen1_transforms import get_item_transform
from yolov6.data.data_augment import letterbox


class Gen1H5GWD(Gen1H5):
    def __getitem__(self, item):
        idx, handle, name = self.convert_idx_to_rel_idx(item)
        bboxes, event_idx = self._load_bbox(handle["bbox"], idx)
        (ev_xyt, ev_p) = self._load_events(handle["events"], event_idx)

        data = self.to_data(bboxes, ev_xyt, ev_p)
        return_data = data.clone()

        if len(return_data.pos) < 500:
            return_data = data

        return_data.t = return_data.pos[:, -1:].type(torch.int32)
        return_data.pos = return_data.pos[:, :2].type(torch.int16)
        return_data.x = return_data.x.type(torch.int8)

        assert (return_data.bbox[:, :4] >= 0).all(), (idx, return_data.bbox)
        xmin, ymin = return_data.pos.min(0).values
        xmax, ymax = return_data.pos.max(0).values
        assert xmin >= 0 and ymin >= 0 and xmax < self.width and ymax < self.height, idx

        rep = None
        if self.transform is not None:
            reshaped_return_data = torch.cat(
                (return_data.pos, return_data.t, return_data.x), 1
            )

            events = deepcopy(reshaped_return_data)

            if not "To3ChannelPseudoFrameRepresentation" in str(self.transform):
                reshaped_return_data = self.fix_events_training(
                    reshaped_return_data.cpu().numpy()
                )
                rep = get_item_transform(
                    reshaped_return_data,
                    str(self.transform),
                    self.transform,
                    self.height,
                    self.width,
                    self.num_events,
                    self.time_window,
                )

        # Letterbox
        img, _, _ = self.resize_image(rep)
        if len(img.shape) == 2:
            img = img[..., np.newaxis]

        shape = self.img_size  # final letterboxed shape

        img, _, _ = letterbox(img, shape, auto=False, scaleup=self.augment)

        if len(img.shape) == 2:
            img = img[..., np.newaxis]

        # Convert
        img = np.ascontiguousarray(img)

        return (events, img)  # events, rep


def measure_otmi(dataset):
    C_ps = []

    chosen_list = extract_indexes(args.event_representation_name)

    for ch_index in chosen_list:
        # for ch_index in [0, 234]:
        events, rep = dataset[ch_index]
        rep_size = rep.shape[0]

        C_p = otmi(events, rep, dataset.height, dataset.width, rep_size)
        C_ps.append(C_p)

    return C_ps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select an event representation")
    parser.add_argument(
        "--event_representation_name",
        type=str,
        default="VoxelGrid",
        help="Name of the event representation to use",
    )
    args = parser.parse_args()

    event_representations = {
        "VoxelGrid": tonic_transforms.ToVoxelGrid,
        "EventHistogram": tonic_transforms.ToImage,
        "TimeSurface": ToTimesurface,
        "EventStack": EventStack,
        "OptimizedRepresentation": MixedDensityEventStack,
        "TORE": events2ToreFeature,
    }
    event_representation = event_representations[args.event_representation_name]
    print(args.event_representation_name)

    dataset = Gen1H5GWD(
        args=None,
        file=Path("/shares/rpg.ifi.uzh/dgehrig/gen1"),
        training=False,
        task="val",
        transform=event_representation,
        num_events=50000,
        rank=None,
        img_size=240,
    )

    C_ps = measure_otmi(dataset)

    print(f"Mean C_p = {np.mean(C_ps):.4f}")

    exit(0)
