import os
import sys
from pathlib import Path
import torch
from gryffin import Gryffin
import pickle
from chosen_indexes import chosen
import numpy.lib.recfunctions as rfn
import numpy as np
from compute_otmi import otmi
from copy import deepcopy

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
sys.path.append(os.path.join(ROOT, "representations"))
sys.path.append(os.path.join(ROOT, "ev-YOLOv6/"))

from representations.representation_search.mixed_density_event_stack import (
    MixedDensityEventStack,
)
from yolov6.data.gen1_2yolo import Gen1H5
from yolov6.data.data_augment import letterbox

N_CHANNELS = 12


class Gen1H5_Optimization(Gen1H5):
    def __init__(self, args, file, training, transform, num_events, rank, img_size):
        super().__init__(
            args, file, training, transform, num_events, rank=rank, img_size=img_size
        )
        self.training = training

    def get_item_transform(
        self,
        reshaped_return_data,
        height,
        width,
        num_events,
        indexes_functions_aggregations,
    ):
        window_indexes, functions, aggregations = indexes_functions_aggregations
        stack_size = N_CHANNELS
        stacking_type = ["SBN", "SBT"][
            0
        ]  # stacking based on number of events (SBN) or time (SBT)
        window_indexes = window_indexes + [None] * (
            stack_size - len(window_indexes)
        )  # [0, 6], 7 windows
        functions = functions + [None] * (stack_size - len(functions))
        aggregations = aggregations + [None] * (stack_size - len(aggregations))

        indexes_functions_aggregations = window_indexes, functions, aggregations

        transformation = MixedDensityEventStack(
            stack_size,
            num_events,
            height,
            width,
            indexes_functions_aggregations,
            stacking_type,
        )
        rep = transformation.stack(reshaped_return_data)

        # Letterbox
        img, _, _ = self.resize_image(rep)
        if len(img.shape) == 2:
            img = img[..., np.newaxis]

        shape = self.img_size  # final letterboxed shape

        img, _, _ = letterbox(img, shape, auto=False, scaleup=self.augment)

        if len(img.shape) == 2:
            img = img[..., np.newaxis]

        img = np.ascontiguousarray(img)

        return img

    def __getitem__(self, item):
        idx, handle, _ = self.convert_idx_to_rel_idx(item)
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

        reshaped_return_data = torch.cat(
            (return_data.pos, return_data.t, return_data.x), 1
        )

        return reshaped_return_data


def fix_events_training(events):
    events = rfn.unstructured_to_structured(events)
    events.dtype = [("x", "<i4"), ("y", "<i4"), ("t", "<i4"), ("p", "<i4")]

    return events


def measure_otmi(param, dataset, window_indexes_dict):
    window_index = param["window"]
    function = param["function"]
    aggregation = param["aggregation"]

    window_indexes = [next(iter(i)) for i in window_indexes_dict] + [window_index]
    functions = [i[next(iter(i))][0] for i in window_indexes_dict] + [function]
    aggregations = [i[next(iter(i))][1] for i in window_indexes_dict] + [aggregation]

    indexes_functions_aggregations = window_indexes, functions, aggregations

    C_ps = []

    for ch_index in range(len(chosen[:2])):
        events = dataset[ch_index]
        rep = dataset.get_item_transform(
            fix_events_training(events.cpu().numpy()),
            dataset.height,
            dataset.width,
            dataset.num_events,
            indexes_functions_aggregations,
        )

        rep_size = rep.shape[0]
        C_p = otmi(events, rep, dataset.height, dataset.width, rep_size)
        C_ps.append(C_p)

    param["C_p"] = float(np.mean(C_ps))

    return param


def known_constraints_cat(param):
    POSSIBLE_SCENARIOS = {
        "timestamp": ["variance", "mean", "max", "sum"],
        "polarity": ["mean", "variance", "sum"],
        "count": ["mean", "sum"],
        "timestamp_pos": ["variance", "mean", "max", "sum"],
        "timestamp_neg": ["variance", "mean", "max", "sum"],
        "count_pos": ["mean", "sum"],
        "count_neg": ["mean", "sum"],
    }

    function = param["function"]
    aggregation = param["aggregation"]

    if aggregation not in POSSIBLE_SCENARIOS[function]:
        return False
    else:
        return True


def sequential_optimization(dataset, budget, WINDOW_INDEXES_DICT):
    window_options = [
        i for i in range(7)
    ]  # We have 7 windows (N, [0, N/3], [N/3, 2N/3], [2N/3, N], N/2, N/4, N/8)
    function_options = [
        "timestamp",
        "polarity",
        "count",
        "timestamp_pos",
        "timestamp_neg",
        "count_pos",
        "count_neg",
    ]
    aggregation_options = ["mean", "max", "sum", "variance"]

    parameters = []

    # no descriptors
    desc_window = {option: None for option in window_options}
    desc_function = {option: None for option in function_options}
    desc_aggregation = {option: None for option in aggregation_options}

    parameters.append(
        {
            "name": "window",
            "type": "categorical",
            "options": window_options,
            "category_details": desc_window,
        }
    )
    parameters.append(
        {
            "name": "function",
            "type": "categorical",
            "options": function_options,
            "category_details": desc_function,
        }
    )
    parameters.append(
        {
            "name": "aggregation",
            "type": "categorical",
            "options": aggregation_options,
            "category_details": desc_aggregation,
        }
    )

    config = {
        "general": {
            "random_seed": 42,
            "boosted": True,
            "caching": True,
            "num_cpus": 32,
            "auto_desc_gen": False,
            "batches": 1,
            "acquisition_optimizer": "genetic",
        },
        "parameters": parameters,
        "objectives": [{"name": "C_p", "goal": "min"}],
    }

    gryffin = Gryffin(
        config_dict=config, silent=True, known_constraints=known_constraints_cat
    )

    observations = []
    sampling_strategies = np.array([-1, 1])

    for num_iter in range(budget):
        print("-" * 20, "Iteration:", num_iter + 1, "-" * 20)

        idx_sampling = num_iter % len(sampling_strategies)
        # using alternating sampling strategies which in this case corresponds to alternating exploitative/explorative behaviour
        sampling_strategy = sampling_strategies[idx_sampling]

        # ask Gryffin for a new sample
        samples = gryffin.recommend(
            observations=observations, sampling_strategies=[sampling_strategy]
        )
        observation = deepcopy(samples)

        measurement = measure_otmi(samples[0], dataset, WINDOW_INDEXES_DICT)
        observations.extend(observation)

    C_ps = [observations[i]["C_p"] for i in range(len(observations))]
    index_best_element = C_ps.index(min(C_ps))
    best_observation = observations[index_best_element]

    WINDOW_INDEXES_DICT.append(
        {
            best_observation["window"]: [
                best_observation["function"],
                best_observation["aggregation"],
            ]
        }
    )

    return best_observation, WINDOW_INDEXES_DICT


def run_optimization(
    dataset,
    number_of_channels,
    budget,
    SAVE_PATH="/data/nzubic/Projects/event_representation_study/OptimizedRepresentation/budget100_genetic",
):
    WINDOW_INDEXES_DICT = []
    best_observations = []

    for _ in range(number_of_channels):
        best_observation, WINDOW_INDEXES_DICT = sequential_optimization(
            dataset, budget, WINDOW_INDEXES_DICT
        )
        print("Best observation:", best_observation)
        best_observations.append(best_observation)

    with open(os.path.join(SAVE_PATH, "best_observations.pkl"), "wb") as f:
        pickle.dump(best_observations, f)
    with open(os.path.join(SAVE_PATH, "windows_indexes.pkl"), "wb") as f:
        pickle.dump(WINDOW_INDEXES_DICT, f)


if __name__ == "__main__":
    np.float = float  # this is a hack to fix a bug in gryffin with newer numpy versions
    dataset = Gen1H5_Optimization(
        args=None,
        file=Path("/shares/rpg.ifi.uzh/dgehrig/gen1"),
        training=False,
        transform=MixedDensityEventStack,
        num_events=50000,
        rank=None,
        img_size=240,
    )

    # budget represents number of optimization steps for one channel
    run_optimization(dataset, number_of_channels=N_CHANNELS, budget=100)
    exit(0)
