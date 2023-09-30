#!/usr/bin/env python

import pytest
import numpy as np
from gryffin import Gryffin

from gryffin.benchmark_functions import dejong, rosenbrock


PARAM_DIM = 2
NUM_OPTS = 21
BUDGET = 3
SAMPLING_STRATEGIES = np.array([-1, 1])

surface = dejong
surface_moo = rosenbrock


def test_mixed():
    param_2_details = {f"x_{i}": [i] for i in range(NUM_OPTS)}
    config = {
        "general": {
            "num_cpus": 4,
            "auto_desc_gen": True,
            "batches": 1,
            "sampling_strategies": 1,
            "boosted": False,
            "caching": True,
            "random_seed": 2021,
            "acquisition_optimizer": "adam",
            "verbosity": 3,
        },
        "parameters": [
            {"name": "param_0", "type": "continuous", "low": 0.0, "high": 1.0},
            {"name": "param_1", "type": "continuous", "low": 0.0, "high": 1.0},
            {
                "name": "param_2",
                "type": "categorical",
                "category_details": param_2_details,
            },
        ],
        "objectives": [
            {"name": "obj", "goal": "min"},
        ],
    }

    gryffin = Gryffin(config_dict=config)
    observations = []
    for iter_ in range(BUDGET):
        select_ix = iter_ % len(SAMPLING_STRATEGIES)
        sampling_strategy = SAMPLING_STRATEGIES[select_ix]

        samples = gryffin.recommend(
            observations, sampling_strategies=[sampling_strategy]
        )
        sample = samples[0]
        # measure contunous surface with first 2 params
        observation = surface([sample["param_0"], sample["param_1"]])
        # mock categorical parameter measurement
        observation += float(sample["param_2"][2:])
        sample["obj"] = observation
        observations.append(sample)

    assert len(observations) == BUDGET
    assert len(observations[0]) == 4


def test_moo():
    param_2_details = {f"x_{i}": [i] for i in range(NUM_OPTS)}
    config = {
        "general": {
            "num_cpus": 4,
            "auto_desc_gen": True,
            "batches": 1,
            "sampling_strategies": 1,
            "boosted": False,
            "caching": True,
            "random_seed": 2021,
            "acquisition_optimizer": "adam",
            "verbosity": 3,
        },
        "parameters": [
            {"name": "param_0", "type": "continuous", "low": 0.0, "high": 1.0},
            {"name": "param_1", "type": "continuous", "low": 0.0, "high": 1.0},
            {
                "name": "param_2",
                "type": "categorical",
                "category_details": param_2_details,
            },
        ],
        "objectives": [
            {"name": "obj_0", "goal": "min", "tolerance": 0.2, "absolute": False},
            {"name": "obj_1", "goal": "max", "tolerance": 0.1, "absolute": False},
        ],
    }

    gryffin = Gryffin(config_dict=config)
    observations = []
    for iter_ in range(BUDGET):
        select_ix = iter_ % len(SAMPLING_STRATEGIES)
        sampling_strategy = SAMPLING_STRATEGIES[select_ix]

        samples = gryffin.recommend(
            observations, sampling_strategies=[sampling_strategy]
        )
        sample = samples[0]
        # measure contunous surface with first 2 params
        observation = surface([sample["param_0"], sample["param_1"]])
        # mock categorical parameter measurement
        observation += float(sample["param_2"][2:])
        sample["obj_0"] = observation

        # measure the second objective
        observation = surface_moo([sample["param_0"], sample["param_1"]])
        observation += float(sample["param_2"][2:])
        sample["obj_1"] = observation

        observations.append(sample)

    assert len(observations) == BUDGET
    assert len(observations[0]) == 5
