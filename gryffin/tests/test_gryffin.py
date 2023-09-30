#!/usr/bin/env python
from gryffin import Gryffin
import shutil
import numpy as np


def test_recommend():
    config = {
        "general": {
            "save_database": False,
            "num_cpus": 1,
            "boosted": True,
            "sampling_strategies": 2,
            "random_seed": 42,
        },
        "parameters": [
            {"name": "param_0", "type": "continuous", "low": 0, "high": 1},
            {"name": "param_1", "type": "continuous", "low": 0, "high": 1},
        ],
        "objectives": [{"name": "obj", "goal": "min"}],
    }

    observations = [
        {"param_0": 0.3, "param_1": 0.4, "obj": 0.1},
        {"param_0": 0.5, "param_1": 0.6, "obj": 0.2},
    ]

    gryffin = Gryffin(config_dict=config)
    _ = gryffin.recommend(observations=observations)


def test_multiobjective():
    config = {
        "general": {
            "save_database": False,
            "num_cpus": 1,
            "boosted": True,
            "sampling_strategies": 2,
            "random_seed": 42,
        },
        "parameters": [
            {"name": "param_0", "type": "continuous", "low": 0, "high": 1},
            {"name": "param_1", "type": "continuous", "low": 0, "high": 1},
        ],
        "objectives": [
            {"name": "obj0", "goal": "min", "tolerance": 0.2, "absolute": False},
            {"name": "obj1", "goal": "max", "tolerance": 0.1, "absolute": False},
        ],
    }

    observations = [
        {"param_0": 0.3, "param_1": 0.4, "obj0": 0.1, "obj1": 0.2},
        {"param_0": 0.5, "param_1": 0.6, "obj0": 0.2, "obj1": 0.1},
    ]

    gryffin = Gryffin(config_dict=config)
    _ = gryffin.recommend(observations=observations)
