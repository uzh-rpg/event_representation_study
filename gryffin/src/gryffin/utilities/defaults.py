#!/usr/bin/env python

import json

__author__ = "Florian Hase, Matteo Aldeghi"

# =============================
# Default general configuration
# =============================
default_general_configuration = {
    "num_cpus": 1,  # Options are a number, or 'all'
    "boosted": True,
    "caching": True,
    "auto_desc_gen": False,
    "batches": 1,
    "sampling_strategies": 2,
    "softness": 0.001,  # softness of Chimera for multiobj optimizations
    # fwa = feasibility-weighted acquisition
    # fia = feasibility-interpolated acquisition
    # fca = feasibility-constrained acquisition
    "feas_approach": "fwa",
    "feas_param": 1,  # sensitivity to feasibility constraints
    "dist_param": 0.5,  # factor modulating density-based penalty in sample selector
    "random_seed": None,  # None for random, or set random seed to a value
    "save_database": False,
    "acquisition_optimizer": "adam",  # options are "adam" or "genetic"
    "obj_transform": "sqrt",  # options are None, "sqrt", "cbrt", "square"
    "num_random_samples": 200,  # num of samples per dimensions to sample when optimizing acquisition
    "reject_tol": 1000,  # tolerance in rejection sampling, relevant when known constraints or fca used
    # verbosity level, from 0 to 5. 0: FATAL, 1: ERROR, 2: WARNING, 3: STATS, 4: INFO, 5: DEBUG
    "verbosity": 4,
}


# ==============================
# Default database configuration
# ==============================
default_database_configuration = {
    "format": "sqlite",
    "path": "./SearchProgress",
    "log_observations": True,
    "log_runtimes": True,
}

# =========================
# Default BNN configuration
# =========================
default_model_configuration = {
    "num_epochs": 2 * 10**3,
    "learning_rate": 0.05,
    "num_draws": 10**3,
    "num_layers": 3,
    "hidden_shape": 6,
    "weight_loc": 0.0,
    "weight_scale": 1.0,
    "bias_loc": 0.0,
    "bias_scale": 1.0,
}

# =============================
# Default overall configuration
# =============================
default_configuration = {
    "general": {
        key: default_general_configuration[key]
        for key in default_general_configuration.keys()
    },
    "database": {
        key: default_database_configuration[key]
        for key in default_database_configuration.keys()
    },
    "model": {
        key: default_model_configuration[key]
        for key in default_model_configuration.keys()
    },
    "parameters": [
        {"name": "param_0", "type": "continuous", "low": 0, "high": 1},
        {"name": "param_1", "type": "continuous", "low": 0, "high": 1}
        # {'name': 'param_1', 'type': 'categorical', 'category_details': {'A':[1,2], 'B'[2,1], ..., 'Z':[4,5]},
    ],
    "objectives": [
        {"name": "obj_0", "goal": "min", "tolerance": 0.2, "absolute": False},
        {"name": "obj_1", "goal": "max", "tolerance": 0.2, "absolute": False},
    ],
}


def get_config_defaults(json_file=None):
    """Returns the default configurations for Gryffin.

    Parameters
    ----------
    json_file: str
        Whether to write the default configurations to a json file with this name. Default is None, i.e.
        do not save json file.

    Returns
    -------
    config : dict
        The default configurations for Gryffin, either as dict or json string.
    """
    if json_file is not None:
        with open(json_file, "w") as f:
            json.dump(default_configuration, f, indent=4)

    return default_configuration
