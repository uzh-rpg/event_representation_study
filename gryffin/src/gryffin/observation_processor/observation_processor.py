#!/usr/bin/env python

__author__ = "Florian Hase"

import numpy as np

from chimera import Chimera
from gryffin.utilities import Logger, GryffinSettingsError


class ObservationProcessor(Logger):
    def __init__(self, config):
        self.config = config
        self.chimera = Chimera(
            tolerances=self.config.obj_tolerances,
            absolutes=self.config.obj_absolutes,
            goals=self.config.obj_goals,
            softness=self.config.get("softness"),
        )
        Logger.__init__(
            self, "ObservationProcessor", verbosity=self.config.get("verbosity")
        )

        # compute some boundaries
        self.feature_lowers = self.config.feature_lowers
        self.feature_uppers = self.config.feature_uppers
        self.soft_lower = self.feature_lowers + 0.1 * (
            self.feature_uppers - self.feature_lowers
        )
        self.soft_upper = self.feature_uppers - 0.1 * (
            self.feature_uppers - self.feature_lowers
        )

        # attributes of the data
        self.min_obj = None
        self.max_obj = None

    def mirror_parameters(self, param_vector):
        # get indices
        lower_indices_prelim = np.where(param_vector < self.soft_lower)[0]
        upper_indices_prelim = np.where(param_vector > self.soft_upper)[0]

        lower_indices = []
        upper_indices = []
        for feature_index, feature_type in enumerate(self.config.feature_types):
            if feature_type != "continuous":
                continue
            if feature_index in lower_indices_prelim:
                lower_indices.append(feature_index)
            if feature_index in upper_indices_prelim:
                upper_indices.append(feature_index)

        index_dict = {index: "lower" for index in lower_indices}
        for index in upper_indices:
            index_dict[index] = "upper"

        # mirror param
        params = []
        index_dict_keys, index_dict_values = list(index_dict.keys()), list(
            index_dict.values()
        )
        for index in range(2 ** len(index_dict)):
            param_copy = param_vector.copy()
            for jndex in range(len(index_dict)):
                if (index // 2**jndex) % 2 == 1:
                    param_index = index_dict_keys[jndex]
                    if index_dict_values[jndex] == "lower":
                        param_copy[param_index] = self.feature_lowers[param_index] - (
                            param_vector[param_index] - self.feature_lowers[param_index]
                        )
                    elif index_dict_values[jndex] == "upper":
                        param_copy[param_index] = self.feature_uppers[param_index] + (
                            self.feature_uppers[param_index] - param_vector[param_index]
                        )
            params.append(param_copy)
        if len(params) == 0:
            params.append(param_vector.copy())
        return params

    def scalarize_objectives(self, objs, transform="sqrt"):
        """Scalarize and transform objective if needed"""

        # save min/max of the objective so that we can un-normalize the surrogate if it is requested
        self.min_obj = np.amin(objs)
        self.max_obj = np.amax(objs)

        # chimera already normalizes the objective such that it is [0,1]
        scalarized = self.chimera.scalarize(objs)

        if transform is None:
            return scalarized
        elif transform == "sqrt":
            # accentuate global minimum
            return np.sqrt(scalarized)
        elif transform == "cbrt":
            # accentuate global minimum more than sqrt
            return np.cbrt(scalarized)
        elif transform == "square":
            # de-emphasise global minimum
            return np.square(scalarized)
        else:
            raise GryffinSettingsError(
                f'cannot understand transform argument "{transform}"'
            )

    def process_observations(self, obs_dicts):
        obs_params = []  # all params
        raw_objs = []  # all objective values (possibly >1 objective)
        obs_feas = (
            []
        )  # all feasibility values, 0 for feasible, 1 for infeasible (i.e. obj == NaN)
        mask_kwn = []  # mask known/feasible objectives
        mask_mirror = []  # mask for original (non-mirrored params)

        # -------------------------------
        # parse parameters and objectives
        # -------------------------------
        for obs_dict in obs_dicts:
            # get param vector
            param_vector = param_dict_to_vector(
                param_dict=obs_dict,
                param_names=self.config.param_names,
                param_options=self.config.param_options,
                param_types=self.config.param_types,
            )
            mirrored_params = self.mirror_parameters(param_vector)

            # get obj-vector
            obj_vector = np.array(
                [obs_dict[obj_name] for obj_name in self.config.obj_names]
            )

            # --------------------
            # add processed params
            # --------------------
            for i, param in enumerate(mirrored_params):
                obs_params.append(param)
                raw_objs.append(obj_vector)

                # if any of the objectives is NaN ==> unknown/infeasible point
                if any(np.isnan(obj_vector)) is True:
                    feas = 1.0  # i.e. infeasible/unknown
                    kwn = False
                else:
                    feas = 0.0  # i.e. feasible/known
                    kwn = True

                # keep track of mirrored params. The first set of params is the original one, if more are present,
                # they are mirrored ones
                if i == 0:
                    mirror = False
                else:
                    mirror = True

                obs_feas.append(feas)
                mask_kwn.append(kwn)
                mask_mirror.append(mirror)

        # lists to np arrays
        obs_params = np.array(obs_params)
        raw_objs = np.array(raw_objs)
        obs_feas = np.array(obs_feas)
        mask_kwn = np.array(mask_kwn)
        mask_mirror = np.array(mask_mirror)

        # ---------------------------
        # process multiple objectives
        # ---------------------------
        obs_objs = np.empty(shape=len(raw_objs))

        # we scalarize of known objectives, while we assign NaN to any objective vector containing NaN values

        if len(raw_objs[mask_kwn]) > 0:  # guard against empty knw objs
            # Note that Chimera takes care of adjusting the objectives based on whether we are
            #  minimizing vs maximizing
            obs_objs_kwn = self.scalarize_objectives(
                objs=raw_objs[mask_kwn], transform=self.config.get("obj_transform")
            )
            obs_objs[mask_kwn] = obs_objs_kwn

        if len(raw_objs[~mask_kwn]) > 0:  # guard against empty uknw objs
            obs_objs_ukwn = np.array(
                [np.nan] * len(raw_objs[~mask_kwn])
            )  # array of NaNs
            obs_objs[~mask_kwn] = obs_objs_ukwn

        return obs_params, obs_objs, obs_feas, mask_kwn, mask_mirror


# ================
# Helper functions
# ================
def param_dicts_to_vectors(param_dicts, param_names, param_options, param_types):
    """Converts a list of param dictionaries to a two-dimensional array of parameters

    Parameters
    ----------
    param_dicts : list
        list of dicts with a set of input parameters.

    Returns
    -------
    param_vectors : array
        array with the parameter vectors.
    """

    param_vectors = []
    for param_dict in param_dicts:
        param_vector = param_dict_to_vector(
            param_dict, param_names, param_options, param_types
        )
        param_vectors.append(param_vector)
    return np.array(param_vectors)


def param_vectors_to_dicts(param_vectors, param_names, param_options, param_types):
    """Converts list of sample arrays to to list of dictionaries"""
    param_dicts = []
    for param_vector in param_vectors:
        param_dict = param_vector_to_dict(
            param_vector, param_names, param_options, param_types
        )
        param_dicts.append(param_dict)
    return param_dicts


def param_dict_to_vector(param_dict, param_names, param_options, param_types):
    """Parse param dict and put into vector format"""
    param_vector = []
    for param_index, param_name in enumerate(param_names):
        param_type = param_types[param_index]
        if param_type == "continuous":
            param = param_dict[param_name]
        elif param_type == "categorical":
            element = param_dict[param_name]
            param = param_options[param_index].index(element)
        elif param_type == "discrete":
            element = param_dict[param_name]
            param = list(param_options[param_index]).index(element)

        param_vector.append(param)

    return np.array(param_vector)


def param_vector_to_dict(param_vector, param_names, param_options, param_types):
    """parse single sample and return a dict"""
    param_dict = {}
    for param_index, param_name in enumerate(param_names):
        param_type = param_types[param_index]

        if param_type == "continuous":
            param_dict[param_name] = param_vector[param_index]

        elif param_type == "categorical":
            options = param_options[param_index]
            selected_option_idx = int(param_vector[param_index])
            selected_option = options[selected_option_idx]
            param_dict[param_name] = selected_option

        elif param_type == "discrete":
            options = param_options[param_index]
            selected_option_idx = int(param_vector[param_index])
            selected_option = options[selected_option_idx]
            param_dict[param_name] = selected_option
    return param_dict
