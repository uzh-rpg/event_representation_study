#!/usr/bin/env

__author__ = "Florian Hase"

# ========================================================================

import time
import numpy as np
import multiprocessing
from multiprocessing import Process, Manager

from gryffin.utilities import Logger

from .kernel_prob_reshaping import KernelReshaper

# ========================================================================


class CategoryReshaper(Logger):
    def __init__(self, config):
        self.config = config
        Logger.__init__(
            self, "CategoryReshaper", verbosity=self.config.get("verbosity")
        )

        self.kernel_reshaper = KernelReshaper()

        if self.config.get("num_cpus") == "all":
            self.num_cpus = multiprocessing.cpu_count()
        else:
            self.num_cpus = int(self.config.get("num_cpus"))

    def cython_recompute_probs(self, cat_probs, descriptors, index, return_dict=None):
        recomputed_probs = self.kernel_reshaper.reshape_probs(cat_probs, descriptors)
        if return_dict.__class__.__name__ == "DictProxy":
            return_dict[index] = recomputed_probs
        else:
            return recomputed_probs

    def python_recompute_probs(self, cat_probs, descriptors, index, return_dict=None):
        """
        shape of cat_probs    ... (# samples, # obs, # kernels)
        shape of descriptors: ... (# kernels, # descriptors)
        """
        num_samples = cat_probs.shape[0]
        num_obs = cat_probs.shape[1]
        num_kernels = cat_probs.shape[2]
        num_descriptors = descriptors.shape[1]

        start = time.time()

        recomputed_probs = np.empty(cat_probs.shape)

        # put on separate thread
        for sample_index in range(num_samples):
            # put on separate thread
            for obs_index in range(num_obs):
                start_0 = time.time()

                probs = cat_probs[sample_index, obs_index]

                # compute distances to all categories
                distances = np.empty(num_kernels)
                for target_cat_index in range(num_kernels):
                    ds2 = 0.0

                    for desc_index in range(num_descriptors):
                        averaged_descriptor = 0.0
                        for kernel_index in range(num_kernels):
                            averaged_descriptor += (
                                probs[kernel_index]
                                * descriptors[kernel_index, desc_index]
                            )

                        dyi = 0.0
                        for kernel_index in range(num_kernels):
                            partial_contrib = (
                                descriptors[target_cat_index, desc_index]
                                - averaged_descriptor
                            )
                            dyi += partial_contrib

                        ds2 += dyi * dyi

                    distances[target_cat_index] = np.sqrt(ds2 / num_descriptors)

                # got the distances
                rescaled_probs = np.exp(-distances) / np.sum(np.exp(-distances))
                recomputed_probs[sample_index, obs_index] = rescaled_probs

                end_0 = time.time()

        end = time.time()
        total_time = end - start

        if return_dict.__class__.__name__ == "DictProxy":
            return_dict[index] = recomputed_probs
        else:
            return recomputed_probs

    def reshape(self, raw_probs, descriptors):
        # ... raw_probs = (# samples, # obs, # kernels)

        # assign kernels to individual parameters
        feature_probs = []
        kernel_sizes = self.config.kernel_sizes
        kernel_start = 0
        while kernel_start < raw_probs.shape[2]:
            kernel_end = kernel_start + kernel_sizes[kernel_start]
            feature_probs.append(raw_probs[:, :, kernel_start:kernel_end])
            kernel_start = kernel_end

        # extract all relevant information for all parameters
        parsed_probs = []
        for feature_index, feature_options in enumerate(self.config.feature_options):
            feature_descriptor = descriptors[feature_index]
            parsed_prob = {
                "options": feature_options,
                "descriptors": feature_descriptor,
                "probs": feature_probs[feature_index],
                "feature_index": feature_index,
            }
            parsed_probs.append(parsed_prob)

        # run reshaping algorithm
        if self.num_cpus > 1:
            result_dict = Manager().dict()
            processes = []
            for feature_index in range(len(parsed_probs)):
                prob_dict = parsed_probs[feature_index]
                if prob_dict["descriptors"] is None:
                    result_dict[feature_index] = prob_dict["probs"]
                    continue
                process = Process(
                    target=self.cython_recompute_probs,
                    args=(
                        prob_dict["probs"],
                        prob_dict["descriptors"],
                        feature_index,
                        result_dict,
                    ),
                )
                # 				process   = Process(target = self.python_recompute_probs, args = (prob_dict['probs'], prob_dict['desriptors'], feature_index, result_dict))
                processes.append(process)
                process.start()
            for process in processes:
                process.join()
        else:
            result_dict = {}
            for feature_index in range(len(parsed_probs)):
                prob_dict = parsed_probs[feature_index]
                if prob_dict["descriptors"] is None:
                    result_dict[feature_index] = prob_dict["probs"]
                    continue
                result_dict[feature_index] = self.cython_recompute_probs(
                    prob_dict["probs"], prob_dict["descriptors"], feature_index
                )
        # 				result_dict[feature_index] = self.python_recompute_probs(prob_dict['probs'], prob_dict['descriptors'], feature_index)

        recomputed_probs = []
        for feature_index in range(len(parsed_probs)):
            if feature_index in result_dict:
                recomputed_probs.append(result_dict[feature_index])
            else:
                recomputed_probs.append(
                    np.ones((raw_probs.shape[0], raw_probs.shape[1], 1))
                )

        # assemble reshaped probs
        reshaped_probs = np.concatenate(recomputed_probs, axis=2)
        return reshaped_probs
