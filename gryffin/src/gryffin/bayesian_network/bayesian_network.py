#!/usr/bin/env python

__author__ = "Florian Hase"

import time
import numpy as np
from . import CategoryReshaper
from gryffin.utilities import Logger, parse_time
from gryffin.utilities import GryffinUnknownSettingsError
from .kernel_evaluations import KernelEvaluator
from .torch_interface import BNNTrainer
from contextlib import nullcontext
import pickle


class BayesianNetwork(Logger):
    def __init__(self, config, frac_feas, model_details=None):
        self.config = config
        self.frac_feas = frac_feas

        # get domain volume
        self.volume = None
        self.inverse_volume = None
        self._get_volume()

        # variables created after sample/build_kernels
        self.obs_objs_kwn = None
        self.obs_objs_feas = None
        self.kernel_regression = None
        self.kernel_classification = None

        # variables for kernel density classification
        self.prior_0 = 1  # default prior is all feasible
        self.prior_1 = 0
        self.log_prior_0 = None
        self.log_prior_1 = None

        # variables for kernel density estimation and regression
        self.trace_kernels = None
        self.cat_reshaper = CategoryReshaper(self.config)

        # get kernel types and sizes
        (
            self.kernel_types,
            self.kernel_sizes,
            self.kernel_ranges,
        ) = self._get_kernel_types_and_sizes(self.config)

        # verbosity settings
        self.verbosity = self.config.get("verbosity")
        Logger.__init__(self, "BayesianNetwork", verbosity=self.verbosity)

        # get bnn model details
        if model_details is None:
            self.model_details = self.config.model_details.to_dict()
        else:
            self.model_details = model_details

        # whether to use boosting, i.e. get lower prob bound
        if self.config.get("boosted"):
            self.lower_prob_bound = 1e-1
            for size in self.config.feature_ranges:
                self.lower_prob_bound *= 1.0 / size
        else:
            self.lower_prob_bound = 1e-25

        # whether to use kernel caching
        self.caching = self.config.get("caching")
        self.cache = None
        if self.caching is True:
            # check if we can actually construct cache and update option
            # caching is done only if all kernel types are categorical (i.e. all type 2)
            if np.sum(self.kernel_types > 1.5) == len(self.kernel_types):
                self.cache = {}
            else:
                self.caching = False

    def _get_volume(self):
        # get domain volume
        self.volume = 1.0
        feature_lengths = self.config.feature_lengths
        feature_ranges = self.config.feature_ranges

        for feature_index, feature_type in enumerate(self.config.feature_types):
            if feature_type == "continuous":
                self.volume *= feature_ranges[feature_index]
            elif feature_type == "categorical":
                self.volume *= feature_lengths[feature_index]
            elif feature_type == "discrete":
                self.volume *= feature_ranges[feature_index]
            else:
                GryffinUnknownSettingsError(
                    'did not understand parameter type: "%s" of variable "%s".\n\t(%s) Please choose from "continuous" or "categorical"'
                    % (
                        feature_type,
                        self.config.feature_names[feature_index],
                        self.template,
                    )
                )
        self.inverse_volume = 1 / self.volume

    def sample(self, obs_params):
        start = time.time()

        if self.verbosity > 3.5:  # i.e. at INFO level
            cm = self.console.status("Training the Bayesian neural network...")
        else:
            cm = nullcontext()

        with cm:
            trainer = BNNTrainer(self.config, self.model_details, self.frac_feas)
            model = trainer.train(obs_params)

        self.trace_kernels = model.get_kernels()

        end = time.time()
        time_string = parse_time(start, end)
        self.log(f"Bayesian neural network trained in {time_string}", "STATS")

    def build_kernels(
        self, descriptors_kwn, descriptors_feas, obs_objs, obs_feas, mask_kwn
    ):
        assert self.trace_kernels is not None  # check we built the kernels

        self.obs_objs_kwn = obs_objs[mask_kwn]
        self.obs_objs_feas = obs_feas

        # get prior probabilities (feasible = 0 and infeasible = 1)
        # prior_0 is essentially the fraction of feasible points
        # prior_1 is essentially the fraction of infeasible points
        self.prior_0 = sum([xi < 0.5 for xi in self.obs_objs_feas]) / len(
            self.obs_objs_feas
        )
        self.prior_1 = sum([xi > 0.5 for xi in self.obs_objs_feas]) / len(
            self.obs_objs_feas
        )
        assert np.abs((self.prior_0 + self.prior_1) - 1.0) < 10e-5

        self.log_prior_0 = np.log(self.prior_0) if self.prior_0 > 0.0 else -np.inf
        self.log_prior_1 = np.log(self.prior_1) if self.prior_1 > 0.0 else -np.inf
        # retrieve kernels densities
        # all kernels - shape of the tensors below: (# samples, # obs, # kernels)
        locs_all = self.trace_kernels["locs"]
        sqrt_precs_all = self.trace_kernels["sqrt_precs"]
        probs_all = self.trace_kernels["probs"]

        # kernels only for known/feasible objectives/params
        locs_kwn = locs_all[:, mask_kwn, :]
        sqrt_precs_kwn = sqrt_precs_all[:, mask_kwn, :]
        probs_kwn = probs_all[:, mask_kwn, :]

        assert locs_kwn.shape[1] == len(self.obs_objs_kwn)
        assert locs_all.shape[1] == len(self.obs_objs_feas)

        # reshape categorical probabilities
        start = time.time()
        probs_kwn = self._reshape_categorical_probabilities(probs_kwn, descriptors_kwn)
        probs_all = self._reshape_categorical_probabilities(probs_all, descriptors_feas)
        end = time.time()

        # report cat reshaping time only if we have categorical variables
        if "categorical" in self.config.kernel_types:
            self.log(
                "[TIME]:  "
                + parse_time(start, end)
                + "  (reshaping categorical space)",
                "DEBUG",
            )

        # kernels used for regression
        self.kernel_regression = KernelEvaluator(
            locs=locs_kwn,
            sqrt_precs=sqrt_precs_kwn,
            cat_probs=probs_kwn,
            kernel_types=self.kernel_types,
            kernel_sizes=self.kernel_sizes,
            kernel_ranges=self.kernel_ranges,
            lower_prob_bound=self.lower_prob_bound,
            objs=self.obs_objs_kwn,
            inv_vol=self.inverse_volume,
        )

        # kernels used for feasibility classification
        self.kernel_classification = KernelEvaluator(
            locs=locs_all,
            sqrt_precs=sqrt_precs_all,
            cat_probs=probs_all,
            kernel_types=self.kernel_types,
            kernel_sizes=self.kernel_sizes,
            kernel_ranges=self.kernel_ranges,
            lower_prob_bound=self.lower_prob_bound,
            objs=self.obs_objs_feas,
            inv_vol=self.inverse_volume,
        )

    def kernel_contribution(self, proposed_sample):
        """
        Computes acquisition terms that rely on kernel densities.
        """
        if self.caching is True:
            sample_string = "-".join([str(int(element)) for element in proposed_sample])
            if sample_string in self.cache:
                num, inv_den = self.cache[sample_string]
            else:
                num, inv_den, _ = self.kernel_regression.get_kernel_contrib(
                    proposed_sample.astype(np.float64)
                )
                self.cache[sample_string] = (num, inv_den)
        else:
            num, inv_den, _ = self.kernel_regression.get_kernel_contrib(
                proposed_sample.astype(np.float64)
            )
        return num, inv_den

    def regression_surrogate(self, proposed_sample):
        y_pred = self.kernel_regression.get_regression_surrogate(
            proposed_sample.astype(np.float64)
        )
        return y_pred

    def classification_surrogate(self, proposed_sample, threshold=0.5):
        prob = self.kernel_classification.get_probability_of_feasibility(
            proposed_sample.astype(np.float64), self.log_prior_0, self.log_prior_1
        )
        if prob > threshold:
            return True
        else:
            return False

    def prob_feasible(self, proposed_sample):
        """
        Computes probability of feasibility based on kernel density classification.
        """
        return self.kernel_classification.get_probability_of_feasibility(
            proposed_sample.astype(np.float64), self.log_prior_0, self.log_prior_1
        )

    def prob_infeasible(self, proposed_sample):
        """
        Computes probability of infeasibility based on kernel density classification.
        """
        return self.kernel_classification.get_probability_of_infeasibility(
            proposed_sample.astype(np.float64), self.log_prior_0, self.log_prior_1
        )

    def infeasible_kernel_density(self, proposed_sample):
        """
        Computes kernel density estimate of infeasible points.
        """
        _, log_density_1 = self.kernel_classification.get_binary_kernel_densities(
            proposed_sample.astype(np.float64)
        )
        return np.exp(log_density_1)

    def _reshape_categorical_probabilities(self, probs, descriptors):
        try:
            if np.all(np.isfinite(np.array(descriptors, dtype=np.float))):
                probs = self.cat_reshaper.reshape(probs, descriptors)
        except ValueError:
            probs = self.cat_reshaper.reshape(probs, descriptors)
        return probs

    @staticmethod
    def _get_kernel_types_and_sizes(config):
        """Convert kernel type string to integers as follows:
        0 : continuous, non-periodic
        1 : continuous, periodic
        2 : categorical or discrete (which is treated as categorical with ordinal descriptors)
        """
        kernel_type_strings = config.kernel_types
        feature_periodic = config.kernel_periodic
        assert len(kernel_type_strings) == len(feature_periodic)

        kernel_types = []
        for kernel_type_string, periodic in zip(kernel_type_strings, feature_periodic):
            if kernel_type_string == "continuous":
                if periodic:
                    kernel_types.append(1)
                else:
                    kernel_types.append(0)
            elif kernel_type_string == "categorical":
                kernel_types.append(2)
            elif kernel_type_string == "discrete":
                kernel_types.append(2)

        kernel_types = np.array(kernel_types, dtype=np.int32)
        kernel_sizes = config.kernel_sizes.astype(np.int32)
        kernel_ranges = config.kernel_ranges.astype(np.float64)
        return kernel_types, kernel_sizes, kernel_ranges
