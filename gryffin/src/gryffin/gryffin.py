#!/usr/bin/env python

__author__ = "Florian Hase, Matteo Aldeghi"

from .acquisition import Acquisition
from .bayesian_network import BayesianNetwork
from .descriptor_generator import DescriptorGenerator
from .observation_processor import (
    ObservationProcessor,
    param_vectors_to_dicts,
    param_dicts_to_vectors,
)
from .random_sampler import RandomSampler
from .sample_selector import SampleSelector
from .utilities import ConfigParser, Logger, GryffinNotFoundError
from .utilities import (
    parse_time,
    memory_usage,
    estimate_feas_fraction,
    compute_constrained_cartesian,
)

import os
import numpy as np
import pandas as pd
import time
from contextlib import nullcontext
from typing import Callable, Union, List, Dict
import ast
from functools import wraps


class Gryffin(Logger):
    def __init__(
        self,
        config_file: str = None,
        config_dict: Dict = None,
        known_constraints: Callable[[Dict], bool] = None,
        frac_feas=None,
        silent: bool = False,
    ) -> None:
        """Initialize Gryffin from a config dict or file.

        A config file or dict must be provided. If both a config file and a config dict
        are provided, the file will be ignored.

        :param config_file: Gryffin config filepath
        :param config_dict: Gryffin config dict
        :param known_constraints: Function imposing contraints on the Gryffin search space.
        :param frac_feas: Feasability fraction TODO: Not sure what this is for
        :param silent: Suppress all standard output. If True, the ``verbosity`` settings
            in ``config`` will be overwritten. Default is False.
        """

        # parse configuration
        self.config = ConfigParser(config_file, config_dict)
        self.config.parse()
        self.config.set_home(os.path.dirname(os.path.abspath(__file__)))

        # set verbosity
        if silent is True:
            self.verbosity = 2
            self.config.general.verbosity = 2
        else:
            self.verbosity = self.config.get("verbosity")

        Logger.__init__(self, "Gryffin", verbosity=self.verbosity)

        # parse constraints function
        self.known_constraints = known_constraints
        if self.known_constraints:
            if frac_feas is not None:
                # override the feasible fraction
                self.frac_feas = frac_feas
            else:
                # if we have known constraints, estimate the feasible fraction
                self.frac_feas = estimate_feas_fraction(
                    self.known_constraints, self.config
                )
        else:
            # no known constriants, assume full domain is feasibile
            self.frac_feas = 1.0

        # if param space is fully categorical, maintain list of all options
        if np.all(
            [p["type"] in ["categorical", "discrete"] for p in self.config.parameters]
        ):
            self.all_options = compute_constrained_cartesian(
                self.known_constraints, self.config
            )
        else:
            self.all_options = None

        # store timings for possible analysis
        self.timings = {}

        np.random.seed(self.config.get("random_seed"))
        self._create_folders()  # folders created only if we are saving to database

        # Instantiate all objects needed
        self.random_sampler = RandomSampler(
            self.config, constraints=self.known_constraints
        )
        self.obs_processor = ObservationProcessor(self.config)
        self.descriptor_generator = DescriptorGenerator(self.config)
        self.descriptor_generator_feas = DescriptorGenerator(self.config)
        self.bayesian_network = BayesianNetwork(
            config=self.config, frac_feas=self.frac_feas
        )
        self.acquisition = Acquisition(
            self.config, known_constraints=self.known_constraints
        )
        self.sample_selector = SampleSelector(self.config, self.all_options)

        self.iter_counter = 0
        self.sampling_param_values = None
        self.sampling_strategies = None
        self.num_batches = None
        # attributes used mainly for investigation/debugging
        self.parsed_input_data = {}
        self.proposals = None

    def _create_folders(self):
        if self.config.get("save_database") is True and not os.path.isdir(
            self.config.get_db("path")
        ):
            try:
                os.mkdir(self.config.get_db("path"))
            except FileNotFoundError:
                GryffinNotFoundError(
                    "Could not create database directory: %s"
                    % self.config.get_db("path")
                )

        if self.config.get("save_database") is True:
            from .database_handler import DatabaseHandler

            self.db_handler = DatabaseHandler(self.config)

    def build_surrogate(self, observations: List = None) -> None:
        """Builds surrogate models of Gryffin without proposing any new experiments.

        :param observations: List of dictionaries with the previous observations
        """
        self.log("", "INFO")
        self.log_chapter("Gryffin", line="=", style="bold #d9ed92")

        if observations is None or len(observations) == 0:
            return None

        self.log(f"{len(observations)} observations found", "STATS")
        # obs_params == all observed parameters
        # obs_objs == all observed objective function evaluations (including NaNs)
        # obs_feas == whether observed parameters are feasible (0) or infeasible (1)
        # mask_kwn == mask that selects only known/feasible params/objs (including mirrored params)
        # mask_mirror == mask that selects the parameters that have been mirrored across opt bounds
        (
            obs_params,
            obs_objs,
            obs_feas,
            mask_kwn,
            mask_mirror,
        ) = self.obs_processor.process_observations(observations)

        # keep for inspection/debugging
        self.parsed_input_data["obs_params"] = obs_params
        self.parsed_input_data["obs_objs"] = obs_objs
        self.parsed_input_data["obs_feas"] = obs_feas
        self.parsed_input_data["mask_kwn"] = mask_kwn
        self.parsed_input_data["mask_mirror"] = mask_mirror

        # -----------------------------
        # Build categorical descriptors
        # -----------------------------
        # can generate descriptors if we have:
        # (i) at least 3 feasible observations (normal desc generation)
        # (ii) at least 2 feasible and 1 infeasible observation (desc generation for feasibility)
        can_generate_desc = len(obs_params[mask_kwn]) > 3 or (
            len(obs_params) > 3 and np.sum(obs_feas) > 0.1
        )
        if self.config.get("auto_desc_gen") is True and can_generate_desc is True:
            self.log_chapter("Descriptor Refinement")
            start = time.time()
            # use status context manager only at INFO verbosity level
            if self.verbosity > 3.5:
                cm = self.console.status("Refining categories descriptors...")
            else:
                cm = nullcontext()
            with cm:
                # only feasible points with known objectives
                if len(obs_params[mask_kwn]) > 3:
                    self.descriptor_generator.generate_descriptors(
                        obs_params[mask_kwn], obs_objs[mask_kwn]
                    )
                # for feasibility descriptors, we use all data, but we run descriptor generation
                # only if we have at least 1 infeasible point, otherwise they are all feasible and there is no point
                # running this. Remember that feasible = 0 and infeasible = 1.
                if len(obs_params) > 3 and np.sum(obs_feas) > 0.1:
                    self.descriptor_generator_feas.generate_descriptors(
                        obs_params, obs_feas
                    )

            end = time.time()
            time_string = parse_time(start, end)
            self.log(
                f"Categorical descriptors refined by [italic]Dynamic Gryffin[/italic] in {time_string}",
                "STATS",
            )

        # extract descriptors and build kernels
        descriptors_kwn = self.descriptor_generator.get_descriptors()
        descriptors_feas = self.descriptor_generator_feas.get_descriptors()

        # ----------------------------------------------
        # sample bnn to get kernels for all observations
        # ----------------------------------------------
        self.log_chapter("Bayesian Network")
        self.bayesian_network.sample(obs_params)  # infer kernel densities
        # build kernel smoothing/classification surrogates
        self.bayesian_network.build_kernels(
            descriptors_kwn=descriptors_kwn,
            descriptors_feas=descriptors_feas,
            obs_objs=obs_objs,
            obs_feas=obs_feas,
            mask_kwn=mask_kwn,
        )

        # -----------
        # Print info
        # -----------
        self.log_chapter("Summary")
        GB, MB, kB = memory_usage()
        self.log(f"Memory usage: {GB:.0f} GB, {MB:.0f} MB, {kB:.0f} kB", "STATS")
        self.log_chapter("End", line="=", style="bold #d9ed92")
        self.log("", "INFO")

    @Acquisition.gryffin_func
    def recommend(
        self,
        observations: List = None,
        sampling_strategies: List = None,
        num_batches: int = None,
        as_array: bool = False,
    ) -> List:
        """Recommends the next set(s) of parameters based on the provided observations.

        :param observations: List of dictionaries with the previous observations.
        :param sampling_strategies: List of the chosen sampling strategies. When providing
            this argument, the config setting ``strategies`` will be ignored.
        :param num_batches: Number of parameter batches requested. When providing this argument,
            the config setting ``batches`` will be ignored.
        :param as_array: Whether to return suggested samples as numpy arrays instead of a list
            of dictionaries. Default is False.

        :return params: List of dictionaries with the suggested parameters.
        """
        self.log("", "INFO")
        self.log_chapter("Gryffin", line="=", style="bold #d9ed92")

        start_time = time.time()
        if sampling_strategies is None:
            num_sampling_strategies = self.config.get("sampling_strategies")
            # positive lambda is exploitation, negative is exploration
            # we start with exploitation and follow with exploration
            # in sample selector, we first choose exploitation, then exploration (with distance penalty for exploration
            # points close to already selected exploitation ones)
            sampling_strategies = np.linspace(1, -1, num_sampling_strategies)
        else:
            sampling_strategies = np.array(sampling_strategies)
            num_sampling_strategies = len(sampling_strategies)

        # register last sampling strategies
        self.sampling_strategies = sampling_strategies
        if num_batches is None:
            self.num_batches = self.config.get("batches")
        else:
            self.num_batches = num_batches

        # print summary of what will be proposed
        num_recommended_samples = self.num_batches * num_sampling_strategies
        samples_str = "samples" if num_recommended_samples > 1 else "sample"
        batches_str = "batches" if self.num_batches > 1 else "batch"
        strategy_str = "strategies" if num_sampling_strategies > 1 else "strategy"
        self.log(
            f"Gryffin will propose {num_recommended_samples} {samples_str}: {self.num_batches} {batches_str} with"
            f" {num_sampling_strategies} sampling {strategy_str}",
            "INFO",
        )

        # -----------------------------------------------------
        # no observations, need to fall back to random sampling
        # -----------------------------------------------------
        if observations is None or len(observations) == 0:
            samples = self.random_sampler.draw(num=num_recommended_samples)
            if self.config.process_constrained:
                dominant_features = self.config.feature_process_constrained
                samples[:, dominant_features] = samples[0, dominant_features]

            # if fully categorical, remove random samples from list of available options
            if np.all([p["type"] == "categorical" for p in self.config.parameters]):
                for sample in samples:
                    sample_ix = np.where(np.all(self.all_options == sample, axis=1))[0]
                    self.all_options = np.delete(self.all_options, sample_ix, axis=0)
                # update sample selector attribute
                setattr(self.sample_selector, "all_options", self.all_options)

        # --------------------
        # we have observations
        # --------------------
        else:
            self.log(f"{len(observations)} observations found", "STATS")
            # obs_params == all observed parameters
            # obs_objs == all observed objective function evaluations (including NaNs)
            # obs_feas == whether observed parameters are feasible (0) or infeasible (1)
            # mask_kwn == mask that selects only known/feasible params/objs (including mirrored params)
            # mask_mirror == mask that selects the parameters that have been mirrored across opt bounds
            (
                obs_params,
                obs_objs,
                obs_feas,
                mask_kwn,
                mask_mirror,
            ) = self.obs_processor.process_observations(observations)

            # keep for inspection/debugging
            self.parsed_input_data["obs_params"] = obs_params
            self.parsed_input_data["obs_objs"] = obs_objs
            self.parsed_input_data["obs_feas"] = obs_feas
            self.parsed_input_data["mask_kwn"] = mask_kwn
            self.parsed_input_data["mask_mirror"] = mask_mirror

            # -----------------------------
            # Build categorical descriptors
            # -----------------------------
            # can generate descriptors if we have:
            # (i) at least 4 feasible observations (normal desc generation)
            # (ii) at least 4 feasible and 1 infeasible observation (desc generation for feasibility)
            can_generate_desc = len(obs_params[mask_kwn]) > 3 or (
                len(obs_params) > 3 and np.sum(obs_feas) > 0.1
            )
            if self.config.get("auto_desc_gen") is True and can_generate_desc is True:
                self.log_chapter("Descriptor Refinement")
                start = time.time()
                # with self.console.status("Refining categories descriptors..."):
                # import pdb; pdb.set_trace()
                # only feasible points with known objectives
                if len(obs_params[mask_kwn]) > 3:
                    self.descriptor_generator.generate_descriptors(
                        obs_params[mask_kwn], obs_objs[mask_kwn]
                    )
                # for feasibility descriptors, we use all data, but we run descriptor generation
                # only if we have at least 1 infeasible point, otherwise they are all feasible and there is no point
                # running this. Remember that feasible = 0 and infeasible = 1.
                if len(obs_params) > 3 and np.sum(obs_feas) > 0.1:
                    self.descriptor_generator_feas.generate_descriptors(
                        obs_params, obs_feas
                    )

                end = time.time()
                time_string = parse_time(start, end)
                self.log(
                    f"Categorical descriptors refined by [italic]Dynamic Gryffin[/italic] in {time_string}",
                    "STATS",
                )

            # extract descriptors and build kernels
            descriptors_kwn = self.descriptor_generator.get_descriptors()
            descriptors_feas = self.descriptor_generator_feas.get_descriptors()

            # ----------------------------------------------
            # get lambda values for exploration/exploitation
            # ----------------------------------------------
            self.sampling_param_values = (
                sampling_strategies * self.bayesian_network.inverse_volume
            )
            dominant_strategy_index = self.iter_counter % len(
                self.sampling_param_values
            )
            dominant_strategy_value = np.array(
                [self.sampling_param_values[dominant_strategy_index]]
            )

            # ----------------------------------------------
            # sample bnn to get kernels for all observations
            # ----------------------------------------------
            self.log_chapter("Bayesian Network")
            self.bayesian_network.sample(obs_params)  # infer kernel densities
            # build kernel smoothing/classification surrogates
            self.bayesian_network.build_kernels(
                descriptors_kwn=descriptors_kwn,
                descriptors_feas=descriptors_feas,
                obs_objs=obs_objs,
                obs_feas=obs_feas,
                mask_kwn=mask_kwn,
            )

            # get incumbent
            if len(obs_params[mask_kwn]) > 0:
                # if we have kwn samples ==> pick params with best merit
                best_params = obs_params[mask_kwn][np.argmin(obs_objs[mask_kwn])]
            else:
                # if we have do not have any feasible sample ==> pick any feasible param at random
                best_params_idx = np.random.randint(low=0, high=len(obs_params))
                best_params = obs_params[best_params_idx]

            # ----------------------------------------------
            # optimize acquisition and select samples
            # ----------------------------------------------
            num_samples_per_dim = self.config.get("num_random_samples")
            # if there are process constraining parameters, run those first
            if self.config.process_constrained:
                self.proposals = self.acquisition.propose(
                    best_params=best_params,
                    bayesian_network=self.bayesian_network,
                    sampling_param_values=self.sampling_param_values,
                    num_samples_per_dim=num_samples_per_dim,
                    dominant_samples=None,
                    timings_dict=None,
                )
                constraining_samples = self.sample_selector.select(
                    num_batches=self.num_batches,
                    proposals=self.proposals,
                    eval_acquisition=self.acquisition.eval_acquisition,
                    sampling_param_values=dominant_strategy_value,
                    obs_params=obs_params,
                )
            else:
                constraining_samples = None

            # then select the remaining proposals
            # note num_samples get multiplied by the number of input variables
            self.log_chapter("Acquisition")
            self.proposals = self.acquisition.propose(
                best_params=best_params,
                bayesian_network=self.bayesian_network,
                sampling_param_values=self.sampling_param_values,
                num_samples_per_dim=num_samples_per_dim,
                dominant_samples=constraining_samples,
                timings_dict=self.timings,
            )

            self.log_chapter("Sample Selector")
            # note: provide `obs_params` as it contains the params for _all_ samples, including the unfeasible ones
            samples = self.sample_selector.select(
                num_batches=self.num_batches,
                proposals=self.proposals,
                eval_acquisition=self.acquisition.eval_acquisition,
                sampling_param_values=self.sampling_param_values,
                obs_params=obs_params,
            )

        # --------------------------------
        # Print overall info for recommend
        # --------------------------------
        self.log_chapter("Summary")
        GB, MB, kB = memory_usage()
        self.log(f"Memory usage: {GB:.0f} GB, {MB:.0f} MB, {kB:.0f} kB", "STATS")
        end_time = time.time()
        time_string = parse_time(start_time, end_time)
        self.log(f"Overall time required: {time_string}", "STATS")
        self.log_chapter("End", line="=", style="bold #d9ed92")
        self.log("", "INFO")

        # -----------------------
        # Return proposed samples
        # -----------------------
        if as_array:
            # return as is
            return_samples = samples
        else:
            # return as dictionary
            return_samples = param_vectors_to_dicts(
                param_vectors=samples,
                param_names=self.config.param_names,
                param_options=self.config.param_options,
                param_types=self.config.param_types,
            )

        if self.config.get("save_database") is True:
            db_entry = {
                "start_time": start_time,
                "end_time": end_time,
                "received_obs": observations,
                "suggested_params": return_samples,
            }
            if self.config.get("auto_desc_gen") is True:
                # save summary of learned descriptors
                descriptor_summary = self.descriptor_generator.get_summary()
                db_entry["descriptor_summary"] = descriptor_summary
            self.db_handler.save(db_entry)

        self.iter_counter += 1
        return return_samples

    def read_db(self, outfile="database.csv", verbose=True):
        self.db_handler.read_db(outfile, verbose)

    @staticmethod
    def _df_to_list_of_dicts(df):
        list_of_dicts = []
        for index, row in df.iterrows():
            d = {}
            for col in df.columns:
                d[col] = row[col]
            list_of_dicts.append(d)
        return list_of_dicts

    def get_regression_surrogate(self, params: Union[List, pd.DataFrame]):
        """Retrieve the surrogate model.

        :param params: List of dicts with input parameters to evaluate. Alternatively it
            can also be a pandas DataFrame where each column name corresponds to one of
            the input parameters in Gryffin.

        :return y_pred: Surrigate model evaluated at the locations defined in params.
        """
        if isinstance(params, pd.DataFrame):
            params = self._df_to_list_of_dicts(params)

        X = param_dicts_to_vectors(
            params,
            param_names=self.config.param_names,
            param_options=self.config.param_options,
            param_types=self.config.param_types,
        )
        y_preds = []
        for x in X:
            y_pred = self.bayesian_network.regression_surrogate(x.astype(np.float64))
            y_preds.append(y_pred)

        # invert transform the surrogate according to the chosen transform
        y_preds = np.array(y_preds)
        transform = self.config.get("obj_transform")
        if transform is None:
            pass
        elif transform == "sqrt":
            # accentuate global minimum
            y_preds = np.square(y_preds)
        elif transform == "cbrt":
            # accentuate global minimum more than sqrt
            y_preds = np.power(y_preds, 3)
        elif transform == "square":
            # de-emphasise global minimum
            y_preds = np.sqrt(y_preds)

        # scale the predicted objective back to the original range
        if self.obs_processor.min_obj != self.obs_processor.max_obj:
            y_preds = (
                y_preds * (self.obs_processor.max_obj - self.obs_processor.min_obj)
                + self.obs_processor.min_obj
            )
        else:
            y_preds = y_preds + self.obs_processor.min_obj

        return y_preds

    def get_feasibility_surrogate(
        self, params: Union[List, pd.DataFrame], threshold: float = None
    ) -> List:
        """Retrieve the feasibility surrogate model.

        :param params: List of dicts with input parameters to evaluate. Alternatively it
            can also be a pandas DataFrame where each column name corresponds to one of
            the input parameters in Gryffin.
        :param threshold: Threshold used to classify whether a set of parameters is feasible or not.
            If ``None``, the probability of feasibility is returned instead of a binary True/False
            (feasible/infeasible) output. Default is None.

        :return y_pred: Surrogate model evaluated at the locations defined in params.

        """
        if isinstance(params, pd.DataFrame):
            params = self._df_to_list_of_dicts(params)

        X = param_dicts_to_vectors(
            params,
            param_names=self.config.param_names,
            param_options=self.config.param_options,
            param_types=self.config.param_types,
        )
        y_preds = []
        for x in X:
            if threshold is None:
                y_pred = self.bayesian_network.prob_feasible(x)
            else:
                y_pred = self.bayesian_network.classification_surrogate(
                    x, threshold=threshold
                )
            y_preds.append(y_pred)
        return np.array(y_preds)

    def get_kernel_density_estimate(
        self, params: Union[List, pd.DataFrame], separate_kwn_ukwn: bool = False
    ) -> List:
        """Retrieve the feasibility surrogate model.

        :param params: List of dicts with input parameters to evaluate. Alternatively it
            can also be a pandas DataFrame where each column name corresponds to one of
            the input parameters in Gryffin.
        :param separate_kwn_ukwn:  Return the density for all samples, or separate the density for feasible/infeasible samples.

        :return y_pred: Kernel density estimates.

        """
        if isinstance(params, pd.DataFrame):
            params = self._df_to_list_of_dicts(params)

        X = param_dicts_to_vectors(
            params,
            param_names=self.config.param_names,
            param_options=self.config.param_options,
            param_types=self.config.param_types,
        )
        y_preds = []
        for x in X:
            (
                log_density_0,
                log_density_1,
            ) = self.bayesian_network.kernel_classification.get_binary_kernel_densities(
                x.astype(np.float64)
            )
            density_0 = np.exp(log_density_0)
            density_1 = np.exp(log_density_1)
            if separate_kwn_ukwn is True:
                y_pred = [density_0, density_1]
            else:
                y_pred = density_0 + density_1
            y_preds.append(y_pred)
        return np.array(y_preds)

    def get_acquisition(self, X):
        """Retrieve the last acquisition functions for a specific lambda value."""
        if isinstance(X, pd.DataFrame):
            X = self._df_to_list_of_dicts(X)
        X_parsed = param_dicts_to_vectors(
            X,
            param_names=self.config.param_names,
            param_options=self.config.param_options,
            param_types=self.config.param_types,
        )

        # collect acquisition values
        acquisition_values = {}
        for batch_index, sampling_param in enumerate(self.sampling_param_values):
            acquisition_values_at_l = []
            for Xi_parsed in X_parsed:
                acq_value = self.acquisition.eval_acquisition(Xi_parsed, batch_index)
                acquisition_values_at_l.append(acq_value)

            lambda_value = self.sampling_strategies[batch_index]
            acquisition_values[lambda_value] = np.array(acquisition_values_at_l)

        return acquisition_values

    def get_descriptor_summary_regression(self):
        """Retrieve a summary of descriptor relavances for the regression surrogate"""
        return self.descriptor_generator.get_summary()

    def get_descriptor_summary_feasibility(self):
        """Retrieve a summary of the descriptor relavances for the feasibility surrogate"""
        return self.descriptor_generator_feas.get_summary()
