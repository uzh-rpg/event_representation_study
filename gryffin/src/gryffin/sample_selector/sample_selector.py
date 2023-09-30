#!/usr/bin/env python

__author__ = "Florian Hase, Matteo Aldeghi"

import numpy as np
import time
from contextlib import nullcontext
import multiprocessing
from multiprocessing import Manager, Process

from gryffin.utilities import Logger, parse_time


class SampleSelector(Logger):
    def __init__(self, config, all_options=None):
        self.config = config
        # factor modulating the density-based penalty in sample selector
        self.dist_param = self.config.get("dist_param")

        self.all_options = all_options

        self.verbosity = self.config.get("verbosity")
        Logger.__init__(self, "SampleSelector", verbosity=self.verbosity)
        # figure out how many CPUs to use
        if self.config.get("num_cpus") == "all":
            self.num_cpus = multiprocessing.cpu_count()
        else:
            self.num_cpus = int(self.config.get("num_cpus"))

        # check to see what kind of problem we have: fully continuous,
        # mixed categorical continuous/discrete, fully discrete or fully categorical
        param_types = [param["type"] for param in self.config.parameters]
        unique_param_types = sorted(list(set(param_types)))

        if unique_param_types == ["continuous"] or unique_param_types == [
            "continuous",
            "discrete",
        ]:
            self.problem_type = "fully_continuous"
        elif unique_param_types == ["categorical"]:
            self.problem_type = "fully_categorical"
        elif unique_param_types == ["discrete"]:
            self.problem_type = "fully_discrete"
        elif unique_param_types == [
            "categorical",
            "continuous",
        ] or unique_param_types == ["categorical", "continuous", "discrete"]:
            self.problem_type = "mixed_continuous"
        elif unique_param_types == ["categorical", "discrete"]:
            self.problem_type = "mixed_discrete"
        else:
            raise GryffinNotFoundError

        # get the non-categorical indices of the params space
        self.non_cat_param_idx = np.array(
            [i for i, param in enumerate(param_types) if param != "categorical"]
        )

    @staticmethod
    def compute_exp_objs(
        proposals,
        eval_acquisition,
        sampling_param_idx,
        return_index=0,
        return_dict=None,
    ):
        # batch_index is the index of the sampling_param_values used
        samples = proposals[sampling_param_idx]
        exp_objs = np.empty(len(samples))

        for sample_index, sample in enumerate(samples):
            acq = eval_acquisition(
                sample, sampling_param_idx
            )  # this is a method of the Acquisition instance
            exp_objs[sample_index] = np.exp(-acq)

        if return_dict.__class__.__name__ == "DictProxy":
            return_dict[return_index] = exp_objs

        return exp_objs

    def _compute_exp_objs(self, proposals, eval_acquisition, sampling_param_values):
        exp_objs = []
        # -----------------------------------------
        # compute exponential of acquisition values
        # -----------------------------------------
        # TODO: this is slightly redundant as we have computed acquisition values already in Acquisition
        for sampling_param_idx, sampling_param in enumerate(sampling_param_values):
            # -------------------
            # parallel processing
            # -------------------
            if self.num_cpus > 1:
                return_dict = Manager().dict()
                # split proposals into approx equal chunks based on how many CPUs we're using
                proposals_splits = np.array_split(proposals, self.num_cpus, axis=1)
                # parallelize over splits
                # -----------------------
                processes = []
                for return_idx, proposals_split in enumerate(proposals_splits):
                    process = Process(
                        target=self.compute_exp_objs,
                        args=(
                            proposals_split,
                            eval_acquisition,
                            sampling_param_idx,
                            return_idx,
                            return_dict,
                        ),
                    )
                    processes.append(process)
                    process.start()
                # wait until all processes finished
                for process in processes:
                    process.join()
                # sort results in return_dict to create batch_exp_objs list with correct sample order
                batch_exp_objs = []
                for idx in range(len(proposals_splits)):
                    batch_exp_objs.extend(return_dict[idx])
            # ---------------------
            # sequential processing
            # ---------------------
            else:
                batch_exp_objs = self.compute_exp_objs(
                    proposals=proposals,
                    eval_acquisition=eval_acquisition,
                    sampling_param_idx=sampling_param_idx,
                    return_index=0,
                    return_dict=None,
                )
            # append the proposed samples for this sampling strategy to the global list of samples
            exp_objs.append(batch_exp_objs)
        # cast to np.array
        exp_objs = np.array(exp_objs)

        return exp_objs

    def select(
        self,
        num_batches,
        proposals,
        eval_acquisition,
        sampling_param_values,
        obs_params,
    ):
        """

        Args:
            num_batches (int): number of samples to select per sampling strategy ('batches'
                argumnet in the configuration)
            proposals (np.nadarray): array of proposals from the acquisition function, shape
                is (num_sampling_strategies, num_samples, num_dims)
            eval_acquisition (obj): instance of acquisition function
            sampling_param_values (np.ndarray): array of sampling parameter values
            obs_params (np.ndarray): array of parameter points which have already been measured,
                shape is (num_obs, num_dims)

        """
        start = time.time()
        if self.verbosity > 3.5:  # i.e. INFO or DEBUG
            cm = self.console.status("Selecting best samples to recommend...")
        else:
            cm = nullcontext()
        with cm:
            samples = self._select(
                num_batches,
                proposals,
                eval_acquisition,
                sampling_param_values,
                obs_params,
            )

        end = time.time()
        time_string = parse_time(start, end)
        samples_str = "samples" if len(samples) > 1 else "sample"
        self.log(f"{len(samples)} {samples_str} selected in {time_string}", "STATS")

        return samples

    def _select(
        self,
        num_batches,
        proposals,
        eval_acquisition,
        sampling_param_values,
        obs_params,
    ):
        """

        Args:
            num_batches (int): number of samples to select per sampling strategy ('batches'
                argumnet in the configuration)
            proposals (np.nadarray): array of proposals from the acquisition function, shape
                is (num_sampling_strategies, num_samples, num_dims)
            eval_acquisition (obj): instance of acquisition function
            sampling_param_values (np.ndarray): array of sampling parameter values
            obs_params (np.ndarray): array of parameter points which have already been measured,
                shape is (num_obs, num_dims)

        """

        # exponential of negative acquisition function values, i.e. np.exp(-acq)
        exp_objs = self._compute_exp_objs(
            proposals, eval_acquisition, sampling_param_values
        )  # (num_sampling_strategies, num_propsals)

        # -------------------------------
        # compute prior recommendations
        # -------------------------------
        # compute normalised obs_params. In this way, we can rely on normalized distance thresholds, otherwise
        # if obs_params has very small range, sample selector is messed up
        obs_params_norm = (obs_params - self.config.param_lowers) / (
            self.config.param_uppers - self.config.param_lowers
        )
        proposals_norm = (proposals - self.config.param_lowers) / (
            self.config.param_uppers - self.config.param_lowers
        )

        # here we set to zero the reward if proposals are too close to previous observed params
        for sampling_param_idx in range(len(sampling_param_values)):
            batch_proposals = proposals_norm[
                sampling_param_idx, : exp_objs.shape[1]
            ]  # (num_proposals, num_dims)

            # compute distance to each obs_param
            distances = [
                np.sum((obs_params_norm - batch_proposal) ** 2, axis=1)
                for batch_proposal in batch_proposals
            ]
            distances = np.array(distances)  # (num_propsals, num_obs)
            # take min distance across previous observations
            min_distances = np.amin(distances, axis=1)  # (num_propsals,)
            # get indices for proposals that are basically the same as previous samples
            ident_indices = np.where(min_distances < 1e-8)[0]
            # set reward to zero for these samples since we do not want to select them
            exp_objs[sampling_param_idx, ident_indices] = 0.0

        # -----------------
        # collect samples
        # -----------------

        if self.problem_type is not "fully_categorical":
            num_obs = len(obs_params)
            feature_ranges = self.config.feature_ranges[self.non_cat_param_idx]
            char_dists = feature_ranges / float(num_obs) ** self.dist_param

            # here we add a penalty term that depends on the minimum distance between the proposals and
            # previously sampled or measured
            selected_samples = []
            for batch_idx in range(num_batches):
                for sampling_param_idx in range(len(sampling_param_values)):
                    batch_proposals = proposals[
                        sampling_param_idx, :, :
                    ]  # (num_proposals, num_dims)
                    batch_proposals = batch_proposals[
                        :, self.non_cat_param_idx
                    ]  # (num_proposals, num_non_cat_dims)

                    # compute diversity punishments
                    num_proposals_in_batch = exp_objs.shape[1]
                    div_crits = np.ones(num_proposals_in_batch)  # (num_proposals,)

                    # iterate over batch proposals and compute min distances to previous observations
                    # or other proposed samples
                    for proposal_idx, proposal in enumerate(batch_proposals):
                        # compute min distance to observed samples
                        obs_min_distance = np.amin(
                            [
                                np.abs(proposal - x[self.non_cat_param_idx])
                                for x in obs_params
                            ],
                            axis=0,
                        )
                        # if we already chose a new sample, compute also min distance to newly chosen samples
                        if len(selected_samples) > 0:
                            min_distance = np.amin(
                                [
                                    np.abs(proposal - x[self.non_cat_param_idx])
                                    for x in selected_samples
                                ],
                                axis=0,
                            )
                            min_distance = np.minimum(min_distance, obs_min_distance)
                        else:
                            min_distance = obs_min_distance

                        # compute distance reward
                        div_crits[proposal_idx] = np.minimum(
                            1.0,
                            np.mean(
                                np.exp(
                                    2.0 * (min_distance - char_dists) / feature_ranges
                                )
                            ),
                        )

                    # reweight computed based on acquisition with rewards based on distance
                    reweighted_rewards = (
                        exp_objs[sampling_param_idx] * div_crits
                    )  # (num_proposals,)
                    # get index of proposal with largest rewards
                    largest_reward_index = np.argmax(reweighted_rewards)

                    # select the sample from batch_proposals
                    # not from batch_proposals_norm that was used only for computing penalties
                    new_sample = proposals[sampling_param_idx, largest_reward_index]
                    selected_samples.append(new_sample)

                    # update the reward of the selected sample
                    exp_objs[sampling_param_idx, largest_reward_index] = 0.0

                    if self.problem_type in ["fully_discrete", "mixed_discrete"]:
                        # take care of duplicate parameters for other sampling strategies
                        exp_objs = self.duplicate_manager(
                            new_sample, proposals, exp_objs
                        )

        else:
            selected_samples = []
            for batch_idx in range(num_batches):
                for sampling_param_idx in range(len(sampling_param_values)):
                    batch_proposals = proposals[
                        sampling_param_idx, :, :
                    ]  # (num_propsals, num_dims)
                    rewards = exp_objs[sampling_param_idx]
                    largest_reward_index = np.argmax(rewards)

                    new_sample = batch_proposals[largest_reward_index]
                    selected_samples.append(new_sample)

                    # take care fo duplicated parameters for other sampling strategies
                    exp_objs = self.duplicate_manager(new_sample, proposals, exp_objs)

        return np.array(selected_samples)

    def duplicate_manager(self, new_sample, proposals, exp_objs):
        """sets exp_obj to 0.0 for all proposals which have already been
        either measured or selected

        Args:
            new_sample ():
            proposals (np.ndarray):
            exp_objs (np.ndarray):
        """
        # get the indices for the duplicates
        samp_idxs, prop_idxs = np.where(np.all(proposals == new_sample, axis=2))
        # replace duplicated samples exp objs with 0.0
        exp_objs[samp_idxs, prop_idxs] = 0.0

        return exp_objs
