#!/usr/bin/env python

__author__ = "Florian Hase"

import copy
import numpy as np
import multiprocessing
from gryffin.utilities import Logger
from .generator import Generator
from multiprocessing import Process, Manager
from copy import deepcopy


class DescriptorGenerator(Logger):
    eta = 1e-3
    max_iter = 10**3

    def __init__(self, config):
        self.config = config
        self.is_generating = False

        self.obs_params = None
        self.obs_objs = None
        self.gen_feature_descriptors = None

        verbosity = self.config.get("verbosity")
        Logger.__init__(self, "DescriptorGenerator", verbosity=verbosity)

        self.num_cpus = 1
        # if self.config.get('num_cpus') == 'all':
        #     self.num_cpus = multiprocessing.cpu_count()
        # else:
        #     self.num_cpus = int(self.config.get('num_cpus'))

    def _generate_single_descriptors(
        self,
        feature_index,
        result_dict=None,
        weights_dict=None,
        sufficient_indices_dict=None,
    ):
        """Parse description generation for a specific parameter, ad indicated by the feature_index"""

        self.log("running one optimization process", "DEBUG")

        feature_types = self.config.feature_types
        feature_descriptors = self.config.feature_descriptors
        obs_params = self.obs_params
        obs_objs = self.obs_objs

        # if continuous ==> no descriptors, return None
        if feature_types[feature_index] in ["continuous", "discrete"]:
            self.weights[feature_index] = None
            self.reduced_gen_descs[feature_index] = None
            if result_dict is not None:
                result_dict[feature_index] = None
            return None

        # if None, i.e. naive Gryffin ==> no descriptors, return None
        if feature_descriptors[feature_index] is None:
            self.weights[feature_index] = None
            self.reduced_gen_descs[feature_index] = None
            if result_dict is not None:
                result_dict[feature_index] = None
            return None

        # if single descriptor ==> cannot get new descriptors, return the same static descriptor
        if feature_descriptors[feature_index].shape[1] == 1:
            self.weights[feature_index] = np.array([[1.0]])
            self.reduced_gen_descs[feature_index] = feature_descriptors[feature_index]
            if result_dict is not None:
                result_dict[feature_index] = feature_descriptors[feature_index]
            return feature_descriptors[feature_index]

        # ------------------------------------------------------------------------------------------
        # Else, we have multiple descriptors for a categorical variable and we perform the reshaping
        # ------------------------------------------------------------------------------------------
        params = obs_params[:, feature_index].astype(np.int32)
        descs = feature_descriptors[feature_index][params]
        objs = np.reshape(obs_objs, (len(obs_objs), 1))

        # run the generation process
        generator = Generator(
            descs=descs, objs=objs, grid_descs=feature_descriptors[feature_index]
        )
        network_results = generator.generate_descriptors()

        # for key in network_results.keys():
        #     print(key, network_results[key].shape)

        # network_results consists of  -->
        # min_corrs: 1. / np.sqrt(self.num_samples - 2) SHAPE: (1,)
        # comp_corr_coeffs : correlations between the generated descriptors and the observed objectives SHAPE: (num_desc,)
        # gen_descs_cov: covariance of the generated descritptors SHAPE: (num_desc, num_desc)
        # weights: weights of the preceptron model SHAPE: (num_desc, num_desc)
        # auto_gen_descs: generated descriptors for each categorical option SHAPE (num_options, num_desc)
        # sufficient_indices: the descriptor indices for which the absolute value of the correlation is larger than the minimium correlation  (num_sufficient_indices,)
        # reduced_gen_descs: auto_gen_descs with only the sufficient_indices kept SHAPE (num_options, num_sufficient_indices)

        if result_dict is not None:
            result_dict[feature_index] = deepcopy(network_results["reduced_gen_descs"])
        if weights_dict is not None:
            weights_dict[feature_index] = deepcopy(network_results["weights"])
        if sufficient_indices_dict is not None:
            sufficient_indices_dict[feature_index] = deepcopy(
                network_results["sufficient_indices"]
            )

        return network_results["reduced_gen_descs"]

    def _generate_some_descriptors(
        self, feature_indices, result_dict, weights_dict, sufficient_indices_dict
    ):
        """Used by generate_descriptors when running in parallel"""

        # print some info
        feature_names = [self.config.feature_names[i] for i in feature_indices]
        features_strings = ", ".join(feature_names)
        self.log(
            f"running parallel descriptor generation for {features_strings}", "DEBUG"
        )

        # run
        for feature_index in feature_indices:
            _ = self._generate_single_descriptors(
                feature_index=feature_index,
                result_dict=result_dict,
                weights_dict=weights_dict,
                sufficient_indices_dict=sufficient_indices_dict,
            )

    @staticmethod
    def _custom_array_split(feature_types, feature_indices, num_splits):
        """Split feature indices into N splits such that the categorical variables are distributed evenly across
        splits. This matters because we run the generation only for categorical variables, so we do not want to have
        a process without any and other with multiple ones."""

        # sort the feature indices according to the alphabetical order of the feature types, so that we get, e.g.:
        # ['categorical', 'continuous', 'discrete', 'categorical', 'categorical']
        # ==> ['categorical', 'categorical', 'categorical', 'continuous', 'discrete', ]
        feature_types_sorted, feature_indices_sorted = zip(
            *sorted(zip(feature_types, feature_indices))
        )

        # create a 2D list
        feature_indices_splits = [[] for n in range(num_splits)]

        # distribute indices across splits
        for i, feature_index in enumerate(feature_indices_sorted):
            split_idx = i % num_splits
            feature_indices_splits[split_idx].append(feature_index)

        return feature_indices_splits

    def generate_descriptors(self, obs_params, obs_objs):
        """Generates descriptors for each categorical parameters"""

        self.obs_params = obs_params
        self.obs_objs = obs_objs

        feature_indices = range(len(self.config.feature_options))
        feature_types = self.config.feature_types
        assert len(feature_types) == len(feature_indices)

        # ------------------------------
        # Parallel descriptor generation
        # ------------------------------
        if self.num_cpus > 1:
            # do not use more splits than number of features available, otherwise we would get some empty splits
            num_splits = min(self.num_cpus, len(feature_indices))
            # splits indices in such a way to evenly distribute categorical vars across processes
            feature_indices_splits = self._custom_array_split(
                feature_types, feature_indices, num_splits
            )

            # store results in share memory dict
            self.weights = Manager().dict()
            self.sufficient_indices = Manager().dict()
            result_dict = Manager().dict()
            processes = []  # store parallel processes here

            for feature_indices_split in feature_indices_splits:
                # run optimization
                process = Process(
                    target=self._generate_some_descriptors,
                    args=(
                        feature_indices_split,
                        result_dict,
                        self.weights,
                        self.sufficient_indices,
                    ),
                )
                processes.append(process)
                process.start()

            # wait until all processes finished
            for process in processes:
                process.join()

        # ----------------------------
        # Serial descriptor generation
        # ----------------------------
        else:
            self.weights = {}
            self.sufficient_indices = {}
            result_dict = {}
            for feature_index in feature_indices:
                # gen_descriptor = self._generate_single_descriptors(feature_index=feature_index, result_dict=None)
                _ = self._generate_single_descriptors(
                    feature_index=feature_index,
                    result_dict=result_dict,
                    weights_dict=self.weights,
                    sufficient_indices_dict=self.sufficient_indices,
                )
                # result_dict[feature_index] = gen_descriptor

        # reorder correctly the descriptors following asynchronous execution
        gen_feature_descriptors = []
        for feature_index in range(len(result_dict.keys())):
            gen_feature_descriptors.append(result_dict[feature_index])

        self.gen_feature_descriptors = gen_feature_descriptors

    def get_descriptors(self):
        if self.gen_feature_descriptors is not None:
            return self.gen_feature_descriptors
        else:
            return self.config.feature_descriptors

    def get_summary(self):
        summary = {}
        feature_types = self.config.feature_types
        # If we have not generated new descriptors
        if self.gen_feature_descriptors is None:
            for feature_index in range(len(self.config.feature_options)):
                contribs = {}
                if feature_types[feature_index] == "continuous":
                    continue
                feature_descriptors = self.config.feature_descriptors[feature_index]
                if feature_descriptors is None:
                    continue
                for desc_index in range(feature_descriptors.shape[1]):
                    desc_summary_dict = {}
                    desc_summary_dict["relevant_given_descriptors"] = np.arange(
                        len(feature_descriptors[:, desc_index])
                    )
                    desc_summary_dict["given_descriptor_contributions"] = np.ones(
                        len(feature_descriptors[:, desc_index])
                    )
                    contribs["descriptor_%d" % desc_index] = copy.deepcopy(
                        desc_summary_dict
                    )
                summary["feature_%d" % feature_index] = copy.deepcopy(contribs)
            return summary

        # If we have generated new descriptors
        for feature_index in range(len(self.config.feature_options)):
            if feature_types[feature_index] == "continuous":
                continue

            weights = self.weights[feature_index]
            sufficient_indices = self.sufficient_indices[feature_index]

            if weights is None:
                continue
            if len(sufficient_indices) == 0:
                # we have no descriptors for which the correlations exceed the minimum correlation
                continue

            # normalize weights
            normed_weights = np.empty(weights.shape)
            for index, weight_elements in enumerate(weights):
                normed_weights[index] = weight_elements / np.sum(
                    np.abs(weight_elements)
                )

            # identify contributing indices
            contribs = {}
            for new_desc_index in sufficient_indices:
                desc_summary_dict = {}
                relevant_weights = normed_weights[new_desc_index]

                sorting_indices = np.argsort(np.abs(relevant_weights))
                cumulative_sum = np.cumsum(np.abs(relevant_weights[sorting_indices]))
                include_indices = np.where(cumulative_sum > 0.1)[0]

                relevant_given_descriptors = sorting_indices[include_indices]
                desc_summary_dict[
                    "relevant_given_descriptors"
                ] = relevant_given_descriptors
                desc_summary_dict["given_descriptor_contributions"] = weights[
                    new_desc_index
                ]
                contribs[f"descriptor_{new_desc_index}"] = copy.deepcopy(
                    desc_summary_dict
                )
            summary[f"feature_{feature_index}"] = copy.deepcopy(contribs)

        return summary
