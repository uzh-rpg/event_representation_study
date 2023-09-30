#!/usr/bin/env

__author__ = "Florian Hase"


import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NumpyGraph:
    def __init__(self, config, model_details):
        self.config = config

        self.model_details = model_details
        for key, value in self.model_details.items():
            setattr(self, "_%s" % str(key), value)

        self.feature_size = len(self.config.kernel_names)
        self.bnn_output_size = len(self.config.kernel_names)
        self.target_size = len(self.config.kernel_names)

    def declare_training_data(self, features):
        self.num_obs = len(features)
        self.features = features.numpy()

    def compute_kernels(self, posteriors, frac_feas):
        tau_rescaling = np.zeros((self.num_obs, self.bnn_output_size))
        kernel_ranges = self.config.kernel_ranges
        for obs_index in range(self.num_obs):
            tau_rescaling[obs_index] += kernel_ranges
        tau_rescaling = tau_rescaling**2

        # sample from BNN
        activations = [
            lambda x: np.maximum(x, 0),
            lambda x: np.maximum(x, 0),
            lambda x: x,
        ]
        post_layer_outputs = [np.array([self.features for _ in range(self._num_draws)])]

        for layer_index in range(self._num_layers):
            weight = posteriors["weight_%d" % layer_index]
            bias = posteriors["bias_%d" % layer_index]
            activation = activations[layer_index]

            outputs = []
            for sample_index in range(len(weight)):
                single_weight = weight[sample_index]
                single_bias = bias[sample_index]

                output = activation(
                    np.matmul(post_layer_outputs[-1][sample_index], single_weight)
                    + single_bias
                )
                outputs.append(output)

            post_layer_output = np.array(outputs)
            post_layer_outputs.append(post_layer_output)

        post_bnn_output = post_layer_outputs[-1]

        # note: np.random.gamma is parametrized with k and theta, while ed.models.Gamma is parametrized with alpha and beta
        post_tau_normed = np.random.gamma(
            12 * (self.num_obs / frac_feas) ** 2 + np.zeros(post_bnn_output.shape),
            np.ones(post_bnn_output.shape),
        )
        #        post_tau_normed = posteriors['gamma']   # shape = (1000, num_obs, 1)
        post_tau = post_tau_normed / tau_rescaling
        post_sqrt_tau = np.sqrt(post_tau)
        post_scale = 1.0 / post_sqrt_tau

        # map BNN output to predictions
        post_kernels = {}

        target_element_index = 0
        kernel_element_index = 0
        while kernel_element_index < len(self.config.kernel_names):
            kernel_type = self.config.kernel_types[kernel_element_index]
            kernel_size = self.config.kernel_sizes[kernel_element_index]

            feature_begin, feature_end = target_element_index, target_element_index + 1
            kernel_begin, kernel_end = (
                kernel_element_index,
                kernel_element_index + kernel_size,
            )

            post_relevant = post_bnn_output[:, :, kernel_begin:kernel_end]

            if kernel_type == "continuous":
                lowers = self.config.kernel_lowers[kernel_begin:kernel_end]
                uppers = self.config.kernel_uppers[kernel_begin:kernel_end]
                post_support = (uppers - lowers) * (
                    1.2 * sigmoid(post_relevant) - 0.1
                ) + lowers
                post_kernels["param_%d" % target_element_index] = {
                    "loc": post_support,
                    "sqrt_prec": post_sqrt_tau[:, :, kernel_begin:kernel_end],
                    "scale": post_scale[:, :, kernel_begin:kernel_end],
                }

            elif kernel_type == "categorical":
                post_temperature = 0.5 + 10.0 / (self.num_obs / frac_feas)
                # post_temperature = 0.4
                post_support = post_relevant

                post_probs = 1.0 / (1.0 + np.exp(-post_support))
                post_probs_normed = post_probs / np.sum(post_probs)

                # sample from relaxed one hot categorical
                gumbel_samples = np.random.gumbel(
                    loc=0.0, scale=1.0, size=post_support.shape
                )

                exp_samples = np.exp((post_support + gumbel_samples) / post_temperature)
                exp_samples_sums = np.sum(exp_samples, axis=2)

                post_predict_relaxed = np.zeros(exp_samples.shape)
                for sample_index in range(exp_samples.shape[0]):
                    for param_index in range(exp_samples.shape[1]):
                        post_predict_relaxed[sample_index, param_index] = (
                            exp_samples[sample_index, param_index]
                            / exp_samples_sums[sample_index, param_index]
                        )

                post_kernels["param_%d" % target_element_index] = {
                    "probs": post_predict_relaxed
                }

            elif kernel_type == "discrete":
                post_temperature = 0.5 + 1.0 / (self.num_obs / frac_feas)
                # post_temperature = 0.4
                post_support = post_relevant

                post_probs = 1.0 / (1.0 + np.exp(-post_support))
                post_probs_normed = post_probs / np.sum(post_probs)

                # sample from relaxed one hot categorical
                gumbel_samples = np.random.gumbel(
                    loc=0.0, scale=1.0, size=post_support.shape
                )
                exp_samples = np.exp((post_support + gumbel_samples) / post_temperature)
                exp_samples_sums = np.sum(exp_samples, axis=2)

                post_predict_relaxed = np.zeros(exp_samples.shape)
                for sample_index in range(exp_samples.shape[0]):
                    for param_index in range(exp_samples.shape[1]):
                        post_predict_relaxed[sample_index, param_index] = (
                            exp_samples[sample_index, param_index]
                            / exp_samples_sums[sample_index, param_index]
                        )

                post_kernels["param_%d" % target_element_index] = {
                    "probs": post_predict_relaxed
                }

            else:
                raise NotImplementedError

            target_element_index += 1
            kernel_element_index += kernel_size

        return post_kernels
