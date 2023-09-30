#!/usr/bin/env python

# cython: language_level=3
# cython: profile=True

__author__ = 'Florian Hase, Matteo Aldeghi'

import  cython
cimport cython
import  numpy as np
cimport numpy as np
from libc.math cimport exp, round


# ==================
# Distance functions
# ==================
@cython.cdivision(True)
cdef double _gauss(double x, double loc, double sqrt_prec):
    cdef double argument, result
    argument = 0.5 * (sqrt_prec * (x - loc))**2
    if argument > 200.:
        result = 0.
    else:
        result = exp(-argument) * sqrt_prec * 0.3989422804014327  # the number is 1. / np.sqrt(2 * np.pi)
    return result


@cython.cdivision(True)
cdef double _gauss_periodic(double x, double loc, double sqrt_prec, double var_range):

    cdef double argument, result, distance

    distance = abs(x - loc)
    if var_range - distance < distance:
        distance = var_range - distance

    argument = 0.5 * (distance * sqrt_prec)**2
    if argument > 200.:
        result = 0.
    else:
        result = exp(-argument) * sqrt_prec * 0.3989422804014327  # the number is 1. / np.sqrt(2 * np.pi)
    return result


# ==========
# Main Class
# ==========
cdef class KernelEvaluator:

    cdef int num_samples, num_obs, num_kernels, num_cats, num_continuous
    cdef double lower_prob_bound, inv_vol

    cdef np.ndarray np_locs, np_sqrt_precs, np_cat_probs
    cdef np.ndarray np_kernel_types, np_kernel_sizes, np_kernel_ranges
    cdef np.ndarray np_objs
    cdef np.ndarray np_probs

    var_dict = {}

    def __init__(self, locs, sqrt_precs, cat_probs, kernel_types, kernel_sizes, kernel_ranges,
                 lower_prob_bound, objs, inv_vol):

        self.np_locs          = locs
        self.np_sqrt_precs    = sqrt_precs
        self.np_cat_probs     = cat_probs
        self.np_kernel_types  = kernel_types
        self.np_kernel_sizes  = kernel_sizes
        self.np_kernel_ranges = kernel_ranges
        self.np_objs          = objs

        self.num_samples      = locs.shape[0]
        self.num_obs          = locs.shape[1]
        self.num_kernels      = locs.shape[2]
        self.lower_prob_bound = lower_prob_bound
        self.inv_vol          = inv_vol

        self.np_probs = np.zeros(self.num_obs, dtype = np.float64)

        # number of continuous variables (continuous kernels have id 0 or 1)
        self.num_continuous = np.sum(kernel_types < 1.5)

    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef double [:] _probs(self, double [:] sample):

        cdef int    sample_index, obs_index, feature_index, kernel_index
        cdef double total_prob, prec_prod, exp_arg_sum, distance

        cdef double [:, :, :] locs       = self.np_locs
        cdef double [:, :, :] sqrt_precs = self.np_sqrt_precs
        cdef double [:, :, :] cat_probs  = self.np_cat_probs

        cdef int [:] kernel_types = self.np_kernel_types
        cdef int [:] kernel_sizes = self.np_kernel_sizes
        cdef double [:] kernel_ranges = self.np_kernel_ranges

        cdef double inv_sqrt_two_pi = 0.3989422804014327

        cdef double [:] probs = self.np_probs
        for obs_index in range(self.num_obs):
            probs[obs_index] = 0.

        cdef double cat_prob
        cdef double obs_probs

        # for each kernel location
        for obs_index in range(self.num_obs):
            obs_probs = 0.

            # for each BNN sample
            for sample_index in range(self.num_samples):
                total_prob     = 1.
                prec_prod      = 1.
                exp_arg_sum    = 0.
                feature_index, kernel_index = 0, 0

                # for each kernel/dimension
                while kernel_index < self.num_kernels:
                    # -----------------
                    # continuous kernel
                    # -----------------
                    if kernel_types[kernel_index] == 0:
                        # get product of inverse standard deviations
                        # this is the product of each dimension's contribution
                        prec_prod = prec_prod * sqrt_precs[sample_index, obs_index, kernel_index]
                        # get sum of the exponent argument
                        # (x_k - \phi_3(\theta, x_k))^2
                        exp_arg_sum = exp_arg_sum + (sqrt_precs[sample_index, obs_index, kernel_index] * (sample[feature_index] - locs[sample_index, obs_index, kernel_index]))**2

                    # --------------------------
                    # continuous periodic kernel
                    # --------------------------
                    elif kernel_types[kernel_index] == 1:
                        # get product of inverse standard deviations
                        prec_prod = prec_prod * sqrt_precs[sample_index, obs_index, kernel_index]
                        # get distance between gaussian mean and sample location
                        distance = abs(sample[feature_index] - locs[sample_index, obs_index, kernel_index])
                        # consider closest distance across boundaries
                        if kernel_ranges[kernel_index] - distance < distance:
                            distance = kernel_ranges[kernel_index] - distance
                        # get sum of the exponent argument
                        exp_arg_sum = exp_arg_sum + (sqrt_precs[sample_index, obs_index, kernel_index] * distance)**2

                    # ------------------
                    # categorical kernel
                    # ------------------
                    elif kernel_types[kernel_index] == 2:
                        # total probability, if we have continuous variables only this stays at 1
                        total_prob *= cat_probs[sample_index, obs_index, kernel_index + <int>round(sample[feature_index])]

                    # increment indices
                    kernel_index  += kernel_sizes[kernel_index]  # kernel size can be >1 for a certain param
                    feature_index += 1

                # combine precision product with exponent argument, and categorical probability
                obs_probs += total_prob * prec_prod * exp(-0.5 * exp_arg_sum)

                # we assume 1000 BNN samples, so 100 is 10%
                if sample_index == 100:
                    # boosting criterion
                    if 0.01 * obs_probs * inv_sqrt_two_pi**self.num_continuous < self.lower_prob_bound:
                        probs[obs_index] = 0.01 * obs_probs
                        break
                else:
                    # we take the average across the BNN samples
                    # normalise the gaussian kernel probabilities
                    probs[obs_index] = (obs_probs * inv_sqrt_two_pi**self.num_continuous) / self.num_samples
        return probs

    cpdef get_kernel_contrib(self, np.ndarray sample):

        cdef int obs_index
        cdef double temp_0, temp_1
        cdef double inv_den

        cdef double [:] sample_memview = sample
        probs_sample = self._probs(sample_memview)

        # construct numerator and denominator of acquisition
        cdef double num = 0.
        cdef double den = 0.
        cdef double [:] objs = self.np_objs

        for obs_index in range(self.num_obs):
            temp_0 = objs[obs_index]
            temp_1 = probs_sample[obs_index]
            num += temp_0 * temp_1
            den += temp_1

        inv_den = 1. / (self.inv_vol + den)

        return num, inv_den, probs_sample

    cpdef get_regression_surrogate(self, np.ndarray sample):

        cdef int obs_index
        cdef double temp_0, temp_1
        cdef double inv_den
        cdef double y_pred
        cdef double [:] sample_memview = sample
        probs_sample = self._probs(sample_memview)

        # construct numerator and denominator of acquisition
        cdef double num = 0.
        cdef double den = 0.
        cdef double [:] objs = self.np_objs

        for obs_index in range(self.num_obs):
            temp_0 = objs[obs_index]
            temp_1 = probs_sample[obs_index]
            num += temp_0 * temp_1
            den += temp_1

        y_pred = num / (den + 1e-8)  # avoid division by zero
        return y_pred

    cpdef get_binary_kernel_densities(self, np.ndarray sample):

        cdef int obs_index
        cdef double density_0 = 0.  # density of feasible
        cdef double density_1 = 0.  # density of infeasible
        cdef double num_0 = 0.
        cdef double num_1 = 0.
        cdef double log_density_0
        cdef double log_density_1

        cdef double [:] sample_memview = sample
        probs_sample = self._probs(sample_memview)

        for obs_index, obj in enumerate(self.np_objs):
            if obj > 0.5:
                density_1 += probs_sample[obs_index]
                num_1 += 1.
            else:
                density_0 += probs_sample[obs_index]
                num_0 += 1.

        # normalize wrt the number of kernels
        log_density_0 = np.log(density_0) - np.log(num_0)
        log_density_1 = np.log(density_1) - np.log(num_1)

        return log_density_0, log_density_1

    cpdef get_probability_of_infeasibility(self, np.ndarray sample, double log_prior_0, double log_prior_1):

        # 0 = feasible, 1 = infeasible
        cdef double prob_infeas
        cdef double log_density_0
        cdef double log_density_1
        cdef double posterior_0
        cdef double posterior_1

        # get log probabilities
        log_density_0, log_density_1 = self.get_binary_kernel_densities(sample)

        # compute unnormalized posteriors
        posterior_0 = exp(log_density_0 + log_prior_0)
        posterior_1 = exp(log_density_1 + log_prior_1)

        # guard against zero division. This may happen if both densities are zero
        if np.log(posterior_0 + posterior_1) < - 230:  # i.e. less then 1e-100
            return exp(log_prior_1) / (exp(log_prior_0) + exp(log_prior_1))  # return prior prob

        # get normalized posterior for prob of infeasible
        prob_infeas = posterior_1 / (posterior_0 + posterior_1)

        return prob_infeas

    cpdef get_probability_of_feasibility(self, np.ndarray sample, double log_prior_0, double log_prior_1):

        # 0 = feasible, 1 = infeasible
        cdef double prob_feas
        cdef double log_density_0
        cdef double log_density_1
        cdef double posterior_0
        cdef double posterior_1

        # get log probabilities
        log_density_0, log_density_1 = self.get_binary_kernel_densities(sample)

        # compute unnormalized posteriors
        posterior_0 = exp(log_density_0 + log_prior_0)
        posterior_1 = exp(log_density_1 + log_prior_1)

        # guard against zero division. This may happen if both densities are zero
        if np.log(posterior_0 + posterior_1) < - 230:  # i.e. less then 1e-100
            return exp(log_prior_1) / (exp(log_prior_0) + exp(log_prior_1))  # return prior prob

        # get normalized posterior for prob of infeasible
        prob_feas = posterior_0 / (posterior_0 + posterior_1)

        return prob_feas
