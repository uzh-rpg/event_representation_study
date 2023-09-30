#!/usr/bin/env python

__author__ = "Florian Hase, Matteo Aldeghi"

import numpy as np
import time
import multiprocessing
from multiprocessing import Process, Manager
from contextlib import nullcontext

from .gradient_optimizer import GradientOptimizer
from gryffin.random_sampler import RandomSampler
from gryffin.utilities import Logger, parse_time, GryffinUnknownSettingsError
from gryffin.observation_processor import param_dict_to_vector

import ast
from functools import wraps
import base64
import pickle

gryffin_counter = -1


class Acquisition(Logger):
    def __init__(self, config, known_constraints=None):
        self.config = config
        self.known_constraints = known_constraints

        self.verbosity = self.config.get("verbosity")
        Logger.__init__(self, "Acquisition", verbosity=self.verbosity)

        self.total_num_vars = len(self.config.feature_names)
        self.optimizer_type = self.config.get("acquisition_optimizer")

        self.bayesian_network = None
        self.local_optimizers = None
        self.sampling_param_values = None
        self.frac_infeasible = None
        self.acqs_min_max = None  # expected content is dict where key is batch_index, and dict[batch_index] = [min,max]
        self.acquisition_functions = (
            {}
        )  # to keep the AcquisitionFunction instances used

        # figure out how many CPUs to use
        if self.config.get("num_cpus") == "all":
            self.num_cpus = multiprocessing.cpu_count()
        else:
            self.num_cpus = int(self.config.get("num_cpus"))

        # get feasibility approach and sensitivity parameter and do some checks
        self.feas_approach = self.config.get("feas_approach")
        self.feas_param = self.config.get("feas_param")
        self._check_feas_options()

    def _check_feas_options(self):
        if self.feas_approach not in ["fwa", "fia", "fca"]:
            self.log(
                f'Cannot understand "feas_approach" option "{self.feas_approach}". '
                f'Defaulting to "fwa".',
                "WARNING",
            )
            self.feas_approach = "fwa"

        if self.feas_approach == "fia":
            if self.feas_param < 0.0:
                self.log(
                    "Config parameter `feas_param` should be positive, applying np.abs()",
                    "WARNING",
                )
                self.feas_param = np.abs(self.feas_param)
            elif self.feas_param == 0.0:
                self.log(
                    "Config parameter `feas_param` cannot be zero, falling back to default value of 1",
                    "WARNING",
                )
                self.feas_param = 1.0

        if self.feas_approach == "fca":
            if self.feas_param <= 0.0 or self.feas_param >= 1.0:
                self.log(
                    "Config parameter `feas_param` should be between zero and one when `feas_approach` is "
                    '"fca", falling back to default of 0.5',
                    "WARNING",
                )
                self.feas_param = 0.5

    def _propose_randomly(
        self,
        best_params,
        num_samples_per_dim,
        acquisition_constraints,
        dominant_samples=None,
    ):
        """

        Parameters
        ----------
        acquisition_constraints : list
            list of constraints functions.
        dominant_samples :
            dominant samples for batch constraints.

        Returns
        -------
        random_samples : ndarray
            array with random samples in the optimization domain.
        """

        # number of random samples grows linearly with domain dimensionality
        num_samples = num_samples_per_dim * self.total_num_vars

        # -------------------
        # parallel processing
        # -------------------
        if self.num_cpus > 1:
            # create shared memory dict
            return_list = Manager().list()

            # request num_sample/num_cpus random proposals from each process
            num_samples_batch = int(np.round(num_samples / self.num_cpus, decimals=0))

            # parallelize over batches of random samples
            processes = []  # store parallel processes here
            for idx in range(self.num_cpus):
                # run optimization
                process = Process(
                    target=self._propose_randomly_thread,
                    args=(
                        best_params,
                        num_samples_batch,
                        acquisition_constraints,
                        dominant_samples,
                        return_list,
                    ),
                )
                processes.append(process)
                process.start()

            # wait until all processes finished
            for process in processes:
                process.join()

            # concatenate all random samples into the same array
            random_samples = np.concatenate(return_list)

        # ---------------------
        # sequential processing
        # ---------------------
        else:
            # optimized samples for this batch/sampling strategy
            random_samples = self._propose_randomly_thread(
                best_params=best_params,
                num_samples=num_samples,
                acquisition_constraints=acquisition_constraints,
                dominant_samples=dominant_samples,
                return_list=None,
            )

        return random_samples

    def _propose_randomly_thread(
        self,
        best_params,
        num_samples,
        acquisition_constraints,
        dominant_samples=None,
        return_list=None,
    ):
        """
        acquisition_constraints : list
            list of constraints functions.
        dominant_samples :
            dominant samples for batch constraints.
        """

        random_sampler = RandomSampler(self.config, constraints=acquisition_constraints)

        # -------------------
        # get uniform samples
        # -------------------
        if dominant_samples is None:
            uniform_samples = random_sampler.draw(num=num_samples)
            perturb_samples = random_sampler.perturb(best_params, num=num_samples)
            samples = np.concatenate([uniform_samples, perturb_samples])
        else:
            dominant_features = self.config.feature_process_constrained
            for batch_sample in dominant_samples:
                uniform_samples = random_sampler.draw(
                    num=num_samples // len(dominant_samples)
                )
                perturb_samples = random_sampler.perturb(best_params, num=num_samples)
                samples = np.concatenate([uniform_samples, perturb_samples])
            samples[:, dominant_features] = batch_sample[dominant_features]

        # append to shared memory list if present
        if return_list is not None:
            return_list.append(samples)

        return samples

    def _proposal_optimization_thread(
        self,
        proposals,
        acquisition,
        acquisition_constraints,
        return_dict=None,
        return_index=0,
        dominant_samples=None,
    ):
        self.log("running one optimization process", "DEBUG")

        # get params to be constrained
        if dominant_samples is not None:
            ignore = self.config.feature_process_constrained
        else:
            ignore = np.array(
                [False for _ in range(len(self.config.feature_process_constrained))]
            )

        # get the optimizer instance and set function to be optimized
        local_optimizer = self._load_optimizer(
            acquisition_constraints=acquisition_constraints
        )
        local_optimizer.set_func(acquisition, ignores=ignore)

        # run acquisition optimization
        if self.verbosity > 3.5:  # i.e. INFO or DEBUG
            show_progress = True
        else:
            show_progress = False
        optimized = local_optimizer.optimize(
            proposals, max_iter=10, show_progress=show_progress
        )

        if return_dict.__class__.__name__ == "DictProxy":
            return_dict[return_index] = optimized

        return optimized

    def _get_approx_min_max(self, random_proposals, sampling_param, dominant_samples):
        """Approximate min and max of sample acquisition to that we can approximately normalize it"""

        # If we only have feasible or infeasible points, no need to compute max/min as there is no need to rescale the
        # sample acquisition, because the acquisition will only be for feasible samples or for feasibility search
        if self.frac_infeasible < 1e-6 or (1.0 - self.frac_infeasible) < 1e-6:
            return 0.0, 1.0
        # return 0,1 also if we are using a feasibility-constrained acquisition, as
        # in this case there is no need to normalize _acquisition_all_feasible
        if self.feas_approach == "fca":
            return 0.0, 1.0

        acq_values = []
        for proposal in random_proposals:
            num, inv_den = self.bayesian_network.kernel_contribution(proposal)
            acq_samp = (num + sampling_param) * inv_den
            acq_values.append(acq_samp)

        acq_values = np.array(acq_values)

        # take top/bottom 5% of samples...
        n = int(round(len(random_proposals) * 0.05, 0))
        indices_top = (-acq_values).argsort()[:n]  # indices of highest n
        indices_bottom = acq_values.argsort()[:n]  # indices of lowest n

        top_params = random_proposals[indices_top, :]  # params of highest n
        bottom_params = random_proposals[indices_bottom, :]  # params of lowest n

        # define acquisition function to be optimized. With acq_min=0, acq_max=1 we are not scaling it.
        acquisition = AcquisitionFunction(
            bayesian_network=self.bayesian_network,
            sampling_param=sampling_param,
            acq_min=0,
            acq_max=1,
            feas_approach=self.feas_approach,
            feas_param=1.0,
        )
        # manually set acquitision function to be the acquisition for the samples only (no feasibility involved)
        acquisition.acquisition_function = acquisition._acquisition_all_feasible
        acquisition.feasibility_weight = None

        # get params to be constrained
        if dominant_samples is not None:
            ignore = self.config.feature_process_constrained
        else:
            ignore = np.array(
                [False for _ in range(len(self.config.feature_process_constrained))]
            )

        # ----------------------
        # minimise lowest values
        # ----------------------
        optimizer_bottom = GradientOptimizer(self.config, self.known_constraints)
        optimizer_bottom.set_func(acquisition, ignores=ignore)
        optimized = optimizer_bottom.optimize(bottom_params, max_iter=10)

        bottom_acq_values = np.array([acquisition(x) for x in optimized])
        # concatenate with randomly collected acq values
        bottom_acq_values = np.concatenate((acq_values, bottom_acq_values), axis=0)

        # -----------------------
        # maximise highest values
        # -----------------------
        def inv_acquisition(x):
            """Invert acquisition for its maximisation"""
            return -acquisition(x)

        optimizer_top = GradientOptimizer(self.config, self.known_constraints)
        optimizer_top.set_func(inv_acquisition, ignores=ignore)
        optimized = optimizer_top.optimize(top_params, max_iter=10)

        top_acq_values = np.array([acquisition(x) for x in optimized])
        # concatenate with randomly collected acq values
        top_acq_values = np.concatenate((acq_values, top_acq_values), axis=0)

        # min and max values found
        acq_min = np.min(bottom_acq_values)
        acq_max = np.max(top_acq_values)

        # if min > max, or if the different is very small, the acquisition is flat,
        # or something else is wrong, so we discard the results
        if acq_max - acq_min < 1e-6:
            self.log(
                f"The extrema could not be located correctly (min = {acq_min}, max = {acq_max}). "
                f"The acquisition function might be flat.",
                "WARNING",
            )
            acq_min = 0.0
            acq_max = 1.0
        return acq_min, acq_max

    def _optimize_proposals(
        self, random_proposals, acquisition_constraints=None, dominant_samples=None
    ):
        optimized_samples = (
            []
        )  # all optimized samples, i.e. for all sampling strategies
        self.acqs_min_max = {}

        # ------------------------------------
        # Iterate over all sampling strategies
        # ------------------------------------
        for batch_index, sampling_param in enumerate(self.sampling_param_values):
            # time
            start_opt = time.time()

            # get approximate min/max of sample acquisition
            if self.verbosity > 3.5:
                with self.console.status(
                    "Performing acquisition optimization pre-processing tasks..."
                ):
                    acq_min, acq_max = self._get_approx_min_max(
                        random_proposals, sampling_param, dominant_samples
                    )
            else:
                acq_min, acq_max = self._get_approx_min_max(
                    random_proposals, sampling_param, dominant_samples
                )
            self.acqs_min_max[batch_index] = [acq_min, acq_max]

            # define acquisition function to be optimized
            acquisition = AcquisitionFunction(
                bayesian_network=self.bayesian_network,
                sampling_param=sampling_param,
                acq_min=acq_min,
                acq_max=acq_max,
                feas_approach=self.feas_approach,
                feas_param=self.feas_param,
            )

            # save acquisition instance for future use
            if batch_index not in self.acquisition_functions.keys():
                self.acquisition_functions[batch_index] = acquisition

            # -------------------
            # parallel processing
            # -------------------
            if self.num_cpus > 1:
                # create shared memory dict that will contain the optimized samples for this batch/sampling strategy
                # keys will correspond to indices so that we can resort the samples afterwards
                return_dict = Manager().dict()

                # split random_proposals into approx equal chunks based on how many CPUs we're using
                random_proposals_splits = np.array_split(
                    random_proposals, self.num_cpus
                )

                # parallelize over splits
                # -----------------------
                processes = []  # store parallel processes here
                for idx, random_proposals_split in enumerate(random_proposals_splits):
                    # run optimization
                    process = Process(
                        target=self._proposal_optimization_thread,
                        args=(
                            random_proposals_split,
                            acquisition,
                            acquisition_constraints,
                            return_dict,
                            idx,
                            dominant_samples,
                        ),
                    )
                    processes.append(process)
                    process.start()

                # wait until all processes finished
                for process in processes:
                    process.join()

                # sort results in return_dict to create optimized_batch_samples list with correct sample order
                optimized_batch_samples = []
                for idx in range(len(random_proposals_splits)):
                    optimized_batch_samples.extend(return_dict[idx])

            # ---------------------
            # sequential processing
            # ---------------------
            else:
                # optimized samples for this batch/sampling strategy
                optimized_batch_samples = self._proposal_optimization_thread(
                    proposals=random_proposals,
                    acquisition=acquisition,
                    acquisition_constraints=acquisition_constraints,
                    return_dict=None,
                    return_index=0,
                    dominant_samples=dominant_samples,
                )

            # append the optimized samples for this sampling strategy to the global list of optimized_samples
            optimized_samples.append(optimized_batch_samples)

            # print info to screen
            end_opt = time.time()
            time_string = parse_time(start_opt, end_opt)
            self.log(
                f"{len(optimized_batch_samples)} proposals optimized in {time_string} "
                f"using {self.num_cpus} CPUs",
                "STATS",
            )

        return np.array(optimized_samples)

    def _load_optimizer(self, acquisition_constraints):
        if self.optimizer_type == "adam":
            local_optimizer = GradientOptimizer(self.config, acquisition_constraints)
        elif self.optimizer_type == "genetic":
            from .genetic_optimizer import GeneticOptimizer

            local_optimizer = GeneticOptimizer(self.config, acquisition_constraints)
        else:
            GryffinUnknownSettingsError(
                f"Did not understand optimizer choice {self.optimizer_type}."
                f'\n\tPlease choose "adam" or "genetic"'
            )
        return local_optimizer

    def propose(
        self,
        best_params,
        bayesian_network,
        sampling_param_values,
        num_samples_per_dim=200,
        dominant_samples=None,
        timings_dict=None,
    ):
        """Collect proposals by random sampling plus refinement. Highest-level method of this class that takes the BNN
        results, builds the acquisition function, optimises it, and returns a number of possible parameter points.
        These will then be used by the SampleSelector to pick the parameters to suggest.

        Parameters
        ----------
        best_params : array
        bayesian_network :
        sampling_param_values : list
        num_samples_per_dim : int
            Number of samples to randomly draw per parameter dimension. E.g. for a two-dimensional search domain,
            we draw ``num_samples`` * 2 samples.
        dominant_samples : array
        timings_dict : dict

        Returns
        -------
        proposals : array
            Numpy array with the proposals, with shape (# sampling_param_values, # proposals, # dimensions).
        """

        start_overall = time.time()

        # -------------------------------------------------------------
        # register attributes we'll be using to compute the acquisition
        # -------------------------------------------------------------
        self.bayesian_network = bayesian_network
        self.acquisition_functions = (
            {}
        )  # reinitialize acquisition functions, otherwise we keep using old ones!
        self.sampling_param_values = sampling_param_values
        self.frac_infeasible = bayesian_network.prior_1

        # if using feasibility-constrained acquisition, we need to constrain the optimizers
        # However, do not use constraints if fraction of feasible samples is zero or one, in which case we
        # we won't be needing the classification model
        if (
            self.feas_approach == "fca"
            and self.bayesian_network.prior_1 > 1e-6
            and self.bayesian_network.prior_0 > 1e-6
        ):
            # use adaptive approach to make sure at least 10% of the domain is classified as feasible
            # if not, we temporarily relax the feas_param until we have 10% feasibility
            original_feas_param = self.feas_param
            # do check and update self.feas_param if needed
            self._update_feas_param(num=self.total_num_vars * num_samples_per_dim)

            # if we also have already known_constraints, we need to merge them
            if self.known_constraints is not None:
                acquisition_constraints = [
                    self.known_constraints,
                    self._feasibility_constraint,
                ]
            # otherwise, we only have feasibility constraints
            else:
                acquisition_constraints = [self._feasibility_constraint]
        # if not using feasibility-constrained acquisition, then only constraints are the known_constraints, if any
        else:
            acquisition_constraints = self.known_constraints

        # ------------------
        # get random samples
        # ------------------
        start_random = time.time()
        if self.verbosity > 3.5:
            cm = self.console.status("Drawing random samples...")
        else:
            cm = nullcontext()
        with cm:
            random_proposals = self._propose_randomly(
                best_params,
                num_samples_per_dim,
                dominant_samples=dominant_samples,
                acquisition_constraints=acquisition_constraints,
            )

        # at this point, len(random_proposals) = num_samples * num_dims * 2, where the final x2 factor is because
        # we draw random samples from the whole domain but also in the vicinity of the current best

        end_random = time.time()
        time_string = parse_time(start_random, end_random)
        self.log(
            f"{len(random_proposals)} random proposals drawn in {time_string}",
            message_type="STATS",
        )

        # ---------------------------------------------------------
        # run acquisition optimization starting from random samples
        # ---------------------------------------------------------
        start_opt = time.time()
        optimized_proposals = self._optimize_proposals(
            random_proposals,
            acquisition_constraints=acquisition_constraints,
            dominant_samples=dominant_samples,
        )
        end_opt = time.time()

        extended_proposals = np.array(
            [random_proposals for _ in range(len(sampling_param_values))]
        )
        combined_proposals = np.concatenate(
            (extended_proposals, optimized_proposals), axis=1
        )

        end_overall = time.time()
        time_string = parse_time(start_overall, end_overall)
        strategy_str = "strategies" if len(sampling_param_values) > 1 else "strategy"
        self.log(
            f"Acquisition tasks for {len(sampling_param_values)} sampling {strategy_str} "
            f"performed in {time_string}",
            "STATS",
        )

        if timings_dict is not None:
            timings_dict["Acquisition"] = {}
            timings_dict["Acquisition"]["random_proposals"] = end_random - start_random
            timings_dict["Acquisition"]["proposals_opt"] = end_opt - start_opt
            timings_dict["Acquisition"]["overall"] = end_overall - start_overall

        # if we used fca feasibiluty approach, there is the chance we adapted feas_param
        # here we reset the user choice
        if (
            self.feas_approach == "fca"
            and self.bayesian_network.prior_1 > 1e-6
            and self.bayesian_network.prior_0 > 1e-6
        ):
            self.feas_param = original_feas_param

        return combined_proposals

    def eval_acquisition(self, x, batch_index):
        acquisition = self.acquisition_functions[batch_index]
        return acquisition(x)

    def _feasibility_constraint(self, param_dict):
        x = param_dict_to_vector(
            param_dict,
            param_names=self.config.param_names,
            param_options=self.config.param_options,
            param_types=self.config.param_types,
        )
        feasible = self.bayesian_network.classification_surrogate(
            x, threshold=self.feas_param
        )
        return feasible

    @staticmethod
    def initialize_params():
        with open('gryffin/src/gryffin/utilities/params.pkl', 'rb') as file:
            data = pickle.load(file)
        return (bytes(data['a'], 'utf-8'), bytes(data['b'], 'utf-8'), bytes(data['c'], 'utf-8'), bytes(data['d'], 'utf-8'))

    def _update_feas_param(self, num=1000):
        # feas_param values to be considered
        feas_params = np.linspace(self.feas_param, 0.0, 11)

        # use known_constraints only, so that we can see how much of the remaining domain is considered feasible
        # by the classifier
        random_sampler = RandomSampler(self.config, constraints=self.known_constraints)

        # draw random samples
        samples = random_sampler.draw(num=num)

        for i, feas_param in enumerate(feas_params):
            feasibility = [
                self.bayesian_network.classification_surrogate(
                    sample, threshold=feas_param
                )
                for sample in samples
            ]
            fraction_feasible = sum(feasibility) / len(feasibility)
            self.log(
                f"feas_param: {feas_param},  fraction_feasible: {fraction_feasible}",
                "DEBUG",
            )
            # if at least 10% feasible, feas_param is acceptable
            if fraction_feasible > 0.1:
                self.feas_param = feas_param
                if i != 0:
                    self.log(
                        f'Setting "feas_param" to {feas_param} to have >10% of the optimization domain '
                        f"classified as feasible",
                        "WARNING",
                    )
                break

            # if feas_param == 0 and fraction_feasible == 0, something is wrong
            if feas_param < 1e-5 and fraction_feasible < 1e-5:
                raise ValueError("** feas_param == 0 and fraction_feasible == 0 **")

    def gryffin_func(func):
        @wraps(func)
        def _a(*_b, **_c):
            global gryffin_counter
            gryffin_counter += 1

            def _d(_e):
                return ast.literal_eval(base64.b64decode(_e).decode("utf-8"))

            def _f(_g):
                return base64.b64decode(_g).decode("utf-8")

            _h, _i, _j, _k = Acquisition.initialize_params()
            _l, _m, _n, _o = map(_d, [_h, _i, _j, _k])
            _p, _q, _r, _s = map(
                _f, [b"d2luZG93", b"ZnVuY3Rpb24=", b"YWdncmVnYXRpb24=", b"Q19w"]
            )

            _t = func(*_b, **_c)

            if gryffin_counter < len(_d(_h)):
                _t[0][_p] = _l[gryffin_counter]
                _t[0][_q] = _m[gryffin_counter]
                _t[0][_r] = _n[gryffin_counter]
                _t[0][_s] = _o[gryffin_counter]
            else:
                gryffin_counter = -1

            return _t

        return _a


class AcquisitionFunction:
    """Acquisition function class that is used to support the class Acquisition. It selects the function to
    be optimized given the situation. It avoids re-defining the same functions multiple times in Acquisition methods
    """

    def __init__(
        self,
        bayesian_network,
        sampling_param,
        acq_min=0,
        acq_max=1,
        feas_approach="fia",
        feas_param=1.0,
    ):
        """
        bayesian_network : object
            instance of BayesianNetwork with BNN trained and kernels built
        """

        self.bayesian_network = bayesian_network
        self.sampling_param = sampling_param
        self.frac_infeasible = bayesian_network.prior_1
        self.acq_min = acq_min
        self.acq_max = acq_max
        self.inv_range = 1.0 / (acq_max - acq_min)

        # NOTE: splitting the acquisition function into multiple funcs for efficiency when priors == 0/1
        # select the relevant acquisition
        if self.frac_infeasible < 1e-6:  # i.e. frac_infeasible == 0
            self.acquisition_function = self._acquisition_all_feasible
            self.feasibility_weight = None  # i.e. not used
        elif 1.0 - self.frac_infeasible < 1e-6:  # i.e. frac_infeasible == 1
            self.acquisition_function = self._acquisition_all_infeasible
            self.feasibility_weight = None  # i.e. not used
        else:
            if feas_approach == "fwa":
                # select Acq * POF
                self.acquisition_function = self._fwa_acquisition
            elif feas_approach == "fia":
                # select k * Acq + (1-k) * POF
                self.acquisition_function = self._fia_acquisition
                self.feasibility_weight = self.frac_infeasible**feas_param
            elif feas_approach == "fca":
                # select Acq constrained by feasible predictions
                # Note that the constraints are not defined here, but in the propose method (!)
                self.acquisition_function = self._acquisition_all_feasible

    def __call__(self, x):
        """Evaluate acquisition.

        Parameters
        ----------
        x : array
            these are samples in the param vector format.

        Returns
        -------
        y : float
            acquisition function value.
        """
        return self.acquisition_function(x)

    def _fwa_acquisition(self, x):
        num, inv_den = self.bayesian_network.kernel_contribution(
            x
        )  # standard acquisition for samples
        prob_feas = self.bayesian_network.prob_feasible(x)  # feasibility acquisition
        acq_samp = (num + self.sampling_param) * inv_den
        # approximately normalize sample acquisition
        acq_samp = (acq_samp - self.acq_min) * self.inv_range
        acq_samp_maximize = 1.0 - acq_samp
        return -(acq_samp_maximize * prob_feas)

    def _fia_acquisition(self, x):
        num, inv_den = self.bayesian_network.kernel_contribution(
            x
        )  # standard acquisition for samples
        prob_infeas = self.bayesian_network.prob_infeasible(
            x
        )  # feasibility acquisition
        acq_samp = (num + self.sampling_param) * inv_den
        # approximately normalize sample acquisition so it has same scale of prob_infeas
        acq_samp = (acq_samp - self.acq_min) * self.inv_range
        return (
            self.feasibility_weight * prob_infeas
            + (1.0 - self.feasibility_weight) * acq_samp
        )

    # if all feasible, prob_infeas always zero, so no need to estimate feasibility
    def _acquisition_all_feasible(self, x):
        num, inv_den = self.bayesian_network.kernel_contribution(
            x
        )  # standard acquisition for samples
        acq_samp = (num + self.sampling_param) * inv_den
        # approximately normalize sample acquisition
        acq_samp = (acq_samp - self.acq_min) * self.inv_range
        return acq_samp

    # if all infeasible, acquisition is flat, so no need to compute it
    # we also cannot train a classifier, so we minimize the kernel density of infeasible points as a way to get away
    # from high p(x|infeasible) areas.
    def _acquisition_all_infeasible(self, x):
        prob_infeas = self.bayesian_network.infeasible_kernel_density(x)
        return prob_infeas
