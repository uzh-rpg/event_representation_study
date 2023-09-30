#!/usr/bin/env python

__author__ = "Florian Hase"


import numpy as np
from gryffin.utilities import Logger
from gryffin.observation_processor import param_vector_to_dict
from rich.progress import track
from . import AdamOptimizer, NaiveDiscreteOptimizer, NaiveCategoricalOptimizer


class GradientOptimizer(Logger):
    def __init__(self, config, constraints=None):
        """
        constraints : list or None
            List of callables that are constraints functions. Each function takes a parameter dict, e.g.
            {'x0':0.1, 'x1':10, 'x2':'A'} and returns a bool indicating
            whether it is in the feasible region or not.
        """
        self.config = config
        Logger.__init__(
            self, "GradientOptimizer", verbosity=self.config.get("verbosity")
        )

        # if constraints not None, and not a list, put into a list
        if constraints is not None and isinstance(constraints, list) is False:
            self.constraints = [constraints]
        else:
            self.constraints = constraints

        # define which single-step optimization function to use
        if constraints is None:
            self._optimize_one_sample = self._optimize_sample
        else:
            self._optimize_one_sample = self._constrained_optimize_sample

        # parse positions
        self.pos_continuous = np.array(
            [True if f == "continuous" else False for f in self.config.feature_types]
        )
        self.pos_categories = np.array(
            [True if f == "categorical" else False for f in self.config.feature_types]
        )
        self.pos_discrete = np.array(
            [True if f == "discrete" else False for f in self.config.feature_types]
        )
        # quick/simple check
        assert (
            sum(self.pos_continuous) + sum(self.pos_categories) + sum(self.pos_discrete)
            == self.config.num_features
        )

        # instantiate optimizers for all variable types
        self.opt_con = AdamOptimizer()
        self.opt_dis = NaiveDiscreteOptimizer()
        self.opt_cat = NaiveCategoricalOptimizer()

    def _within_bounds(self, sample):
        return not (
            np.any(sample < self.config.param_lowers)
            or np.any(sample > self.config.param_uppers)
        )

    def _project_sample_onto_bounds(self, sample):
        # project sample onto opt boundaries
        if not self._within_bounds(sample):
            sample = np.where(
                sample < self.config.param_lowers, self.config.param_lowers, sample
            )
            sample = np.where(
                sample > self.config.param_uppers, self.config.param_uppers, sample
            )
            sample = sample.astype(np.float32)
        return sample

    def _optimize_continuous(self, sample):
        proposal = self.opt_con.get_update(sample)
        if self._within_bounds(proposal):
            return proposal
        else:
            return sample

    def _optimize_discrete(self, sample):
        proposal = self.opt_dis.get_update(sample)
        return proposal

    def _optimize_categorical(self, sample):
        proposal = self.opt_cat.get_update(sample)
        return proposal

    def set_func(self, kernel, ignores=None):
        pos_continuous = self.pos_continuous.copy()
        pos_discrete = self.pos_discrete.copy()
        pos_categories = self.pos_categories.copy()
        if ignores is not None:
            for ignore_index, ignore in enumerate(ignores):
                if ignore:
                    pos_continuous[ignore_index] = False
                    pos_discrete[ignore_index] = False
                    pos_categories[ignore_index] = False

        self.opt_con.set_func(kernel, select=pos_continuous)
        self.opt_dis.set_func(
            kernel,
            pos=np.arange(self.config.num_features)[pos_discrete],
            highest=self.config.feature_sizes[self.pos_discrete],
        )
        self.opt_cat.set_func(
            kernel, select=pos_categories, feature_sizes=self.config.feature_sizes
        )

    def optimize(self, samples, max_iter=10, show_progress=False):
        """Optimise a list of samples

        Parameters
        ----------
        samples :
        max_iter : int
            maximum number of steps in the optimization.
        show_progress : bool
            whether to display the optimization progress. Default is False.
        """

        optimized = []

        if show_progress is True:
            # run loop with progress bar
            iterable = track(
                enumerate(samples),
                total=len(samples),
                description="Optimizing proposals...",
                transient=False,
            )
        else:
            # run loop without progress bar
            iterable = enumerate(samples)

        for sample_index, sample in iterable:
            self.opt_con.reset()  # reset Adam optimizer for each sample
            opt = self._optimize_one_sample(sample, max_iter=max_iter)
            optimized.append(opt)

        optimized = np.array(optimized)
        return optimized

    def _single_opt_iteration(self, optimized):
        # one step of continuous
        if np.any(self.pos_continuous):
            optimized = self._optimize_continuous(optimized)

        # one step of categorical perturbation
        if np.any(self.pos_categories):
            optimized = self._optimize_categorical(optimized)

        # one step of discrete optimization
        if np.any(self.pos_discrete):
            optimized = self._optimize_discrete(optimized)

        return optimized

    def _optimize_sample(self, sample, max_iter=10, convergence_dx=1e-7):
        # copy sample
        sample_copy = sample.copy()
        optimized = sample.copy()
        # optimize
        for num_iter in range(max_iter):
            # one step of optimization
            optimized = self._single_opt_iteration(optimized)
            # make sure we're still within the domain
            optimized = self._project_sample_onto_bounds(optimized)
            # check for convergence
            if (
                np.any(self.pos_continuous)
                and np.linalg.norm(sample_copy - optimized) < convergence_dx
            ):
                break
            else:
                sample_copy = optimized.copy()
        return optimized

    def _constrained_optimize_sample(self, sample, max_iter=10, convergence_dx=1e-7):
        # use copy to create a new object, otherwise we have mutable np arrays that keep getting updated
        prev_optimized = sample.copy()
        optimized = sample.copy()

        # --------
        # optimize
        # --------
        for num_iter in range(max_iter):
            # one step of optimization
            optimized = self._single_opt_iteration(optimized)
            # make sure we're still within the domain
            optimized = self._project_sample_onto_bounds(optimized)

            # evaluate whether the optimized sample violates the known constraints
            param = param_vector_to_dict(
                param_vector=optimized,
                param_names=self.config.param_names,
                param_options=self.config.param_options,
                param_types=self.config.param_types,
            )
            feasible = [constr(param) for constr in self.constraints]
            if not all(feasible):
                # stop optimization and return last feasible point
                optimized = prev_optimized.copy()
                break

            # check for convergence
            if (
                np.any(self.pos_continuous)
                and np.linalg.norm(prev_optimized - optimized) < convergence_dx
            ):
                break
            else:
                prev_optimized = optimized.copy()
        return optimized
