#!/usr/bin/env python

__author__ = "Florian Hase, Matteo Aldeghi"


import numpy as np
from gryffin.utilities import Logger
from gryffin.utilities import GryffinUnknownSettingsError, GryffinComputeError
from gryffin.observation_processor import param_vector_to_dict


class RandomSampler(Logger):
    def __init__(self, config, constraints=None):
        """
        known_constraints : list of callable
            List of constraint functions. Each is a function that takes a parameter dict, e.g.
            {'x0':0.1, 'x1':10, 'x2':'A'} and returns a bool indicating
            whether it is in the feasible region or not.
        """

        # register attributes
        self.config = config
        self.reject_tol = self.config.get("reject_tol")

        # if constraints not None, and not a list, put into a list
        if constraints is not None and isinstance(constraints, list) is False:
            self.constraints = [constraints]
        else:
            self.constraints = constraints

        # set verbosity
        verbosity = self.config.get("verbosity")
        Logger.__init__(self, "RandomSampler", verbosity)

    def draw(self, num=1):
        # if no constraints, we do not need to do any "rejection sampling"
        if self.constraints is None:
            samples = self._fast_draw(num=num)
        else:
            samples = self._slow_draw(num=num)
        return samples

    def perturb(self, ref_sample, num=1, scale=0.05):
        """Take ref_sample and perturb it num times"""
        # if no constraints, we do not need to do any "rejection sampling"
        if self.constraints is None:
            perturbed_samples = self._fast_perturb(ref_sample, num=num, scale=scale)
        else:
            perturbed_samples = self._slow_perturb(ref_sample, num=num, scale=scale)
        return perturbed_samples

    def _fast_draw(self, num=1):
        samples = []
        for param_index, param_settings in enumerate(self.config.parameters):
            param_type = param_settings["type"]
            specs = param_settings["specifics"]
            param_samples = self._draw_single_parameter(
                num=num, param_type=param_type, specs=specs
            )
            samples.append(param_samples)
        samples = np.concatenate(samples, axis=1)
        return samples

    def _slow_draw(self, num=1):
        samples = []
        counter = 0

        # keep trying random samples until we get num samples
        while len(samples) < num:
            sample = []  # we store the random sample used by Gryffin here

            # iterate over each variable and draw at random
            for param_index, param_settings in enumerate(self.config.parameters):
                specs = param_settings["specifics"]
                param_type = param_settings["type"]
                param_sample = self._draw_single_parameter(
                    num=1, param_type=param_type, specs=specs
                )[0]
                sample.append(param_sample[0])

            # evaluate whether the sample violates the known constraints
            param = param_vector_to_dict(
                param_vector=sample,
                param_names=self.config.param_names,
                param_options=self.config.param_options,
                param_types=self.config.param_types,
            )

            feasible = [constr(param) for constr in self.constraints]
            if all(feasible) is True:
                samples.append(sample)

            counter += 1
            if counter % num == 0:
                self.log(f"drawn {counter} random samples", "DEBUG")
            if counter > self.reject_tol * num:
                p = 100.0 / self.reject_tol
                raise GryffinComputeError(
                    f"the feasible region seems to be less than {p}% of the optimization "
                    f"domain. Consider redefining the problem or increasing 'reject_tol'."
                )

        samples = np.array(samples)
        return samples

    def _draw_single_parameter(self, num, param_type, specs):
        if param_type == "continuous":
            sampled_values = self._draw_continuous(
                low=specs["low"], high=specs["high"], size=(num, 1)
            )
        elif param_type == "categorical":
            sampled_values = self._draw_categorical(
                num_options=len(specs["options"]), size=(num, 1)
            )
        elif param_type == "discrete":
            sampled_values = self._draw_discrete(
                low=specs["low"], high=specs["high"], size=(num, 1)
            )
        else:
            GryffinUnknownSettingsError(
                f'cannot understand parameter type "{param_type}"'
            )
        return sampled_values

    def _fast_perturb(self, ref_sample, num=1, scale=0.05):
        """Perturbs a reference sample by adding random uniform noise around it"""
        perturbed_samples = []
        for param_index, param_settings in enumerate(self.config.parameters):
            param_type = param_settings["type"]
            specs = param_settings["specifics"]
            ref_value = ref_sample[param_index]
            perturbed_param_samples = self._perturb_single_parameter(
                ref_value=ref_value,
                num=num,
                param_type=param_type,
                specs=specs,
                scale=scale,
            )
            perturbed_samples.append(perturbed_param_samples)

        perturbed_samples = np.concatenate(perturbed_samples, axis=1)
        return perturbed_samples

    def _slow_perturb(self, ref_sample, num=1, scale=0.05):
        perturbed_samples = []
        counter = 0
        new_scale = scale
        perturb_categorical = (
            False  # start perturb categories if we cannot find feasible perturbations
        )

        # keep trying random samples until we get num samples
        while len(perturbed_samples) < num:
            perturbed_sample = []  # we store the samples here

            # iterate over each variable and perturb ref_sample
            for param_index, param_settings in enumerate(self.config.parameters):
                specs = param_settings["specifics"]
                param_type = param_settings["type"]
                ref_value = ref_sample[param_index]
                perturbed_param = self._perturb_single_parameter(
                    ref_value=ref_value,
                    num=1,
                    param_type=param_type,
                    specs=specs,
                    scale=new_scale,
                    perturb_categorical=perturb_categorical,
                )[0]
                perturbed_sample.append(perturbed_param[0])

            # evaluate whether the sample violates the known constraints
            param = param_vector_to_dict(
                param_vector=perturbed_sample,
                param_names=self.config.param_names,
                param_options=self.config.param_options,
                param_types=self.config.param_types,
            )

            feasible = [constr(param) for constr in self.constraints]
            if all(feasible) is True:
                perturbed_samples.append(perturbed_sample)

            counter += 1
            if counter > 100 * num:
                # double scale if counter > 100, triple if >200, etc.
                new_scale = ((counter // 100) + 1) * scale
                perturb_categorical = True
            if counter % num == 0:
                self.log(f"randomly perturbed {counter} times", "DEBUG")
            if counter > self.reject_tol * num:
                # be forgiving here: if we cannot find enough feasible perturbations, just return what we have
                self.log(
                    f"we cannot find enough feasible solutions to perturbations of the incumbent. "
                    f"Only {len(perturbed_samples)} perturbed samples have been identified. This may "
                    "indicate a problem with either the setup or the code.",
                    "WARNING",
                )
                break

        perturbed_samples = np.array(perturbed_samples)
        return perturbed_samples

    def _perturb_single_parameter(
        self, ref_value, num, param_type, specs, scale, perturb_categorical=False
    ):
        if param_type in ["continuous", "discrete"]:
            # draw uniform within unit range
            sampled_values = self._draw_continuous(-scale, scale, (num, 1))
            # scale to actual range
            sampled_values *= specs["high"] - specs["low"]
            # if discrete, we round to nearest integer
            if param_type == "discrete":
                sampled_values = np.around(sampled_values, decimals=0)
            # add +/- 5% perturbation to sample
            perturbed_sample = ref_value + sampled_values
            # make sure we do not cross optimization boundaries
            perturbed_sample = np.where(
                perturbed_sample < specs["low"], specs["low"], perturbed_sample
            )
            perturbed_sample = np.where(
                perturbed_sample > specs["high"], specs["high"], perturbed_sample
            )
        elif param_type == "categorical":
            # i.e. do not perturb
            if perturb_categorical is False:
                perturbed_sample = ref_value * np.ones((num, 1)).astype(np.float32)
            # i.e. random draw
            else:
                perturbed_sample = self._draw_categorical(
                    num_options=len(specs["options"]), size=(num, 1)
                )
        else:
            GryffinUnknownSettingsError("did not understand settings")
        return perturbed_sample

    @staticmethod
    def _draw_categorical(num_options, size):
        if size[0] > num_options:
            replace = True
        else:
            replace = False
        return np.random.choice(num_options, size=size, replace=replace).astype(
            np.float32
        )

    @staticmethod
    def _draw_continuous(low, high, size):
        return np.random.uniform(low=low, high=high, size=size).astype(np.float32)

    @staticmethod
    def _draw_discrete(low, high, size):
        return np.random.randint(low=0, high=high - low + 1, size=size).astype(
            np.float32
        )
