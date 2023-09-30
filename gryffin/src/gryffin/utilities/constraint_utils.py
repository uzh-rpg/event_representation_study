#!/usr/bin/env python

import numpy as np
import itertools


def compute_constrained_cartesian(known_constraints, config):
    """For parameter spaces with all categorical parameters, compute the
    cartesian product space with constraints applied
    """
    options = []
    for param in config.parameters:
        options.append(np.arange(len(param["specifics"]["options"])))
    # compute cartesian product space
    options = np.array(list(itertools.product(*options)))
    # apply constraint(s)
    if known_constraints is not None:
        constrained_options = []
        for option in options:
            # map to param dict
            p_dict = {
                p["name"]: p["specifics"]["options"][int(o_ix)]
                for p, o_ix in zip(config.parameters, option)
            }
            is_valid = known_constraints(p_dict)
            if is_valid:
                constrained_options.append(option)
            else:
                pass
    else:
        constrained_options = options

    return list(constrained_options)


def estimate_feas_fraction(known_constraints, config, resolution=100):
    """Produces an estimate of the fraction of the domain which
    is feasible. For continuous valued parameters, we build a grid with
    "resolution" number of points in each dimensions. We measure each of
    possible categorical options


    Args:
        known_constraints (callable): callable function which retuns the
            feasibility mask
        config (): gryffin config
        resolution (int): the number of points to query per continuous dimension
    """
    samples = []
    for param_ix, param in enumerate(config.parameters):
        if param["type"] == "continuous":
            sample = np.linspace(
                param["specifics"]["low"], param["specifics"]["high"], resolution
            )
        elif param["type"] == "discrete":
            # num_options = int((param['specifics']['low']-param['specifics']['high'])/param['specific']['stride']+1)
            # sample = np.linspace(param['specifics']['low'], param['specifics']['high'], num_options)
            sample = param["specifics"]["options"]
        elif param["type"] == "categorical":
            sample = param["specifics"]["options"]
        else:
            quit()
        samples.append(sample)
    # make meshgrid
    meshgrid = np.stack(np.meshgrid(*samples), len(samples))
    num_samples = np.prod(np.shape(meshgrid)[:-1])
    # reshape into 2d array
    samples = np.reshape(meshgrid, newshape=(num_samples, len(samples)))

    samples_dict = []
    for sample in samples:
        s = {
            f"{name}": element for element, name in zip(sample, config.parameters.name)
        }
        samples_dict.append(s)

    num_feas = 0.0
    num_total = len(samples_dict)
    for sample in samples_dict:
        if known_constraints(sample):
            num_feas += 1
        else:
            pass
    frac_feas = num_feas / num_total
    assert 0.0 <= frac_feas <= 1.0

    return frac_feas
