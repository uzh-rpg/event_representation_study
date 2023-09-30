#!/usr/bin/env python

from .benchmark_functions_cont import (
    dejong as dejong,
    hyper_ellipsoid as hyperellipsoid,
    rosenbrock_function as rosenbrock,
    rastrigin_function as rastrigin,
    schwefel_function as schwefel,
    ackley_path_function as ackley,
    linear_funnel,
    narrow_funnel,
    discrete_ackley,
    discrete_michalewicz,
    double_well,
    discrete_valleys,
)

from .benchmark_functions_cat import (
    Dejong as CatDejong,
    Ackley as CatAckley,
    Camel as CatCamel,
    Dejong as CatDejong,
    Michalewicz as CatMichalewicz,
    Slope as CatSlope,
    RandomCorrelated as CatRandomCorrelated,
    RandomUncorrelated as CatRandomUncorrelated,
)
