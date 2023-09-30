#!/usr/bin/env python

try:
    from deap import base
except ImportError:
    print('package "deap" required by GeneticOptimizer not found, please install')

from .genetic_optimizer import GeneticOptimizer
