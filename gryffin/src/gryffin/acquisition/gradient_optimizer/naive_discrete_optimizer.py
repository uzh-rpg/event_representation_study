#!/usr/bin/env python

__author__ = "Florian Hase"


import numpy as np


class NaiveDiscreteOptimizer:
    def __init__(self, func=None):
        self.func = func

    def _set_func(self, func, pos=None):
        self.func = func
        if pos is not None:
            self.pos = pos
            self.num_pos = len(pos)

    def set_func(self, func, pos=None, highest=None):
        self.highest = highest
        self._set_func(func, pos)

    def get_update(self, vector):
        func_best = self.func(vector)
        for pos_index, pos in enumerate(self.pos):
            if pos is None:
                continue

            current = vector[pos]
            perturb = np.random.choice(self.highest[pos_index])
            vector[pos] = perturb

            func_cand = self.func(vector)
            if func_cand < func_best:
                func_best = func_cand
            else:
                vector[pos] = current
        return vector
