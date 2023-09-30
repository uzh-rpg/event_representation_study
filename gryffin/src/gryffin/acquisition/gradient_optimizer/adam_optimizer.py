#!/usr/bin/env python

__author__ = "Florian Hase, Matteo Aldeghi"

import numpy as np


class AdamOptimizer:
    def __init__(
        self,
        func=None,
        select=None,
        eta=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        decay=False,
    ):
        """
        Adam optimizer: https://arxiv.org/abs/1412.6980.

        Parameters
        ----------
        func : callable
            function to be optimized.
        select : list
            list of bools selecting the dimensions to be optimized.
        eta : float
            ste size. This is alpha in the Adam paper.
        beta1 : float
            exponential decay rate for the first moment estimates.
        beta2 : float
            exponential decay rate for the second-moment estimates.
        epsilon : float
            small number to prevent any division by zero in the implementation.
        decay : bool
            whether to use rate decay. 1/sqrt(t) decay is used as per the Adam paper.
        """
        self.func = func
        self.eta = eta
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.decay = decay
        self.iterations = 0

        # init Adam parameters if pos is provided
        if select is not None:
            self.init_params(select)
        else:
            self.select_bool = None
            self.select_idx = None
            self.num_dims = None
            self.ms = None
            self.vs = None

        # step used to estimate gradients numerically
        self.dx = 1e-6

    def init_params(self, select):
        """
        select : list
            list of bools selecting the dimensions to be optimized.
        """
        self.select_bool = np.array(select)
        self.num_dims = len(self.select_bool)
        self.select_idx = np.arange(self.num_dims)[self.select_bool]
        self.ms = np.zeros(
            self.num_dims
        )  # moment vector (length is size of input vector, i.e. opt domain)
        self.vs = np.zeros(self.num_dims)  # exponentially weighted infinity norm

    def reset(self):
        self.iterations = 0
        self.ms = np.zeros(self.num_dims)
        self.vs = np.zeros(self.num_dims)

    def set_func(self, func, select=None):
        """
        func : callable
            function to be optimized.
        select : list
            list of bools selecting the dimensions to be optimized.
        """
        self.func = func
        self.reset()
        if select is not None:
            self.init_params(select)

    def grad(self, sample):
        """
        Estimate the gradients.
        Note that Adam is invariant to diagonal rescaling of the gradients.
        """
        gradients = np.zeros(len(sample), dtype=np.float32)
        perturb = np.zeros(len(sample), dtype=np.float32)

        for i in self.select_idx:
            perturb[i] += self.dx
            gradient = (self.func(sample + perturb) - self.func(sample - perturb)) / (
                2.0 * self.dx
            )
            gradients[i] = gradient
            perturb[i] -= self.dx

        return gradients

    def get_update(self, sample):
        """Update sample according to Adam method.

        Parameters
        ----------
        sample : nd.array
            starting position for the sample.

        Returns
        -------
        sample : nd.array
            updated sample position.
        """

        # get gradients: g
        grads = self.grad(sample)
        # get iteration: t
        self.iterations += 1

        if self.decay is True:
            eta = self.eta / np.sqrt(self.iterations)
        else:
            eta = self.eta

        # eta(t) = eta * sqrt(1 – beta2(t)) / (1 – beta1(t))
        # where: beta(t) = beta^t
        eta_next = eta * (
            np.sqrt(1.0 - np.power(self.beta_2, self.iterations))
            / (1.0 - np.power(self.beta_1, self.iterations))
        )
        # m(t) = beta1 * m(t-1) + (1 – beta1) * g(t)
        ms_next = (self.beta_1 * self.ms) + (1.0 - self.beta_1) * grads
        # v(t) = beta2 * v(t-1) + (1 – beta2) * g(t)^2
        vs_next = (self.beta_2 * self.vs) + (1.0 - self.beta_2) * np.square(grads)

        # update sample: x(t) = x(t-1) – eta(t) * m(t) / (sqrt(v(t)) + eps)
        sample_next = sample - eta_next * ms_next / (np.sqrt(vs_next) + self.epsilon)

        # update params
        self.ms = ms_next
        self.vs = vs_next

        return sample_next


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    adam = AdamOptimizer()

    def func(x):
        return (x - 1) ** 2

    adam.set_func(func, select=[True])

    domain = np.linspace(-1, 3, 200)
    values = func(domain)

    start = np.zeros(1) - 0.8

    plt.ion()

    for _ in range(10**3):
        plt.clf()
        plt.plot(domain, values)
        plt.plot(start, func(start), marker="o", color="k")

        start = adam.get_update(start)

        plt.pause(0.05)
