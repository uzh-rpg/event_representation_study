#!/usr/bin/env python

__author__ = "Florian Hase, Matteo Aldeghi"

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as td
import torchbnn as bnn

import numpy as np
from gryffin.utilities.decorators import processify
from gryffin.utilities import GryffinComputeError


class GeneratorNetwork(nn.Module):
    def __init__(self, num_descs):
        super(GeneratorNetwork, self).__init__()

        self.linear_layer = nn.Linear(in_features=num_descs, out_features=num_descs)

        nn.init.eye_(self.linear_layer.weight)
        nn.init.zeros_(self.linear_layer.bias)

    def forward(self, x):
        x = self.linear_layer(x)
        return F.softsign(x)


class Generator:
    def __init__(self, descs, objs, grid_descs, max_epochs=1000, learning_rate=0.001):
        self.descs = torch.tensor(descs).float()
        self.objs = torch.tensor(objs).float()
        self.grid_descs = torch.tensor(grid_descs).float()
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate

        self.num_samples = descs.shape[0]
        self.num_descs = descs.shape[1]

    def _train_generator_network(self):
        generator_network = GeneratorNetwork(self.num_descs)
        optimizer = optim.Adam(generator_network.parameters(), lr=self.learning_rate)

        for epoch in range(self.max_epochs):
            gen_descs = generator_network(self.descs)
            corr_coeffs, cov_gen_descs = self._compute_correlations(
                gen_descs, self.objs
            )
            min_corr = self._compute_min_corr()

            # compute loss for deviating from target binary matrix
            norm_corr_coeffs = F.leaky_relu(
                (torch.abs(corr_coeffs) - min_corr) / (1.0 - min_corr), 0.01
            )

            loss_0 = torch.mean(torch.sin(3.14159 * norm_corr_coeffs) ** 2)
            loss_1 = 1.0 - torch.max(torch.abs(norm_corr_coeffs))

            # compute loss for non-zero correlations in generated descriptors
            norm_cov_x = F.leaky_relu(
                (torch.abs(cov_gen_descs) - min_corr) / (1.0 - min_corr), 0.01
            )
            loss_2 = torch.sum((torch.sin(3.14159 * norm_cov_x / 2.0)) ** 2) / (
                self.num_descs**2 - self.num_descs
            )

            loss_3 = 1e-2 * torch.mean(torch.abs(generator_network.linear_layer.weight))

            loss = loss_0 + loss_1 + loss_2 + loss_3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return generator_network

    def _compute_min_corr(self):
        return 1.0 / math.sqrt(self.num_samples - 2)

    def _compute_correlations(self, descs, objs):
        gen_descs_mean = descs.mean(axis=0)
        gen_descs_var = descs.var(axis=0)

        objs_mean = objs.mean(axis=0)
        objs_var = objs.var(axis=0)

        gen_descs_var += 1e-6
        objs_var += 1e-6

        # compute correlation coefficients between descriptors and objectives
        numerator = torch.mean((objs - objs_mean) * (descs - gen_descs_mean), axis=0)
        denominator = torch.sqrt(gen_descs_var * objs_var)
        corr_coeffs = numerator / denominator

        # compute correlation coefficients among descriptors
        gen_descs_expand = (descs - gen_descs_mean).unsqueeze(-1)
        gen_descs_transpose = torch.transpose(gen_descs_expand, -2, -1)

        gen_descs_var_expand = gen_descs_var.unsqueeze(-1)
        gen_descs_var_transpose = torch.transpose(gen_descs_var_expand, -2, -1)

        cov_gen_descs = torch.mean(gen_descs_expand @ gen_descs_transpose, axis=0)
        cov_gen_descs = cov_gen_descs / torch.sqrt(
            gen_descs_var_expand @ gen_descs_var_transpose
        )

        return corr_coeffs, cov_gen_descs

    def generate_descriptors(self):
        generator_network = self._train_generator_network()

        gen_descs = generator_network(self.descs)
        corr_coeffs, cov_gen_descs = self._compute_correlations(gen_descs, self.objs)
        min_corr = self._compute_min_corr()

        auto_gen_descs = generator_network(self.grid_descs)

        results = {}
        results["auto_gen_descs"] = auto_gen_descs.cpu().detach().numpy()
        results["comp_corr_coeffs"] = corr_coeffs.cpu().detach().numpy()
        results["gen_descs_cov"] = cov_gen_descs.cpu().detach().numpy()
        results["min_corrs"] = min_corr

        results["weights"] = (
            generator_network.linear_layer.weight.detach().clone().numpy()
        )

        sufficient_desc_indices = np.where(
            np.abs(results["comp_corr_coeffs"]) > results["min_corrs"]
        )[0]
        if len(sufficient_desc_indices) == 0:
            sufficient_desc_indices = np.array([0])
        reduced_gen_descs = results["auto_gen_descs"][:, sufficient_desc_indices]
        results["reduced_gen_descs"] = reduced_gen_descs.astype(np.float64)
        results["sufficient_indices"] = sufficient_desc_indices

        return results
