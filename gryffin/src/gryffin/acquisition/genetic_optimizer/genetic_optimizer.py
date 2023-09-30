#!/usr/bin/env python

__author__ = "Matteo Aldeghi"


import numpy as np
from gryffin.utilities import Logger
from gryffin.observation_processor import param_vector_to_dict
from deap import base, creator, tools
from rich.progress import track


class GeneticOptimizer(Logger):
    def __init__(self, config, constraints=None):
        """
        constraints : list or None
            List of callables that are constraints functions. Each function takes a parameter dict, e.g.
            {'x0':0.1, 'x1':10, 'x2':'A'} and returns a bool indicating
            whether it is in the feasible region or not.
        """
        self.config = config
        self.verbosity = self.config.get("verbosity")
        Logger.__init__(self, "GeneticOptimizer", verbosity=self.verbosity)

        # if constraints not None, and not a list, put into a list
        if constraints is not None and isinstance(constraints, list) is False:
            self.constraints = [constraints]
        else:
            self.constraints = constraints

        # define which single-step optimization function to use
        if self.constraints is None:
            self._one_step_evolution = self._evolution
        else:
            self._one_step_evolution = self._constrained_evolution

        # range of opt domain dimensions
        self.param_ranges = self.config.param_uppers - self.config.param_lowers

    def acquisition(self, x):
        return (self._acquisition(x),)

    def set_func(self, acquisition, ignores=None):
        self._acquisition = acquisition

        if any(ignores) is True:
            raise NotImplementedError(
                "GeneticOptimizer with process constraints has not been implemented yet. "
                'Please choose "adam" as the "acquisition_optimizer".'
            )

    def optimize(self, samples, max_iter=10, show_progress=False):
        """
        show_progress : bool
            whether to display the optimization progress. Default is False.
        """

        # print generations if verbosity set to DEBUG
        if self.verbosity > 3.5:
            verbose = True
        else:
            verbose = False

        # crossover and mutation probabilites
        CXPB = 0.5
        MUTPB = 0.4

        if self.acquisition is None:
            self.log("cannot optimize without a function being defined", "ERROR")
            return None

        # setup GA with DEAP
        creator.create(
            "FitnessMin", base.Fitness, weights=[-1.0]
        )  # we minimize the acquisition
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # ------------
        # make toolbox
        # ------------
        toolbox = base.Toolbox()
        toolbox.register("population", param_vectors_to_deap_population)
        toolbox.register("evaluate", self.acquisition)
        # use custom mutations for continuous, discrete, and categorical variables
        toolbox.register("mutate", self._custom_mutation, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # mating type depends on how many genes we have
        if np.shape(samples)[1] == 1:
            toolbox.register("mate", cxDummy)  # i.e. no crossover
        elif np.shape(samples)[1] == 2:
            toolbox.register("mate", tools.cxUniform, indpb=0.5)  # uniform crossover
        else:
            toolbox.register("mate", tools.cxTwoPoint)  # two-point crossover

        # ---------------------
        # Initialise population
        # ---------------------
        population = toolbox.population(samples)

        # Evaluate pop fitnesses
        fitnesses = list(map(toolbox.evaluate, np.array(population)))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # create hall of fame
        num_elites = int(round(0.05 * len(population), 0))  # 5% of elite individuals
        halloffame = tools.HallOfFame(num_elites)  # hall of fame with top individuals
        halloffame.update(population)

        # register some statistics and create logbook
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])
        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(population), **record)
        if verbose is True:
            split_stream = logbook.stream.split("\n")
            self.log(split_stream[0], "DEBUG")
            self.log(split_stream[1], "DEBUG")

        # ------------------------------
        # Begin the generational process
        # ------------------------------
        if show_progress is True:
            # run loop with progress bar
            iterable = track(
                range(1, max_iter + 1),
                total=max_iter,
                description="Optimizing proposals...",
                transient=False,
            )
        else:
            # run loop without progress bar
            iterable = range(1, max_iter + 1)

        for gen in iterable:
            offspring = self._one_step_evolution(
                population=population,
                toolbox=toolbox,
                halloffame=halloffame,
                cxpb=CXPB,
                mutpb=MUTPB,
            )

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, np.array(invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # add the best back to population
            offspring.extend(halloffame.items)

            # Update the hall of fame with the generated individuals
            halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose is True:
                self.log(logbook.stream, "DEBUG")

            # convergence criterion, if the population has very similar fitness, stop
            # we quit if the population is found in the hypercube with edge 10% of the optimization domain
            if self._converged(population, slack=0.1) is True:
                break

        # DEAP cleanup
        del creator.FitnessMin
        del creator.Individual

        return np.array(population)

    def _converged(self, population, slack=0.1):
        """If all individuals within specified subvolume, the population is not very diverse"""
        pop_ranges = np.max(population, axis=0) - np.min(
            population, axis=0
        )  # range of values in population
        normalized_ranges = pop_ranges / self.param_ranges  # normalised ranges
        bool_array = normalized_ranges < slack
        return all(bool_array)

    @staticmethod
    def _evolution(population, toolbox, halloffame, cxpb=0.5, mutpb=0.3):
        # size of hall of fame
        hof_size = len(halloffame.items) if halloffame.items else 0

        # Select the next generation individuals (allow for elitism)
        offspring = toolbox.select(population, len(population) - hof_size)

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        return offspring

    def _constrained_evolution(
        self, population, toolbox, halloffame, cxpb=0.5, mutpb=0.3
    ):
        # size of hall of fame
        hof_size = len(halloffame.items) if halloffame.items else 0

        # Select the next generation individuals (allow for elitism)
        offspring = toolbox.select(population, len(population) - hof_size)

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < cxpb:
                parent1 = list(
                    map(toolbox.clone, child1)
                )  # both are parents to both children, but we select one here
                parent2 = list(map(toolbox.clone, child2))
                # mate
                toolbox.mate(child1, child2)
                # apply constraints
                self._apply_feasibility_constraint(child1, parent1)
                self._apply_feasibility_constraint(child2, parent2)
                # clear fitness values
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.random() < mutpb:
                parent = list(map(toolbox.clone, mutant))
                # mutate
                toolbox.mutate(mutant)
                # apply constraints
                self._apply_feasibility_constraint(mutant, parent)
                # clear fitness values
                del mutant.fitness.values
        return offspring

    def _evaluate_feasibility(self, param_vector):
        # evaluate whether the optimized sample violates the known constraints
        param = param_vector_to_dict(
            param_vector=param_vector,
            param_names=self.config.param_names,
            param_options=self.config.param_options,
            param_types=self.config.param_types,
        )
        feasible = [constr(param) for constr in self.constraints]
        return all(feasible)

    @staticmethod
    def _update_individual(ind, value_vector):
        for i, v in enumerate(value_vector):
            ind[i] = v

    def _apply_feasibility_constraint(self, child, parent):
        child_vector = np.array(
            child, dtype=object
        )  # object needed to allow strings of different lengths
        feasible = self._evaluate_feasibility(child_vector)
        # if feasible, stop, no need to project the mutant
        if feasible is True:
            return

        # If not feasible, we try project parent or child onto feasibility boundary following these rules:
        # - for continuous parameters, we do stick breaking that is like a continuous version of a binary tree search
        #   until the norm of the vector connecting parent and child is less than a chosen threshold.
        # - for discrete parameters, we do the same until the "stick" is as short as possible, i.e. the next step
        #   makes it infeasible
        # - for categorical variables, we first reset them to the parent, then after having changed continuous
        #   and discrete, we reset the child. If feasible, we keep the child's categories, if still infeasible,
        #   we keep the parent's categories.

        parent_vector = np.array(
            parent, dtype=object
        )  # object needed to allow strings of different lengths
        new_vector = child_vector

        child_continuous = child_vector[self.config.continuous_mask]
        child_discrete = child_vector[self.config.discrete_mask]
        child_categorical = child_vector[self.config.categorical_mask]

        parent_continuous = parent_vector[self.config.continuous_mask]
        parent_discrete = parent_vector[self.config.discrete_mask]
        parent_categorical = parent_vector[self.config.categorical_mask]

        # ---------------------------------------
        # (1) assign parent's categories to child
        # ---------------------------------------
        if any(self.config.categorical_mask) is True:
            new_vector[self.config.categorical_mask] = parent_categorical
            # If this fixes is, update child and return
            # This is equivalent to assigning the category to the child, and then going to step 2. Because child
            # and parent are both feasible, the procedure will converge to parent == child and will return parent
            if self._evaluate_feasibility(new_vector) is True:
                self._update_individual(child, new_vector)
                return

        # -----------------------------------------------------------------------
        # (2) follow stick breaking/tree search procedure for continuous/discrete
        # -----------------------------------------------------------------------
        if any(self.config.continuous_mask) or any(self.config.discrete_mask) is True:
            # data needed to normalize continuous values
            lowers = self.config.feature_lowers[self.config.continuous_mask]
            uppers = self.config.feature_uppers[self.config.continuous_mask]
            inv_range = 1.0 / (uppers - lowers)
            counter = 0
            while True:
                # update continuous
                new_continuous = np.mean(
                    np.array([parent_continuous, child_continuous]), axis=0
                )
                # update discrete, note that it can happen that child_discrete reverts to parent_discrete
                # add noise so that we can converge to the parent if needed
                noisy_mean = np.mean(
                    [parent_discrete, child_discrete], axis=0
                ) + np.random.uniform(low=-0.1, high=0.1, size=len(parent_discrete))
                new_discrete = np.round(noisy_mean, 0)

                new_vector[self.config.continuous_mask] = new_continuous
                new_vector[self.config.discrete_mask] = new_discrete

                # if child is now feasible, parent becomes new_vector (we expect parent to always be feasible)
                if self._evaluate_feasibility(new_vector) is True:
                    parent_continuous = new_vector[self.config.continuous_mask]
                    parent_discrete = new_vector[self.config.discrete_mask]
                # if child still infeasible, child becomes new_vector (we expect parent to be the feasible one
                else:
                    child_continuous = new_vector[self.config.continuous_mask]
                    child_discrete = new_vector[self.config.discrete_mask]

                # convergence criterion is that length of stick is less than 1% in all continuous dimensions
                # for discrete variables, parent and child should be same
                if (
                    np.sum(parent_discrete - child_discrete) < 0.1
                ):  # check all differences are zero
                    parent_continuous_norm = (parent_continuous - lowers) * inv_range
                    child_continuous_norm = (child_continuous - lowers) * inv_range
                    # check all differences are within 1% of range
                    if all(
                        np.abs(parent_continuous_norm - child_continuous_norm) < 0.01
                    ):
                        break

                counter += 1
                if (
                    counter > 150
                ):  # convergence above should be reached in 128 iterations max
                    self.log(
                        "constrained evolution procedure ran into trouble - using more iterations than "
                        "theoretically expected",
                        "ERROR",
                    )

        # last parent values are the feasible ones
        new_vector[self.config.continuous_mask] = parent_continuous
        new_vector[self.config.discrete_mask] = parent_discrete

        # ---------------------------------------------------------
        # (3) Try reset child's categories, otherwise keep parent's
        # ---------------------------------------------------------
        if any(self.config.categorical_mask) is True:
            new_vector[self.config.categorical_mask] = child_categorical
            if self._evaluate_feasibility(new_vector) is True:
                self._update_individual(child, new_vector)
                return
            else:
                # This HAS to be feasible, otherwise there is a bug
                new_vector[self.config.categorical_mask] = parent_categorical
                self._update_individual(child, new_vector)
                return
        else:
            self._update_individual(child, new_vector)
            return

    def _custom_mutation(
        self, individual, indpb=0.3, continuous_scale=0.1, discrete_scale=0.1
    ):
        """Custom mutation that can handled continuous, discrete, and categorical variables.

        Parameters
        ----------
        individual :
        indpb : float
            Independent probability for each attribute to be mutated.
        continuous_scale : float
            Scale for normally-distributed perturbation of continuous values.
        discrete_scale : float
            Scale for normally-distributed perturbation of discrete values.
        """

        assert len(individual) == len(self.config.param_types)

        for i, param in enumerate(self.config.parameters):
            param_type = param["type"]

            # determine whether we are performing a mutation
            if np.random.random() < indpb:
                if param_type == "continuous":
                    # Gaussian perturbation with scale being 0.1 of domain range
                    bound_low = self.config.feature_lowers[i]
                    bound_high = self.config.feature_uppers[i]
                    scale = (bound_high - bound_low) * continuous_scale
                    individual[i] += np.random.normal(loc=0.0, scale=scale)
                    individual[i] = _project_bounds(
                        individual[i], bound_low, bound_high
                    )
                elif param_type == "discrete":
                    # add/substract an integer by rounding Gaussian perturbation
                    # scale is 0.1 of domain range
                    bound_low = self.config.feature_lowers[i]
                    bound_high = self.config.feature_uppers[i]
                    # if we have very few discrete variables, just move +/- 1
                    if bound_high - bound_low < 10:
                        delta = np.random.choice([-1, 1])
                        individual[i] += delta
                    else:
                        scale = (bound_high - bound_low) * discrete_scale
                        delta = np.random.normal(loc=0.0, scale=scale)
                        individual[i] += np.round(delta, decimals=0)
                    individual[i] = _project_bounds(
                        individual[i], bound_low, bound_high
                    )
                elif param_type == "categorical":
                    # resample a random category
                    num_options = float(
                        self.config.feature_sizes[i]
                    )  # float so that np.arange returns doubles
                    individual[i] = np.random.choice(list(np.arange(num_options)))
                else:
                    raise ValueError()
            else:
                continue

        return (individual,)


def cxDummy(ind1, ind2):
    """Dummy crossover that does nothing. This is used when we have a single gene in the chromosomes, such that
    crossover would not change the population.
    """
    return ind1, ind2


def _project_bounds(x, x_low, x_high):
    if x < x_low:
        return x_low
    elif x > x_high:
        return x_high
    else:
        return x


def param_vectors_to_deap_population(param_vectors):
    population = []
    for param_vector in param_vectors:
        ind = creator.Individual(param_vector)
        population.append(ind)
    return population
