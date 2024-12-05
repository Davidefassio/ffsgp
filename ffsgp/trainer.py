# Fassio's Genetic Programming Toolbox
#
# MIT License
# Copyright (c) 2024 Davide Fassio

from .tree import Tree

import numpy as np
from numpy.typing import NDArray
from collections.abc import Callable
import os
from threading import Thread
from queue import Queue


class Trainer:
    def __init__(self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        initial_population: list[Tree],
        offspring_mult: float,
        n_generations: int,
        fitness: Callable[[NDArray[np.float64], NDArray[np.float64]], np.float64],
        parent_selection: Callable[[list[Tree]], Tree],
        crossover: Callable[[Tree, Tree],tuple[Tree, Tree]],
        mutations: list[Callable[[Tree], None]],
        probs_mutation: list[float],
        elitism: float = 0,
        verbose: bool = False,
        n_jobs: int = 1) -> None:

        # Verify that parameters are reasonable
        assert offspring_mult >= 1, f"offspring_mult must be >= 1 (generational approach)! Got {offspring_mult}"
        assert n_generations > 0, f"n_generations must be positive! Got {n_generations}"
        assert len(mutations) == len(probs_mutation), "Mutations must be same length!"
        assert sum(probs_mutation) <= 1, f"Mutation probabilities must sum to <= 1! Got {self.probs_mutation} (sum: {sum(self.probs_mutation)})"
        assert 0 <= elitism <= 1, f"Elitism must be a probability, in [0, 1]! Got {elitism}"

        self.x = x  # Training input data
        self.y = y  # Training output data

        self.curr_population = initial_population
        self.population_size = len(self.curr_population)
        self.offspring_size = int(self.population_size * offspring_mult)
        self.n_generations = n_generations

        self.fitness = fitness  # To maximize

        self.parent_selection = parent_selection  # Given a list of individual choose one to be a parent

        self.crossover = crossover  # Crossover function
        self.mutations = mutations  # list of mutation functions
        self.probs_mutation = probs_mutation

        self.elitism = elitism  # percentage of population to carry over to next generation
        
        self.verbose = verbose
        
        # Parallelization
        # Compute n_jobs following scikit-learn conventions
        total_cores = os.cpu_count()
        if total_cores is None:
            raise RuntimeError("Unable to determine the number of CPU cores.")
        
        if n_jobs > 0:
            self.n_jobs = n_jobs if n_jobs <= total_cores else total_cores
        elif n_jobs < 0:
            # Calculate cores to use based on the negative value
            threads = total_cores + n_jobs + 1
            if threads > 0:
                self.n_jobs = threads
            else:
                raise ValueError(f"n_jobs={n_jobs} results in a non-positive number of threads ({threads}).")
        else:
            # Raise an error for invalid n_jobs=0 or other unexpected values
            raise ValueError(f"Invalid value for n_jobs: {n_jobs}. Must be a different from 0.")

        print(self.n_jobs)
        self.offspring_per_thread = int(self.offspring_size / self.n_jobs)
        self.new_offspring_queue = Queue()

    def vprint(self, string: str) -> None:
        """Print if verbose is True."""
        if self.verbose:
            print(string)

    def evaluate_tree(self, t: Tree) -> None:
        """
        Compute the fitness of a tree.
        Store also its length to prefer smaller trees at the same fitness.
        """
        f = self.fitness(self.y, t(self.x))
        t.fitness = (f if not np.isnan(f) else -np.inf, -t.length)

    def variationAnd(self) -> None:
        """
        Given a population ALWAYS apply crossover AND, with probability probs_mutation,
        apply a mutation to generate a new offspring population.

        THREAD SAFETY:
          - the original population is never modified
          - the results are stored in a thread-safe queue
        """
        offspring = []
        while len(offspring) < self.offspring_per_thread:
            # Parent selection + crossover
            parent1, parent2 = self.crossover(self.parent_selection(self.curr_population), self.parent_selection(self.curr_population))
            
            # Mutation
            m_values = np.random.random(size=2)
            m_cum = 0
            for func, prob in zip(self.mutations, self.probs_mutation):
                m_cum += prob
                if m_values[0] < m_cum:
                    func(parent1)
                    m_values[0] = 2  # A value greater than the cumulative
                if m_values[1] < m_cum:
                    func(parent2)
                    m_values[1] = 2  # A value greater than the cumulative
            
            # Fitness evaluation
            self.evaluate_tree(parent1)
            self.evaluate_tree(parent2)
            offspring.append(parent1)
            offspring.append(parent2)

        self.new_offspring_queue.put(offspring)

    # TODO: survivor selection + elitism (top 1%)
    # TODO: limit bloat (hard limit?)
    #
    # Return best individual
    def train(self) -> Tree:
        # Evaluate the initial population
        for t in self.curr_population:
            self.evaluate_tree(t)
        
        for gen in range(self.n_generations):
            # Generate offspring
            offspring = []
            threads = [Thread(target=self.variationAnd) for _ in range(self.n_jobs)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            for _ in range(self.n_jobs):
                offspring.extend(self.new_offspring_queue.get())
            offspring = sorted(offspring, key=lambda t: t.fitness, reverse=True)

            # Survivor selection
            self.curr_population = offspring[:self.population_size]

            # Print info
            self.vprint(f"Gen {gen}. Best: {self.curr_population[0].to_human()}, Fitness: {self.curr_population[0].fitness}")

        return self.curr_population[0]
