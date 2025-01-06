# Fassio's Genetic Programming Toolbox
#
# MIT License
# Copyright (c) 2024 Davide Fassio

from .operators import add, mul
from .tree import Tree
from .utility_threads import get_nthreads

import numpy as np
from numpy.typing import NDArray
from collections.abc import Callable
from threading import Thread
from queue import Queue


class Trainer:
    def __init__(self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        normalize: bool,
        initial_population: list[Tree],
        offspring_mult: float,
        n_generations: int,
        fitness: Callable[[NDArray[np.float64], NDArray[np.float64]], np.float64],
        parent_selection: Callable[[list[Tree]], Tree],
        crossover: Callable[[Tree, Tree],tuple[Tree, Tree]],
        mutations: list[Callable[[Tree], None]],
        probs_mutation: list[float],
        max_depth: int | None = None,
        max_length: int | None = None,
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

        # Save the statistics for the normalization step
        self.y_mean = self.y.mean()
        self.y_std = self.y.std()
        
        # If requested, normalize the y to have mean 0 and standard deviation 1
        self.normalize = normalize
        if self.normalize:
            self.y = (self.y - self.y_mean) / self.y_std

        self.curr_population = initial_population
        self.population_size = len(self.curr_population)
        self.offspring_size = int(self.population_size * offspring_mult)
        self.n_generations = n_generations

        self.fitness = fitness  # To maximize

        self.parent_selection = parent_selection  # Given a list of individual choose one to be a parent

        self.crossover = crossover  # Crossover function
        self.mutations = mutations  # List of mutation functions
        self.probs_mutation = probs_mutation  # Probability to apply a mutation 
        
        # Set limits, if None set them to infinite
        self.max_depth = max_depth if max_depth is not None else np.inf
        self.max_length = max_length if max_length is not None else np.inf

        self.elitism = int(np.round(elitism * self.population_size))  # Convert the percentage to the actual number

        self.verbose = verbose
        
        # Parallelization
        self.n_jobs = get_nthreads(n_jobs)
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

            # Evaluate fitness and add to offspring if inside limits
            if parent1.depth <= self.max_depth and parent1.length <= self.max_length:
                self.evaluate_tree(parent1)
                offspring.append(parent1)

            if parent2.depth <= self.max_depth and parent2.length <= self.max_length:
                self.evaluate_tree(parent2)
                offspring.append(parent2)

        self.new_offspring_queue.put(offspring)

    def train(self) -> Tree:
        """
        Evolve a population to fit the training dataset.
        Return the best individual ever found.
        """
        # Evaluate the initial population and sort it
        for t in self.curr_population:
            self.evaluate_tree(t)
        self.curr_population = sorted(self.curr_population, key=lambda t: t.fitness, reverse=True)
        best_individual = self.curr_population[0]

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

            # Elitism: meritocracy baby!
            offspring.extend(self.curr_population[:self.elitism])

            # Survivor selection
            offspring = sorted(offspring, key=lambda t: t.fitness, reverse=True)
            self.curr_population = offspring[:self.population_size]

            # Hall of fame
            if self.curr_population[0].fitness > best_individual.fitness:
                best_individual = self.curr_population[0]

            # Print info
            self.vprint(f"Gen {gen}. Best: {self.curr_population[0].to_human()}, Fitness: {self.curr_population[0].fitness}")
        
        if self.normalize:
            # Add the denormalization step:
            # np.add(np.mul("formula", stddev), mean)
            best_individual.data = np.concat((best_individual.data, Tree(self.y_std, mul, self.y_mean, add, update=False).data))
            Tree.update_stats(best_individual.data)
            
        return best_individual
