# Fassio's Genetic Programming Toolbox
#
# MIT License
# Copyright (c) 2024 Davide Fassio

from .tree import Tree

import numpy as np
from numpy.typing import NDArray
from collections.abc import Callable
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
        hypermodern: bool,
        crossovers: list[Callable[[Tree, Tree],tuple[Tree, Tree]]],
        mutations: list[Callable[[Tree], None]],
        probs_crossover: list[float],
        probs_mutation: list[float],
        elitism: float = 0,
        verbose: bool = False,
        n_jobs: int = 1) -> None:

        # Verify that parameters are reasonable
        assert offspring_mult >= 1, f"offspring_mult must be >= 1 (generational approach)! Got {offspring_mult}"
        assert n_generations > 0, f"n_generations must be positive! Got {n_generations}"
        assert len(crossovers) == len(probs_crossover), "Crossovers must be same length!"
        assert len(mutations) == len(probs_mutation), "Mutations must be same length!"
        assert sum(probs_crossover) <= 1, f"Crossover probabilities must sum to <= 1! Got {self.probs_crossover} (sum: {sum(self.probs_crossover)})"
        assert sum(probs_mutation) <= 1, f"Mutation probabilities must sum to <= 1! Got {self.probs_mutation} (sum: {sum(self.probs_mutation)})"
        assert 0 <= elitism <= 1, f"Elitism must be a probability, in [0, 1]! Got {elitism}"
        assert n_jobs > 0, f"n_jobs must be positive! Got {n_jobs}"

        # Training data
        self.x = x
        self.y = y

        self.curr_population = initial_population
        self.population_size = len(self.curr_population)
        self.offspring_size = int(self.population_size * offspring_mult / n_jobs)
        self.n_generations = n_generations

        self.fitness = fitness  # To maximize

        self.parent_selection = parent_selection  # Given a list of individual choose one to be a parent

        self.generator = self.variationOr if hypermodern else self.variationAnd
        self.crossovers = crossovers  # list of crossover functions
        self.mutations = mutations  # list of mutation functions
        self.probs_crossover = probs_crossover
        self.probs_mutation = probs_mutation

        self.elitism = elitism  # percentage of population to carry over to next generation
        
        self.verbose = verbose

        self.n_jobs = n_jobs
        self.new_offspring_queue = Queue()

    def vprint(self, string: str) -> None:
        """Print if verbose is True."""
        if self.verbose:
            print(string)

    def evaluate_tree(self, t: Tree) -> None:
        f = self.fitness(self.y, t(self.x))
        t.fitness = (f if not np.isnan(f) else -np.inf, -t.length)

    # TODO: fix
    #   se non faccio il xover muto un solo parent
    #   se non faccio il xover devo copiare il parent
    #   se faccio il xover probabilità diversa di fare mutation ciascuno
    #   se non faccio nulla passo il parent alla generazione successiva
    def variationAnd(self) -> None:
        """Given a population apply xover AND mutation with prob p_xo p_mut"""
        offspring = []
        while len(offspring) < self.offspring_size:
            parent1 = self.parent_selection(self.curr_population)
            parent2 = self.parent_selection(self.curr_population)

            xo_value = np.random.random()
            xo_cum = 0
            for func, prob in zip(self.crossovers, self.probs_crossover):
                xo_cum += prob
                if xo_value < xo_cum:
                    parent1, parent2 = func(parent1, parent2)
                    break
            
            m_value = np.random.random()
            m_cum = 0
            for func, prob in zip(self.mutations, self.probs_mutation):
                m_cum += prob
                if m_value < m_cum:
                    func(parent1)
                    func(parent2)
                    break
            
            self.evaluate_tree(parent1)
            self.evaluate_tree(parent2)
            offspring.append(parent1)
            offspring.append(parent2)

        self.new_offspring_queue.put(offspring)

    # TODO: fix
    #   permettere alle sum prob di essere <= 1
    #   se non faccio nulla passo il parent alla generazione successiva
    def variationOr(self) -> None:
        """Given a population apply xover or mutation with prob p_xo p_mut"""
        assert sum(self.probs_crossover + self.probs_mutation) == 1, f"In 'varOr' probabilities must sum to 1! Got {self.probs_crossover} and {self.probs_mutation} (sum: {sum(self.probs_crossover + self.probs_mutation)})"

        offspring = []
        while len(offspring) < self.offspring_size:
            no_xo = True

            random_value = np.random.random()
            cum = 0
            for func, prob in zip(self.crossovers, self.probs_crossover):
                cum += prob
                if random_value < cum:
                    parent1 = self.parent_selection(self.curr_population)
                    parent2 = self.parent_selection(self.curr_population)
                    parent1, parent2 = func(parent1, parent2)
                    self.evaluate_tree(parent1)
                    self.evaluate_tree(parent2)
                    offspring.append(parent1)
                    offspring.append(parent2)
                    no_xo = False
                    break
            if no_xo:
                for func, prob in zip(self.mutations, self.probs_mutation):
                    cum += prob
                    if random_value < cum:
                        parent1 = self.parent_selection(self.curr_population)
                        func(parent1)
                        self.evaluate_tree(parent1)
                        offspring.append(parent1)
                        break
            
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
            threads = [Thread(target=self.generator) for _ in range(self.n_jobs)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            for _ in range(self.n_jobs):
                offspring.extend(self.new_offspring_queue.get())
            offspring = sorted(offspring, key=lambda t: t.fitness, reverse=True)

            # population selection
            self.curr_population = offspring[:self.population_size]

            # Print info
            self.vprint(f"Gen {gen}. Best: {self.curr_population[0].to_human()}, Fitness: {self.curr_population[0].fitness}")

        return self.curr_population[0]
