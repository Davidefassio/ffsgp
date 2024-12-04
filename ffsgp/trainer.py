# Fassio's Genetic Programming Toolbox
#
# MIT License
# Copyright (c) 2024 Davide Fassio

from .tree import Tree

import numpy as np
from numpy.typing import NDArray
from threading import Thread
from queue import Queue

class Trainer:
    def __init__(self,
        initial_population: NDArray[Tree],
        offspring_mult: float,
        n_generations: int,
        fitness: Callable[[NDArray[np.float64], NDArray[np.float64]], np.float64],
        maximize_fitness: bool,
        parent_selection: Callable[..., Tree],
        parent_selection_kwargs: dict,
        generator: Callable[[],],
        crossovers: list[Callable[[],]],
        mutations: list[Callable[[],]],
        probs_crossover: list[float],
        probs_mutation: list[float],
        elitism: float,
        verbose: bool = False,
        n_jobs: int = 1):

        self.curr_population = initial_population
        self.offspring_mult = offspring_mult
        self.population_size = self.curr_population.size
        self.offspring_size = int(self.population_size * self.offspring_mult)
        self.n_generations = n_generations

        self.fitness = fitness
        self.maximize_fitness = maximize_fitness

        self.parent_selection = parent_selection
        self.parent_selection_kwargs = parent_selection_kwargs

        self.generator = generator
        self.crossovers = crossovers
        self.mutations = mutations
        self.prob_crossover = prob_crossover
        self.prob_mutation = prob_mutation
        
        self.elitism = elitism
        
        self.verbose = verbose

        self.n_jobs = n_jobs

    def vprint(self, string: str) -> None:
        """Print if verbose is True."""
        if self.verbose:
            print(string)

    # TODO: take inspiration from Operon for the evolutionary workflow
    #
    # initial_pop -> parents_pop => offspring_gen => offspring_pop
    #                      ^-------------------------------v
    # 
    # To generate new offspring select two parents, do crossover, then with probability
    # p_m do a mutation, then evaluate fitness.
    #
    # TODO: survivor selection + elitism (top 1%)
    def train():
        pass
