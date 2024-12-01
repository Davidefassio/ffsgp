# Fassio's Genetic Programming Toolbox

from .genetic_operators import create_full, create_grow
from .tree import Tree

import numpy as np
from numpy.typing import NDArray


def init_pop_full(n: np.int64, max_vars: np.int64, depth: np.int64) -> NDArray[Tree]:
    """
    Generate the initial population using the Full Method.
    """
    return np.array([create_full(max_vars, depth) for _ in range(n)])


def init_pop_grow(n: np.int64, max_vars: np.int64, depth: np.int64, length: np.int64) -> NDArray[Tree]:
    """
    Generate the initial population using the Grow Method.
    """
    return np.array([create_grow(max_vars, depth, length) for _ in range(n)])


def init_pop_half(n: np.int64, max_vars: np.int64, depth: np.int64, length: np.int64, n_jobs: int = 1) -> NDArray[Tree]:
    """
    Generate the initial population using the Half-and-Half Method.

    Half are generated using the Full Method, half with the Grow Method.
    N.B: If n is odd, then generate one more with the Full Method.
    """
    if n_jobs == 1:
        return np.concatenate((init_pop_full(n // 2 + n % 2, max_vars, depth), init_pop_grow(n // 2, max_vars, depth, length)))
    else:
        pass # Do it parallel


def init_pop_ramp(n: np.int64, max_vars: np.int64, depth: np.int64, length: np.int64) -> NDArray[Tree]:
    """
    Generate the initial population using the Ramped Half-and-Half Method.

    Randomly, with p=0.5, create a tree with the full or grow method.
    If the tree is already present in the population discard it.
    N.B: this method might be biased towards full trees.
    """
    population = []

    while len(population) < n:
        if np.random.random() < 0.5:
            tree = create_full(max_vars, depth)
        else:
            tree = create_grow(max_vars, depth, length)
        
        if tree not in population:
            population.append(tree)

    return np.array(population)
