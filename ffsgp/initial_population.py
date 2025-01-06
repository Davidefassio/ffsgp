# Fassio's Genetic Programming Toolbox
#
# MIT License
# Copyright (c) 2024 Davide Fassio

from .genetic_operators import create_full, create_grow
from .tree import Tree
from .utility_threads import get_nthreads

import numpy as np
from threading import Thread
from queue import Queue


def init_pop_full(n: np.int64, nvars: np.int64, depth: np.int64) -> list[Tree]:
    """
    Generate the initial population using the Full Method.
    """
    return [create_full(nvars, depth) for _ in range(n)]


def init_pop_grow(n: np.int64, nvars: np.int64, depth: np.int64, length: np.int64, p_term: float = 0.5) -> list[Tree]:
    """
    Generate the initial population using the Grow Method.
    """
    return [create_grow(nvars, depth, length, p_term) for _ in range(n)]


def init_pop_half(n: np.int64, nvars: np.int64, depth: np.int64, length: np.int64, p_term: float = 0.5, n_jobs: int = 1) -> list[Tree]:
    """
    Generate the initial population using the Half-and-Half Method.

    Half are generated using the Full Method, half with the Grow Method.
    Note: If n is odd, then generate one more with the Full Method.
    """
    n_jobs = get_nthreads(n_jobs)

    if n_jobs == 1:
        return init_pop_full(n // 2 + n % 2, nvars, depth) + init_pop_grow(n // 2, nvars, depth, length, p_term)
    else:  
        # In this function parallelization is at most two threads.
        result_queue = Queue()
        t1 = Thread(target=lambda rq, *args: rq.put(init_pop_full(*args)), args=(result_queue, n // 2 + n % 2, nvars, depth))
        t2 = Thread(target=lambda rq, *args: rq.put(init_pop_grow(*args)), args=(result_queue, n // 2, nvars, depth, length, p_term))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        return result_queue.get() + result_queue.get()


def init_pop_ramp(n: np.int64, nvars: np.int64, depth: np.int64, length: np.int64, p_term: float = 0.5) -> list[Tree]:
    """
    Generate the initial population using the Ramped Half-and-Half Method.

    Randomly, with p=0.5, create a tree with the full or grow method.
    If the tree is already present in the population discard it.
    Note: this method might be biased towards full trees.
    """
    population = []

    while len(population) < n:
        if np.random.random() < 0.5:
            tree = create_full(nvars, depth)
        else:
            tree = create_grow(nvars, depth, length, p_term)
        
        if tree not in population:
            population.append(tree)

    return population
