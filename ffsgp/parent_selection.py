# Fassio's Genetic Programming Toolbox
#
# MIT License
# Copyright (c) 2024 Davide Fassio

from .tree import Tree

import numpy as np


def tournament(population: list[Tree], n: int = 2, p_fh: float = 0.0) -> Tree:
    """
    Tournament selection with fitness hole to reduce bloat.

    n: size of the tournament. Default: 2
    p_fh: probability to choose the smallest and not the fittest. Default: 0.0

    Returns the winner of the tournament.
    """
    knights = np.random.choice(population, size=n)
    if np.random.random() < p_fh:
        return min(knights, key=lambda t: t.length)
    else:
        return max(knights, key=lambda t: t.fitness)
