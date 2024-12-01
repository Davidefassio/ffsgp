# Fassio's Genetic Programming Toolbox

from .operators import Op, NUM_OP, IDX_TWO_ARITY
from .variables import Var
from .node import Node
from .tree import Tree

import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def crossover_subtree(parent1: Tree, parent2: Tree) -> Tuple[Tree, Tree]:
    """
    Subtree crossover.

    A subtree is selected from each parent and they are swapped.
    Two offsprings are generated, so this operator is symmetrical.
    """
    # Choose the points
    p1 = np.random.randint(0, parent1.data.size)
    p2 = np.random.randint(0, parent2.data.size)

    # Get the length of the subtrees rooted in p1 and p2
    len1 = parent1.data[p1]['length']
    len2 = parent2.data[p2]['length']

    # Generate the new offspring by swapping the subtrees
    off1 = np.concatenate((parent1.data[:p1 - len1 + 1], parent2.data[p2 - len2 + 1:p2 + 1], parent1.data[p1 + 1:]))
    off2 = np.concatenate((parent2.data[:p2 - len2 + 1], parent1.data[p1 - len1 + 1:p1 + 1], parent2.data[p2 + 1:])) 

    # Return the childs in the proper type
    return Tree(data=off1), Tree(data=off2)


def mutation_single_point(individual: Tree, max_vars: np.int64 = 0) -> None:
    """
    Single-Point Mutation.

    Selects a single node in the tree and replaces it with a node of the same type.
    """
    # Choose a point
    p = np.random.randint(0, individual.data.size)

    # Get type and modify by type
    if individual.data[p]['f'] >= 0:
        # Match the arity
        if Op.arity(individual.data[p]['f']) == 1:
            individual.data[p]['f'] = np.random.randint(0, IDX_TWO_ARITY)
        else:
            individual.data[p]['f'] = np.random.randint(IDX_TWO_ARITY, NUM_OP)
    elif individual.data[p]['name'] >= 0:
        individual.data[p]['name'] = np.random.randint(0, max_vars)
    else:
        individual.data[p]['value'] += np.random.normal()


def mutation_hoist(individual: Tree) -> None:
    """
    Hoist Mutation.

    Selects a subtree and replaces the entire individual with this subtree.
    """
    # Choose a point
    p = np.random.randint(0, individual.data.size)

    # Replace the whole genome with the subtree rooted in p
    individual.data = individual.data[p - individual.data[p]['length'] + 1:p + 1]


def mutation_subtree(individual: Tree) -> None:
    """
    Subtree mutation.

    A subtree is selected and replaced with a new one.
    The generated subtree is constructed using the grow method,
    ensuring its depth and length do not exceed the current ones.
    """
    # Choose a point
    p = np.random.randint(0, individual.data.size)

    # Get the length and depth of the subtree rooted in p
    l = individual.data[p1]['length']
    d = individual.data[p1]['depth']

    individual.data = Tree.update_stats(np.concatenate((individual.data[:p - l + 1], create_grow(d, l).data, individual.data[p + 1:])))


def _rec_create_full(max_vars: np.int64, max_depth: np.int64) -> NDArray[Node]:
    if max_depth == 1:
        # Choose from terminal set
        if np.random.random() < 0.5:
            return np.array([Node(Var(np.random.randint(0, max_vars))).numpy()])  # Variable
        else:
            return np.array([Node(np.random.normal(1, 1)).numpy()])  # Constant (biased toward positive)
    
    # Choose from operators set
    operator = np.random.randint(0, NUM_OP)
    op_to_arr = np.array([Node(Op(operator)).numpy()])
    if Op.arity(operator) == 1:
        return np.concatenate((_rec_create_full(max_vars, max_depth - 1), op_to_arr))
    else:
        return np.concatenate((_rec_create_full(max_vars, max_depth - 1), _rec_create_full(max_vars, max_depth - 1), op_to_arr))


def _rec_create_grow(max_vars: np.int64, max_depth: np.int64, max_length: np.int64) -> NDArray[Node]:
    # TODO
    pass


def create_full(max_vars: np.int64, max_depth: np.int64) -> Tree:
    """
    Generate subtree using the Full Method.
    """
    if max_depth == 0:
        raise ValueError(f"Depth must be greater than 0! Got {max_depth}")

    return Tree(data=_rec_create_full(max_vars, max_depth))


# TODO: generate subtree using grow method (give max depth + max length)
def create_grow(max_vars: np.int64, max_depth: np.int64, max_length: np.int64) -> Tree:
    pass