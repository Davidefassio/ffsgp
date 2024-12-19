# Fassio's Genetic Programming Toolbox
#
# MIT License
# Copyright (c) 2024 Davide Fassio

from .operators import Op, NUM_OP, IDX_TWO_ARITY
from .variables import Var
from .node import Node
from .tree import Tree

import numpy as np
from numpy.typing import NDArray


def crossover_subtree(parent1: Tree, parent2: Tree) -> tuple[Tree, Tree]:
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


def mutation_single_point(individual: Tree, nvars: np.int64) -> None:
    """
    Single-Point Mutation.

    Selects a single node in the tree and replaces it with a node of the same type.
    """
    # Choose a point
    p = np.random.randint(0, individual.data.size)

    # Get type and modify by type
    if individual.data[p]['f'] >= 0:
        # Match the arity
        if individual.data[p]['arity'] == 1:
            individual.data[p]['f'] = np.random.randint(0, IDX_TWO_ARITY)
        else:
            individual.data[p]['f'] = np.random.randint(IDX_TWO_ARITY, NUM_OP)
    elif individual.data[p]['name'] >= 0:
        individual.data[p]['name'] = np.random.randint(0, nvars)
    else:
        individual.data[p]['value'] += np.random.normal()


def mutation_constant(individual: Tree) -> None:
    """
    Mutation of constant values.

    Selects a node of type constant in the tree (if aviable) and mutate its value.
    """
    # List all nodes of type constant    
    idxs = []
    for i, node in enumerate(individual.data):
        if node['f'] < 0 and node['name'] < 0:
            idxs.append(i)
    
    if not idxs:  # No constants to modify
        return

    # Choose one and mutate its value
    individual.data[np.random.choice(idxs)]['value'] += np.random.normal()


def mutation_hoist(individual: Tree) -> None:
    """
    Hoist Mutation.

    Selects a subtree and replaces the entire individual with this subtree.
    """
    # Choose a point
    p = np.random.randint(0, individual.data.size)

    # Replace the whole genome with the subtree rooted in p
    individual.data = individual.data[p - individual.data[p]['length'] + 1:p + 1]


def mutation_subtree(individual: Tree, nvars: np.int64, p_term: float = 0.5) -> None:
    """
    Subtree mutation.

    A subtree is selected and replaced with a new one.
    The generated subtree is constructed using the grow method,
    ensuring its depth and length do not exceed the current ones.
    """
    # Choose a point
    p = np.random.randint(0, individual.data.size)

    # Get the length and depth of the subtree rooted in p
    l = individual.data[p]['length']
    d = individual.data[p]['depth']

    individual.data = np.concatenate((individual.data[:p - l + 1], create_grow(nvars, d, l, p_term).data, individual.data[p + 1:]))
    Tree.update_stats(individual.data)


def _rec_create_full(nvars: np.int64, max_depth: np.int64) -> NDArray[Node]:
    """Inner recursion function"""
    if max_depth <= 1:
        # Choose from terminal set
        if np.random.random() < 0.5:  # Variable
            return np.array([Node(Var(np.random.randint(0, nvars))).numpy()])
        else:  # Constant
            # Biased toward positive: N(1,1)
            return np.array([Node(np.random.normal(1, 1)).numpy()])

    # Choose from operators set
    operator = np.random.randint(0, NUM_OP)
    op_to_arr = np.array([Node(Op(operator)).numpy()])
    if Op.arity(operator) == 1:
        return np.concatenate((_rec_create_full(nvars, max_depth - 1), op_to_arr))
    else:
        return np.concatenate((_rec_create_full(nvars, max_depth - 1), _rec_create_full(nvars, max_depth - 1), op_to_arr))


def _rec_create_grow(nvars: np.int64, max_depth: np.int64, max_length: np.int64, p_term: float = 0.5) -> NDArray[Node]:
    """Inner recursion function"""
    if max_depth <= 1 or max_length <= 1 or np.random.random() < p_term:
        # Choose from terminal set
        if np.random.random() < 0.5:  # Variable
            return np.array([Node(Var(np.random.randint(0, nvars))).numpy()])
        else:  # Constant
            # Biased toward positive: N(1,1)
            return np.array([Node(np.random.normal(1, 1)).numpy()])
    else:  # Operator
        # If max_length == 2 constraint to 1-arity operators
        operator = np.random.randint(0, NUM_OP if max_length > 2 else IDX_TWO_ARITY)
        op_to_arr = np.array([Node(Op(operator)).numpy()])

        if Op.arity(operator) == 1:
            return np.concatenate((_rec_create_grow(nvars, max_depth - 1, max_length - 1, p_term), op_to_arr))
        else:
            # Generate left and right subtrees.
            # The left must leave space for at least a terminal node in the right one, hence max_length-2.
            # The right is also constrained by the already generated left one.
            subtree_left = _rec_create_grow(nvars, max_depth - 1, max_length - 2, p_term)
            subtree_right = _rec_create_grow(nvars, max_depth - 1, max_length - subtree_left.size - 1, p_term)

            # Randomize order to reduce bias, otherwise the left subtree
            # (the first one to be generated) is statistically bigger.
            if np.random.random() < 0.5:
                return np.concatenate((subtree_left, subtree_right, op_to_arr))
            else:
                return np.concatenate((subtree_right, subtree_left, op_to_arr))


def create_full(nvars: np.int64, max_depth: np.int64) -> Tree:
    """
    Generate a subtree using the Full Method.
    """
    if max_depth == 0:
        raise ValueError(f"Depth must be greater than 0! Got {max_depth}")

    return Tree(data=_rec_create_full(nvars, max_depth))


def create_grow(nvars: np.int64, max_depth: np.int64, max_length: np.int64, p_term: float = 0.5) -> Tree:
    """
    Generate a subtree using the Grow Method.

    p_term: the probability to select the node from the terminal set
    """
    if max_depth == 0:
        raise ValueError(f"Depth must be greater than 0! Got {max_depth}")

    if max_length == 0:
        raise ValueError(f"Length must be greater than 0! Got {max_length}")

    return Tree(data=_rec_create_grow(nvars, max_depth, max_length, p_term))
