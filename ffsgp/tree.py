# Fassio's Genetic Programming Toolbox
#
# MIT License
# Copyright (c) 2024 Davide Fassio

from .node import Node

import numpy as np
from numpy.typing import NDArray


class Tree:
    def __init__(self, *args, data: NDArray[Node] | None = None, update: bool = True) -> None:
        # Allow to either: ...
        if data is not None:  # ... provide an array of nodes in numpy format
            self.data = data
        else:  # ... provide as args the types of nodes
            self.data = np.array([Node(a).numpy() for a in args], dtype=Node)

        if update:  # Set length and depth
            Tree.update_stats(self.data)
        
        self.fitness = None

    def __call__(self, vvars: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate the formula at the given points.

        vvars is the value of the variables in the formula.
        vvars is a 2D array with shape: (#vars, #points).
        Example for a function of 2 variable evaluated in 3 points:
            pt1  pt2  pt2
        x    0   0.5   1
        y    0   1.2  -20

        It returns a 1D array with shape: (#points,).
        """
        stack = []

        for n in self.data:
            stack.append(Node.call(n, stack, vvars))  # Note: the stack is modified by Node.call

        if isinstance(stack[0], np.ndarray):
            return stack[0]
        else:
            # If we got a single value, then convert the result to match the input shape.
            # This happens when the formula is constant, because there is no broadcasting.
            return np.full_like(vvars[0], stack[0])

    def __eq__(self, other: "Tree") -> bool:
        return np.array_equal(self.data, other.data)

    def __repr__(self) -> str:
        return f"{self.data}"

    @property
    def length(self) -> np.int64:
        return self.data[-1]['length']

    @property
    def depth(self) -> np.int64:
        return self.data[-1]['depth']

    def to_human(self) -> str:
        """
        Provide human readable, and python executable, formula.
        """
        stack = []

        for n in self.data:
            stack.append(Node.to_human(n, stack))

        return stack[0]


    @staticmethod
    def update_stats(data: NDArray[Node]) -> None:
        """
        Recompute the length and depth metrics for each node.
        """
        lstack = []
        dstack = []
        for i, n in enumerate(data):
            if n['arity'] > 0:
                data[i]['length'] = sum(lstack[-n['arity']:]) + 1
                data[i]['depth'] = max(dstack[-n['arity']:]) + 1
                del lstack[-n['arity']:]
                del dstack[-n['arity']:]
            lstack.append(n['length'])
            dstack.append(n['depth'])
