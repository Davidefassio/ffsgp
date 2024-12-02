# Fassio's Genetic Programming Toolbox

from .node import Node

import numpy as np
from numpy.typing import NDArray


class Tree:
    def __init__(self, *args, data: NDArray[Node] | None = None, update: bool = True) -> None:
        if data is not None:
            self.data = data
        else:
            self.data = np.array([Node(a).numpy() for a in args], dtype=Node)

        if update:  # Set length and depth
            Tree.update_stats(self.data)

    def __call__(self, vvars: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
        """
        Evaluate the formula at the given point(s).

        vvars is the value of the variables in the formula.
        There are 3 cases:
        * None: there are no variables in the formula only constants
        * vvars 1D array => shape (#vars,): one value for each variable
        * vvars 2D array => shape (#vars,#points): one row for each variable and one column for each point
        
        Note: ALWAYS use the vvars 2D interface for multiple points.
        It is really fast thanks to NumPy's vectorized computation.
        """
        stack = []

        for n in self.data:
            stack.append(Node.call(n, stack, vvars))

        if (vvars is None) or (vvars.ndim == 1) or (isinstance(stack[0], np.ndarray)):
            return stack[0]
        else:
            # If we return a single value when we evaluated on multiple points
            # convert the result to match the input shape.
            # This happens when the formula is constant, because there is no broadcasting.
            return np.full_like(vvars[0], stack[0])

    def __eq__(self, other: "Tree") -> bool:
        return np.array_equal(self.data, other.data)

    def __repr__(self) -> str:
        return f"{self.data}"

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
