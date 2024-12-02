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

    def __call__(self, vvars: list[np.float64] | None = None) -> np.float64:
        # TODO: modify vvars to allow to pass multiple data points
        stack = []

        for n in self.data:
            stack.append(Node.call(n, stack, vvars))

        return stack[0]

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
