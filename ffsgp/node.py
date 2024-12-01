# Fassio's Genetic Programming Toolbox

from .operators import Op
from .variables import Var

import numpy as np
from numpy.typing import NDArray


class Node:
    dtype = np.dtype([
        ('f', 'i8'),
        ('name', 'i8'),
        ('value', 'f8'),
        ('arity', 'i8'),
        ('length', 'i8'),
        ('depth', 'i8'),
    ])

    def __init__(self, 
                 node: Op | Var | np.number | int | float,
                 length: np.int64 = 1,
                 depth: np.int64 = 1) -> None:

        self.f: np.int64 = np.int64(-1)  # (OP) index of the operator
        self.name: np.int64 = np.int64(-1)  # (VAR) index in the vars list
        self.value: np.float64 = np.float64(0)  # (CONST) value of the constant
        self.arity: np.int64 = np.int64(0)  # Number of operands of f

        if isinstance(node, Op):
            self.f = node.idx
            self.arity = Op.arity(self.f)
        elif isinstance(node, Var):
            self.name = np.int64(node.n)
        elif isinstance(node, (np.number, int, float)):
            self.value = np.float64(node)
        else:
            raise TypeError(f"Unknown type! Input: {node}).")

        self.length: np.int64 = length  # Total number of nodes in the subtree rooted here
        self.depth: np.int64 = depth  # Height of the subtree rooted here

    @staticmethod
    def call(this, stack: list[np.float64], vvars: list[np.float64] | None) -> np.float64:
        if this['f'] >= 0:
            retval = Op.call(this['f'], *stack[-this['arity']:])
            del stack[-this['arity']:]
            return retval
        elif this['name'] >= 0:
            return vvars[this['name']]
        else:
            return this['value']

    def numpy(self) -> NDArray["Node"]:
        return np.array((self.f, self.name, self.value, self.arity, self.length, self.depth), dtype=Node.dtype)