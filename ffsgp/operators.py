# Fassio's Genetic Programming Toolbox
#
# MIT License
# Copyright (c) 2024 Davide Fassio

import numpy as np
from numpy.typing import NDArray

# Note: we implemented log-abs instead of log because it is just a domain extension.
class Op:
    def __init__(self, idx: int):
        self.idx = np.int64(idx)

    @staticmethod
    def arity(idx: np.int64) -> np.int64:
        return np.int64(1 if idx < IDX_TWO_ARITY else 2)

    @staticmethod
    def call(idx: np.int64, *args) -> NDArray[np.float64]:
        assert Op.arity(idx) == len(args), f"Wrong number of arguments! Expecting {Op.arity(idx)}, got {len(args)} {args}."
        match idx:
            case 0:
                return np.log(np.abs(args[0]))
            case 1:
                return np.sin(args[0])
            case 2:
                return np.cos(args[0])
            case 3:
                return np.abs(args[0])
            case 4:
                return np.add(args[0], args[1])
            case 5:
                return np.subtract(args[0], args[1])
            case 6:
                return np.multiply(args[0], args[1])
            case 7:
                return np.divide(args[0], args[1])
            case 8:
                return np.pow(args[0], args[1])
            case _:
                raise f"Unknown function {idx}!"

    @staticmethod
    def repr(idx: np.int64) -> str:
        match idx:
            case 0:
                return "<op 'log'>"
            case 1:
                return "<op 'sin'>"
            case 2:
                return "<op 'cos'>"
            case 3:
                return "<op 'abs'>"
            case 4:
                return "<op 'add'>"
            case 5:
                return "<op 'sub'>"
            case 6:
                return "<op 'mul'>"
            case 7:
                return "<op 'div'>"
            case 8:
                return "<op 'pow'>"
            case _:
                raise f"Unknown function {idx}!"

    @staticmethod
    def to_human(idx: np.int64) -> tuple[str, list[int]]:
        match idx:
            case 0:
                return "np.log(np.abs())", [-2]
            case 1:
                return "np.sin()", [-1]
            case 2:
                return "np.cos()", [-1]
            case 3:
                return "np.abs()", [-1]
            case 4:
                return "np.add(,)", [-2, -1]
            case 5:
                return "np.subtract(,)", [-2, -1]
            case 6:
                return "np.multiply(,)", [-2, -1]
            case 7:
                return "np.divide(,)", [-2, -1]
            case 8:
                return "np.pow(,)", [-2, -1]
            case _:
                raise f"Unknown function {idx}!"


# Define operators, ordered by arity
log = Op(0)
sin = Op(1)
cos = Op(2)
abs = Op(3)
add = Op(4)
sub = Op(5)
mul = Op(6)
div = Op(7)
pow = Op(8)


IDX_TWO_ARITY = 4  # Index of the first operator with arity == 2
NUM_OP = 9  # Number of operators defined
