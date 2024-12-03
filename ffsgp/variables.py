# Fassio's Genetic Programming Toolbox
#
# MIT License
# Copyright (c) 2024 Davide Fassio

import numpy as np


class Var:
    def __init__(self, n: np.number | int | float) -> None:
        self.n = n
