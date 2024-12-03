# Fassio's Genetic Programming Toolbox
#
# MIT License
# Copyright (c) 2024 Davide Fassio

import numpy as np
from numpy.typing import NDArray


def mae(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> np.float64:
    """Mean Absolute Error"""
    return np.abs(np.subtract(y_true, y_pred)).mean()


def mse(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> np.float64:
    """Mean Squared Error"""
    return np.square(np.subtract(y_true, y_pred)).mean()


def rmse(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> np.float64:
    """Root Mean Squared Error"""
    return np.sqrt(mse(y_true, y_pred))


def nmse(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> np.float64:
    """Normalized Mean Squared Error"""
    return mse(y_true, y_pred) / np.var(y_true)


def r2(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> np.float64:
    """R^2 (coefficient of determination) score, with constant data adjustment."""
    if np.all(y_true == y_true[0]):  # y_true is constant
        if np.all(y_pred == y_true[0]):  # Perfect prediction
            return np.float64(1)
        else:  # Imperfect prediction
            return np.float64(0)
    
    return 1 - (np.sum(np.square(np.subtract(y_true, y_pred))) / np.sum(np.square(np.subtract(y_true, np.mean(y_true)))))
