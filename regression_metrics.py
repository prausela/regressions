import numpy as np
import math

def error(Y: np.ndarray, F: np.ndarray) -> np.ndarray:
    return Y - F

def absolute_error(Y: np.ndarray, F: np.ndarray) -> float:
    E = error(Y, F)
    E = abs(E)
    err = E.sum()
    return err

def square_error(Y: np.ndarray, F: np.ndarray) -> float:
    E = error(Y, F)
    E = E.T.dot(E)
    err = E.sum()
    return err

def mean_square_error(Y: np.ndarray, F: np.ndarray) -> float:
    return square_error(Y, F) / Y.size

def absolute_mean_error(Y: np.ndarray, F: np.ndarray) -> float:
    return absolute_error(Y, F) / Y.size

def residual_standard_error(Y: np.ndarray, F: np.ndarray) -> float:
    n = Y.size
    mse = mean_square_error(Y, F)
    return math.sqrt(mse / (n - 2))
    