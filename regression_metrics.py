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

def r_squared(Y: np.ndarray, F: np.ndarray) -> float:
    sqr_err = square_error(Y, F)

    mean_mtx = np.ones(Y.shape) * Y.mean() 
    sqr_diff_mean = square_error(Y, mean_mtx)

    return 1 - (sqr_err / sqr_diff_mean)

def adj_r_squared(R_2: float, Y: np.ndarray, K: np.ndarray) -> float:
    n = Y.size
    p = K.size
    return 1 - (((1 - R_2)*(n - 1)) / (n - p))

def sigma_hat_sqr(Y: np.ndarray, K: np.ndarray, F: np.ndarray) -> float:
    n = Y.size
    p = K.size
    return (1 / (n - p -1))*square_error(Y, F)
    
def coefficients_variance(X: np.ndarray, sigma_hat_sqr: float) -> np.ndarray:
    X_sqr = X.T.dot(X)
    X_sqr_inv = np.linalg.inv(X_sqr)
    return sigma_hat_sqr * X_sqr_inv