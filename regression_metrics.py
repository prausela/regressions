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

def total_sum_of_squares(Y: np.ndarray) -> float:
    mean_mtx = np.ones(Y.shape) * Y.mean() 
    sqr_diff_mean = square_error(Y, mean_mtx)
    return sqr_diff_mean

def r_squared(Y: np.ndarray, F: np.ndarray) -> float:
    rss = square_error(Y, F)
    tss = total_sum_of_squares(Y)

    return 1 - (rss / tss)

def adj_r_squared(R_2: float, Y: np.ndarray, K: np.ndarray) -> float:
    n = Y.size
    p = K.size - 1
    return 1 - (((1 - R_2)*(n - 1)) / (n - p))

def f_stat(Y: np.ndarray, K: np.ndarray, F: np.ndarray) -> float:
    n = Y.size
    p = K.size - 1
    rss = square_error(Y, F)
    tss = total_sum_of_squares(Y)
    return ((tss - rss) / p) / (rss / (n - p - 1))

def sigma_hat_sqr(Y: np.ndarray, K: np.ndarray, F: np.ndarray) -> float:
    n = Y.size
    p = K.size - 1
    return (1 / (n - p - 1))*square_error(Y, F)
    
def coefficients_variance_matrix(X: np.ndarray, s_hat_sqr: float) -> np.ndarray:
    X_sqr = X.T.dot(X)
    X_sqr_inv = np.linalg.inv(X_sqr)
    return s_hat_sqr * X_sqr_inv

def coefficients_variance(X: np.ndarray, s_hat_sqr: float) -> np.ndarray:
    var_K_mat = coefficients_variance(X, s_hat_sqr)
    var_K = var_K_mat.diagonal()
    return var_K