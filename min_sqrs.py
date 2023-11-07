import numpy as np
from typing import Callable, Any

def __make_column_mat__(X: np.ndarray) -> np.ndarray:
    return X.reshape((X.size, 1))

def __evaluate_X_in_K_fs__(X: np.ndarray, K_fs: list[Callable[[Any], Any]]) -> np.ndarray:
    K_fs_len = len(K_fs)
    f_X      = np.zeros((X.shape[0], K_fs_len))

    for i in range(K_fs_len):
        vec_f = np.vectorize(pyfunc=K_fs[i])
        f_X[:, i].flat = vec_f(X[:, i])

    return f_X

def min_sqrs(X: np.ndarray, K_fs: list[Callable[[Any], Any]], Y: np.ndarray) -> np.ndarray:

    f_X   = __evaluate_X_in_K_fs__(X, K_fs)
    f_X_t = f_X.transpose()

    X_sqr   = f_X_t.dot(f_X)
    p_inv_X = np.linalg.inv(X_sqr)

    K = p_inv_X.dot(f_X.T).dot(Y)

    return K
    
def square_error(Y: np.ndarray, F: np.ndarray) -> float:
    E     = Y - F
    E     = E.T.dot(E)
    error = E.sum()
    return error 

def mean_square_error(Y: np.ndarray, F: np.ndarray) -> float:
    return square_error(Y, F) / Y.size