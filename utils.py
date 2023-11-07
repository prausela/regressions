import numpy as np

def add_bias(X: np.ndarray) -> np.ndarray:
    b_X = np.ones((X.shape[0], X.shape[1] + 1))
    b_X[:, :-1] = X
    return b_X