import numpy as np
from scipy.spatial.distance import cdist

def magnitude_at_t(X: np.ndarray, t: float, reg: float = 1e-6) -> float:
    """
    Compute the magnitude |tA| for a set of embeddings X at scale t.

    Parameters
    ----------
    X   : (n, d) embedding matrix
    t   : scale parameter
    reg : small regularization for stability

    """
    D = cdist(X, X, metric='euclidean')  # Pairwise distances
    Z = np.exp(-t * D)
    Z += reg * np.eye(Z.shape[0])

    # Solve Z w = 1
    w = np.linalg.solve(Z, np.ones(Z.shape[0]))
    return float(w.sum())

