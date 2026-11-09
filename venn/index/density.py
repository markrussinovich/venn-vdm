"""k-nearest-neighbor outlier scores."""

import numpy as np


def knn_outlier_scores(vectors, k=5):
    """Return each row's squared distance to its ``k``-th nearest other row; larger means more isolated.

    Raises ``ValueError`` unless ``vectors`` is a finite 2-D array and ``1 <= k < len(vectors)``.
    """
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all() or not isinstance(k, int) or not 1 <= k < len(values):
        raise ValueError("Provide finite vectors and 1 <= k < row count")
    norms = np.sum(values * values, axis=1)
    distances = np.maximum(0, norms[:, None] + norms[None] - 2 * values @ values.T)
    np.fill_diagonal(distances, np.inf)
    return np.partition(distances, k - 1, axis=1)[:, k - 1]
