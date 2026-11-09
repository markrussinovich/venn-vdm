"""Permutation-calibrated kernel test for distribution shift between two vector populations."""

import numpy as np


def rbf_gram(vectors, bandwidth):
    """Return the Gaussian (RBF) kernel matrix of ``vectors``; raises ``ValueError`` if ``bandwidth`` is not positive."""
    if bandwidth <= 0:
        raise ValueError("Bandwidth must be positive")
    values = np.asarray(vectors, dtype=np.float64)
    norms = np.sum(values * values, axis=1)
    distances = np.maximum(norms[:, None] + norms[None] - 2 * values @ values.T, 0)
    return np.exp(-distances / (2 * bandwidth ** 2))


def population_shift(a, b, bandwidth, permutations=199, seed=42, alpha=0.05):
    """Test whether populations ``a`` and ``b`` differ with a permutation-calibrated RBF MMD statistic.

    Returns a dict with ``shifted`` (``p_value <= alpha``), ``p_value``, ``effect_size`` and ``test``.
    Raises ``ValueError`` unless both populations have at least two rows and ``permutations >= 19``.
    """
    if len(a) < 2 or len(b) < 2 or permutations < 19:
        raise ValueError("Need two nonempty populations and at least 19 permutations")
    vectors = np.concatenate([a, b])
    kernel = rbf_gram(vectors, bandwidth)
    weights = np.concatenate([np.full(len(a), 1 / len(a)), np.full(len(b), -1 / len(b))])
    observed = float(weights @ kernel @ weights)
    rng = np.random.default_rng(seed)
    null_weights = np.stack([rng.permutation(weights) for _ in range(permutations)])
    null = np.einsum("bi,ij,bj->b", null_weights, kernel, null_weights, optimize=True)
    p = float((1 + np.sum(null >= observed)) / (permutations + 1))
    return {
        "shifted": p <= alpha, "p_value": p,
        "effect_size": float(np.sqrt(max(observed, 0))),
        "test": "permutation-calibrated RBF MMD; p-value is not P(shifted)",
    }
