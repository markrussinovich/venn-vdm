"""Permutation-calibrated kernel test for distribution shift between two vector populations."""

import numpy as np


def rbf_gram(vectors, bandwidth):
    """Return the RBF kernel; require a finite positive bandwidth and a finite float64 kernel."""
    if isinstance(bandwidth, (bool, np.bool_)) or not isinstance(
            bandwidth, (int, float, np.integer, np.floating)):
        raise ValueError("Bandwidth must be a finite positive number")
    try:
        bandwidth = float(bandwidth)
    except OverflowError as exc:
        raise ValueError("Bandwidth must be a finite positive number") from exc
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Bandwidth must be a finite positive number")
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Vectors must be a matrix")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        norms = np.sum(values * values, axis=1)
        distances = np.maximum(norms[:, None] + norms[None] - 2 * values @ values.T, 0)
        # Divide twice so squaring a valid bandwidth cannot underflow or overflow.
        kernel = np.exp(-0.5 * (distances / bandwidth) / bandwidth)
    if not np.isfinite(kernel).all():
        raise ValueError("RBF kernel must be finite; vector distances exceed float64 range")
    return kernel


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
