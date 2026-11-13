"""Small logged-bandit diagnostics; does not reconstruct unobserved rewards.

These estimators require correctly recorded action propensities, conditional
exchangeability given the logged state/context, and overlap between the
behavior and target policies. The three arrays alone cannot verify those
assumptions. Outputs are observational estimates, never ground truth.
"""

import numpy as np


def estimate_logged_policy(rewards, logging_probability, target_probability):
    """Return IPS, self-normalized IPS and effective sample size (ESS).

    For each logged action, ``logging_probability`` is the probability that
    the behavior policy assigned to that action, and ``target_probability``
    is the *target policy's* probability of choosing that same action. These
    arrays must cover corresponding logged events (not all counterfactual
    actions). The target is policy reward per decision under stated assumptions.

    There is no clipping, hidden baseline model, bootstrap interval or automatic
    positivity certificate. Very small propensities remain visible via weights
    and ESS so a high-variance estimate cannot be mistaken for strong evidence.
    """
    def column(value, name):
        a = np.asarray(value, dtype=np.float64)
        if a.ndim != 1 or len(a) == 0 or not np.isfinite(a).all():
            raise ValueError(f"{name} must be a nonempty finite 1-D array")
        return a

    observed = column(rewards, "rewards")
    logging = column(logging_probability, "logging_probability")
    target = column(target_probability, "target_probability")
    if len(observed) != len(logging) or len(logging) != len(target):
        raise ValueError("Inputs must have one record per logged action")
    if not (np.all((logging > 0) & (logging <= 1)) and np.all((target >= 0) & (target <= 1))):
        raise ValueError("Action propensities must be valid probabilities; logged ones must be positive")
    with np.errstate(over="ignore"):
        weights = target / logging
    if not np.isfinite(weights).all():
        raise ValueError("Nonfinite importance weights; logging support is too small")
    maximum = weights.max()
    if maximum <= 0:
        raise ValueError("No logged actions have target support; estimate is undefined")
    scaled = weights / maximum
    total = scaled.sum()
    squares = np.dot(scaled, scaled)
    with np.errstate(over="ignore", invalid="ignore"):
        ips = np.dot(scaled / len(observed), observed) * maximum
        snips = np.dot(scaled / total, observed)
    if not np.isfinite(ips) or not np.isfinite(snips):
        raise ValueError("Importance-weighted rewards overflowed")
    return {
        "n": len(observed),
        "ips": float(ips),
        "snips": float(snips),
        "effective_sample_size": float((total / np.sqrt(squares)) ** 2),
        "nonzero_weight_rows": int(np.count_nonzero(weights)),
        "max_weight": float(maximum),
        "min_logging_probability": float(logging.min()),
        "observed_reward_mean": float(np.dot(np.full(len(observed), 1 / len(observed)), observed)),
    }
