"""Analytic and hostile controls for logged-action importance sampling."""

import numpy as np
import pytest

from venn.offpolicy import estimate_logged_policy


def test_hand_computed_deterministic_target_policy():
    # Four actions were selected with behavior propensity 1/2.
    # Target policy selects only the two successful logged actions.
    out = estimate_logged_policy([1, 0, 1, 0], [.5] * 4, [1, 0, 1, 0])
    assert out["n"] == 4
    assert out["ips"] == 1.
    assert out["snips"] == 1.
    assert out["effective_sample_size"] == pytest.approx(2.)
    assert out["nonzero_weight_rows"] == 2
    assert out["max_weight"] == 2.
    assert out["observed_reward_mean"] == .5


def test_identical_policies_recover_observed_mean_and_full_ess():
    rng = np.random.default_rng(73)
    y = rng.normal(size=50)
    p = rng.uniform(.01, 1., size=50)
    out = estimate_logged_policy(y, p, p)
    assert out["ips"] == pytest.approx(y.mean())
    assert out["snips"] == pytest.approx(y.mean())
    assert out["effective_sample_size"] == pytest.approx(50.)
    assert out["nonzero_weight_rows"] == 50


def test_reward_scale_is_linear_but_ess_does_not_change():
    values = [-2., .5, 1., 0.]
    logging = [.3, .4, .6, .2]
    target = [.7, .1, .3, .5]
    a = estimate_logged_policy(values, logging, target)
    b = estimate_logged_policy([3 * v for v in values], logging, target)
    assert b["ips"] == pytest.approx(3 * a["ips"])
    assert b["snips"] == pytest.approx(3 * a["snips"])
    assert b["effective_sample_size"] == a["effective_sample_size"]


def test_one_extreme_weight_exposes_low_effective_sample_size():
    r = estimate_logged_policy([1.] * 10, [1e-7] + [1.] * 9, [1.] * 10)
    assert r["max_weight"] == 1e7
    assert r["effective_sample_size"] < 1.001
    assert r["snips"] == pytest.approx(1.)
    assert r["ips"] > 1e5  # Large variance risk, not evidence of high policy reward.


@pytest.mark.parametrize("args", [
    ([], [], []),
    ([1, 2], [.5], [.5]),
    ([1], [0.], [1.]),
    ([1], [1.2], [1.]),
    ([1], [.5], [1.2]),
    ([1], [.5], [-.1]),
    ([1], [.5], [0.]),
    ([float("nan")], [.5], [.5]),
    ([[1]], [.5], [.5]),
])
def test_invalid_or_unidentifiable_inputs_rejected(args):
    with pytest.raises(ValueError):
        estimate_logged_policy(*args)


def test_randomized_same_policy_identity():
    rng = np.random.default_rng(2)
    for n in range(1, 32):
        rewards = rng.normal(size=n)
        p = rng.uniform(.01, .99, size=n)
        report = estimate_logged_policy(rewards, p, p)
        assert report["snips"] == pytest.approx(rewards.mean(), rel=1e-12, abs=1e-12)
        assert report["ips"] == pytest.approx(rewards.mean(), rel=1e-12, abs=1e-12)


def test_ess_remains_finite_when_summed_weights_square_would_overflow():
    # 1000 weights of 1e152: total**2 overflows, sum(weights**2) does not.
    # A stable implementation must still report ESS ~= 1000.
    r = estimate_logged_policy(np.ones(1000), np.full(1000, 1e-152), np.ones(1000))
    assert r["effective_sample_size"] == pytest.approx(1000., rel=1e-12)
    assert r["snips"] == pytest.approx(1.)


@pytest.mark.parametrize("target,logging,expected_ips", [
    (1e-200, .5, 4e-200),
    (1., 1e-200, 2e200),
])
def test_normalized_estimators_survive_weight_square_underflow_and_overflow(target, logging, expected_ips):
    report = estimate_logged_policy([1., 2., 3.], [logging] * 3, [target] * 3)
    assert report["effective_sample_size"] == pytest.approx(3.)
    assert report["snips"] == pytest.approx(2.)
    assert report["ips"] == pytest.approx(expected_ips, rel=1e-12, abs=0)


def test_finite_ips_is_not_rejected_when_unnormalized_weighted_sum_overflows():
    report = estimate_logged_policy([1.] * 1000, [1e-306] * 1000, [1.] * 1000)
    assert report["ips"] == pytest.approx(1e306)
    assert report["snips"] == pytest.approx(1.)
    assert report["effective_sample_size"] == pytest.approx(1000.)
