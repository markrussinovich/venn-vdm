"""Synthetic tests for paired-logit diagnostics; not semantic ground truth."""

import numpy as np
import pytest

from venn.index.counterfactual import compare_choice_logits


def test_reordering_options_preserves_winners_and_logit_identity():
    left = np.array([[2., 1., -5.], [.2, 3., 0.]])
    right = left[:, [1, 0, 2]]
    report = compare_choice_logits(["a", "b"], left, ["b", "a"], right)
    assert report["same_option_set"] is True
    assert report["added"] == report["removed"] == []
    assert report["compared_options"] == ["a", "b", "none"]
    assert [r["base_choice"] for r in report["rows"]] == ["a", "b"]
    assert all(not r["choice_changed"] and r["shared_scores_unchanged"] for r in report["rows"])


def test_abstention_transition_and_logit_drift_are_separate_observables():
    r = compare_choice_logits(["x", "y"], [[2., 1., -3.]],
                              ["x", "y"], [[2.1, 1., 4.]])["rows"][0]
    assert r["base_choice"] == "x" and r["variant_choice"] == "none"
    assert r["choice_changed"] and r["abstention_changed"]
    assert r["max_shared_logit_delta"] == pytest.approx(7.)
    assert r["shared_scores_unchanged"] is False


def test_replacement_option_is_reported_without_declaring_an_error():
    r = compare_choice_logits(["a", "b"], [[4., 1., 0.]],
                              ["b", "c"], [[1., 7., 0.]])
    assert r["same_option_set"] is False
    assert r["added"] == ["c"] and r["removed"] == ["a"]
    assert r["compared_options"] == ["b", "none"]
    assert r["rows"][0]["choice_changed"] is True


def test_same_choice_can_hide_logit_drift():
    r = compare_choice_logits(["a", "b"], [[5., 2., 0.]],
                              ["a", "b"], [[3., 2., 0.]])["rows"][0]
    assert not r["choice_changed"]
    assert r["max_shared_logit_delta"] == 2.


def test_ties_are_visible_instead_of_treating_argmax_as_unique():
    r = compare_choice_logits(["a", "b"], [[2., 2., 0.]],
                              ["b", "a"], [[2., 2., 0.]])["rows"][0]
    assert r["base_choice"] == "a" and r["variant_choice"] == "b"
    assert r["base_ties"] == ["a", "b"]
    assert r["variant_ties"] == ["b", "a"]
    assert r["choice_changed"] is True and r["shared_scores_unchanged"]


@pytest.mark.parametrize("args", [
    ([], [[0.]], ["b"], [[1., 0.]]),
    (["none"], [[1., 0.]], ["b"], [[1., 0.]]),
    (["a", "a"], [[1., 0., -1.]], ["b"], [[1., 0.]]),
    (["a"], [[float("nan"), 0.]], ["b"], [[1., 0.]]),
    (["a"], [[1., 0.]], ["b"], [[1., 0.], [1., 0.]]),
    (["a"], [1., 0.], ["b"], [[1., 0.]]),
])
def test_invalid_inputs_fail_closed(args):
    with pytest.raises(ValueError):
        compare_choice_logits(*args)


@pytest.mark.parametrize("atol", [-1., float("inf"), float("nan"), True])
def test_invalid_tolerance_rejected(atol):
    with pytest.raises(ValueError, match="atol"):
        compare_choice_logits(["a"], [[1., 0.]], ["a"], [[1., 0.]], atol=atol)


def test_seeded_randomized_permutation_parity():
    rng = np.random.default_rng(13)
    for _ in range(50):
        values = rng.normal(size=(7, 6))  # five candidate logits + none
        permutation = rng.permutation(5)
        report = compare_choice_logits([f"opt{i}" for i in range(5)], values,
                                       [f"opt{i}" for i in permutation],
                                       np.concatenate([values[:, permutation], values[:, -1:]], axis=1))
        assert all(row["shared_scores_unchanged"] and not row["choice_changed"]
                   for row in report["rows"])
