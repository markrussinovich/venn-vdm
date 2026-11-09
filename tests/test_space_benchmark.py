import json
from pathlib import Path

import numpy as np
import pytest

from venn.index.density import knn_outlier_scores
from venn.benchmark import geometric_tasks, shift_tasks


def toy():
    rng = np.random.default_rng(42)
    centers = np.eye(12, 16, dtype=np.float32)
    def rows(count):
        labels = np.repeat(np.arange(12), count)
        values = centers[labels] + rng.normal(0, .001, (len(labels), 16))
        return (values / np.linalg.norm(values, axis=1, keepdims=True)).astype(np.float32), labels
    return *rows(40), *rows(25)


def protocol():
    value = json.loads((Path(__file__).parents[1] / "benchmarks/vennspace-v1.json").read_text())
    value["membership_small"]["trials"] = 4
    value["membership_large"]["trials"] = 4
    value["population_shift"]["trials_per_condition"] = 2
    value["population_shift"]["permutations"] = 19
    return value


def test_frozen_geometric_tasks_are_deterministic_and_use_distinct_rows():
    x, y, held, held_y = toy()
    score, raw = geometric_tasks(x, y, held, held_y, protocol(), 42)
    repeated, repeated_raw = geometric_tasks(x, y, held, held_y, protocol(), 42)
    assert score == repeated
    assert score["nearest_reference"]["accuracy"] == 1
    assert score["membership_7plus1"]["accuracy"] == 1
    assert score["membership_190plus10"]["recall_at10"] == 1
    assert all(len(set(ids)) == 200 for ids in raw["large_rows"])
    for key in raw:
        np.testing.assert_array_equal(raw[key], repeated_raw[key])
    with pytest.raises(ValueError, match="Insufficient"):
        geometric_tasks(x[:10], y[:10], held, held_y, protocol(), 42)


def test_shift_populations_are_disjoint_and_recorded():
    _, _, vectors, labels = toy()
    result, raw = shift_tasks(vectors, labels, protocol(), 42)
    assert set(result) == {"null", "shifted"}
    for kind in result:
        for a, b in raw[f"shift_{kind}_rows"]:
            assert len(a) == len(b) == 48
            assert not set(a) & set(b)


def test_density_finds_known_intruder_and_rejects_invalid_k():
    rng = np.random.default_rng(42)
    values = np.r_[rng.normal(0, .01, (7, 4)), np.full((1, 4), 10)]
    assert knn_outlier_scores(values).argmax() == 7
    with pytest.raises(ValueError):
        knn_outlier_scores(values, k=len(values))
