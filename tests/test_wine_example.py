"""Exercise both runnable wine configurations without downloading notes or a reader."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from test_regression_extended import service_for

_spec = importlib.util.spec_from_file_location(
    "wine_example", Path(__file__).resolve().parents[1] / "examples/wine/wine.py")
wine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wine)


@pytest.mark.parametrize("form", ["ridge", "kernel_ridge"])
def test_wine_workflow_keeps_evaluation_targets_out_of_fitting(tmp_path, form):
    service = service_for(tmp_path)
    notes = [f"Wine {i}: " + "ripe cherry oak " * (i + 1) for i in range(24)]
    points = np.arange(24, dtype=float) / 2 + 80
    report, predictions, _ = wine.fit_and_predict(service, notes, points, 20, form)
    assert report["slices"] == {"train": 20, "selection": 0, "report": 0}
    assert len(predictions) == 4 and np.isfinite(predictions).all()
    family = service.families["wine_points"]
    assert family["prompt"] == ("SentenceSimilarity" if form == "kernel_ridge" else "Classification")
    assert family["inputs"] == (["log_words", "log_chars"] if form == "kernel_ridge" else [])
    changed = points.copy()
    changed[20:] += 1000000
    rerun, repeated, _ = wine.fit_and_predict(service, notes, changed, 20, form)
    assert rerun["checksum"] == report["checksum"]
    np.testing.assert_array_equal(repeated, predictions)
