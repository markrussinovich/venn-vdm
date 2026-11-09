import copy
import json
from pathlib import Path

import pytest

from scripts.compare_reference import compare
from venn.quickstart import run
from venn.reproduction import preflight


def test_cpu_quickstart_does_not_download_models(monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: pytest.fail("Unexpected download"))
    result = run()
    assert result["answer"] == "account"
    assert set(result["options"]) == {"account", "billing"}


def test_comparison_rejects_missing_nonfinite_and_wrong_protocol():
    reference = json.loads((Path(__file__).parents[1] / "benchmarks/vennspace-v1-reference.json").read_text())
    run = {"protocol_sha256": reference["protocol_sha256"]}
    for metric in reference["metrics"]:
        node = run
        for key in metric["path"][:-1]:
            node = node.setdefault(key, {})
        node[metric["path"][-1]] = metric["value"]
    assert compare(run, reference)["passed"]
    changed = copy.deepcopy(run)
    changed["datasets"]["banking77"]["nearest_reference"]["accuracy"] = 0.
    assert not compare(changed, reference)["passed"]
    changed["datasets"]["banking77"]["nearest_reference"]["accuracy"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        compare(changed, reference)
    with pytest.raises(ValueError, match="Missing"):
        compare({"protocol_sha256": reference["protocol_sha256"]}, reference)
    with pytest.raises(ValueError, match="checksum"):
        compare({}, reference)


def test_preflight_fails_before_output_or_download_for_missing_gpu(tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="visible"):
        preflight(1, output, tmp_path / "cache.sqlite")
    with pytest.raises(ValueError, match="CPU demos"):
        preflight(0, output, tmp_path / "cache.sqlite")
    assert not output.exists()
