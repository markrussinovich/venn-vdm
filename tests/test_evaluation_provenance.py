import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from venn.reproduction import implementation_provenance


SPEC = importlib.util.spec_from_file_location(
    "precedent_eval", Path(__file__).parents[1] / "examples/precedent_desk/eval.py")
desk = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(desk)


def test_auxiliary_vector_and_cached_comparison_inputs_are_fingerprinted(tmp_path):
    paths = {name: tmp_path / name for name in ("banking.npz", "gate.npz", "clinc.npz", "train.npz")}
    for name, path in paths.items():
        path.write_bytes(name.encode())
    head = tmp_path / "head"
    head.mkdir()
    (head / "config.json").write_text("{}")
    (head / "manifest.json").write_text(json.dumps({"sha256": {"config.json": "test"}}))
    (tmp_path / "j.json").write_text('{"jev": {}}')
    args = SimpleNamespace(vectors=str(paths["banking.npz"]), gate_vectors=str(paths["gate.npz"]),
                           clinc_vectors=str(paths["clinc.npz"]), clinc_train_vectors=str(paths["train.npz"]),
                           output=str(tmp_path), model_dir=str(head))
    gate = desk.input_provenance("h", args)
    assert gate["files_sha256"][str(paths["gate.npz"])] == hashlib.sha256(b"gate.npz").hexdigest()
    matched = desk.input_provenance("j", args)
    assert str(paths["clinc.npz"]) in matched["files_sha256"]
    assert str(paths["train.npz"]) not in matched["files_sha256"]
    clinc = desk.input_provenance("k", args)
    assert {str(paths[name]) for name in ("banking.npz", "clinc.npz", "train.npz")} <= clinc["files_sha256"].keys()
    assert str(paths["gate.npz"]) not in clinc["files_sha256"]
    assert str(tmp_path / "j.json") in clinc["files_sha256"]
    assert clinc["sources"]["clinc150"] == desk.CLINC_SOURCE
    paths["train.npz"].write_bytes(b"changed")
    assert desk.input_provenance("k", args)["files_sha256"] != clinc["files_sha256"]


@pytest.mark.parametrize("field", ["sha256", "labels", "rows", "vectors"])
def test_clinc_identity_rejects_altered_inputs(field):
    rows = {"train": [{"sha256": "a", "label": 2, "row": 7}, {"sha256": "b", "label": 3, "row": 9}]}
    data = {"train_sha256": np.array(["a", "b"]), "train_labels": np.array([2, 3]),
            "train_rows": np.array([7, 9]), "train_vectors": np.eye(2)}
    desk.validate_clinc_rows(data, rows, "train")
    data[f"train_{field}"] = data[f"train_{field}"][::-1] if field != "vectors" else np.eye(1)
    with pytest.raises(ValueError, match="pinned train rows"):
        desk.validate_clinc_rows(data, rows, "train")


def test_overlap_diagnostic_retains_duplicates_and_distinguishes_labels():
    data = {"train_sha": ["a", "a", "b"], "train_y": np.array([1, 1, 2]),
            "test_sha": ["a", "b", "c"], "test_y": np.array([1, 4, 5])}
    result = desk.overlap_diagnostic(data)
    assert (result["rows"], result["normalized_unique_texts"]) == (6, 3)
    assert (result["test_rows_matching_train"], result["same_label_test_matches"]) == (2, 1)
    assert data["train_sha"] == ["a", "a", "b"]


def test_clinc_history_preserves_offline_path_but_opt_in_checks_source(tmp_path, monkeypatch):
    import venn.benchmark

    test = tmp_path / "test.npz"
    train = tmp_path / "train.npz"
    np.savez(test, test_vectors=np.eye(2), test_labels=[0, 0],
             test_sha256=["a", "b"], test_rows=[0, 1])
    np.savez(train, train_vectors=np.eye(2), train_labels=[0, 0],
             train_sha256=["altered", "d"], train_rows=[0, 1])
    args = SimpleNamespace(clinc_vectors=str(test), clinc_train_vectors=str(train),
                           output=str(tmp_path), draws=1, check_texts=False)
    monkeypatch.setattr(venn.benchmark, "fetch_source", lambda *a: pytest.fail("Unexpected source access"))
    monkeypatch.setattr(desk, "STOPS", ("all",))
    monkeypatch.setattr(desk, "draw_store", lambda *a: np.arange(2))
    monkeypatch.setattr(desk, "predict", lambda *a: (np.zeros(2), None))
    monkeypatch.setattr(desk, "row_bootstrap", lambda *a: np.ones(2))
    monkeypatch.setattr(desk, "vote_predict", lambda *a: np.zeros(2))
    assert "not checked against source" in desk.item_k({}, args)["row_identity_verification"]
    rows = {"test": [{"sha256": h, "label": 0, "row": i} for i, h in enumerate(["a", "b"])],
            "train": [{"sha256": h, "label": 0, "row": i} for i, h in enumerate(["c", "d"])]}
    monkeypatch.setattr(venn.benchmark, "fetch_source", lambda *a: (rows, {}))
    args.check_texts = True
    with pytest.raises(ValueError, match="pinned train rows"):
        desk.item_k({}, args)


def test_clinc_matched_pair_cli_check_texts_records_pinned_source(tmp_path, monkeypatch):
    import venn.benchmark

    banking = tmp_path / "banking.npz"
    banking.write_bytes(b"banking fixture")
    clinc = tmp_path / "clinc-vectors.npz"
    np.savez(clinc, test_vectors=np.eye(2), test_labels=[0, 2],
             test_sha256=["a", "b"], test_rows=[7, 9],
             intent_names=["topic_a", "oos", "topic_b"],
             option_names=["topic a", "oos", "topic b"],
             option_vectors=np.array([[1, 0], [1, 1], [0, 1]]),
             instruction=[1, 0])
    head = tmp_path / "head"
    head.mkdir()
    (head / "manifest.json").write_text('{"sha256": {}}')
    output = tmp_path / "output"
    vectors = {"train_sha": ["c"], "train_y": np.array([0]),
               "test_sha": ["a", "b"], "test_y": np.array([0, 2])}
    monkeypatch.setattr(desk, "load", lambda path: vectors)
    checked = []

    def check_texts(value, source):
        checked.append(source)
        return {"train": [], "test": []}

    def fetch_source(source):
        checked.append(source)
        return {"test": [{"sha256": "a", "label": 0, "row": 7},
                         {"sha256": "b", "label": 2, "row": 9}]}, {}

    monkeypatch.setattr(desk, "check_texts", check_texts)
    monkeypatch.setattr(venn.benchmark, "fetch_source", fetch_source)
    monkeypatch.setattr(desk, "service_for", lambda args: SimpleNamespace(generic=1))
    monkeypatch.setattr(desk, "choice_predictions", lambda *args: np.array([0, 1]))
    monkeypatch.setattr(desk, "row_bootstrap", lambda *args: np.ones(2))
    monkeypatch.delenv("VENN_TEST_NO_API_KEY", raising=False)
    monkeypatch.setattr(desk.sys, "argv", ["eval.py", "--vectors", str(banking),
                        "--output", str(output), "--model-dir", str(head),
                        "--items", "j", "--check-texts", "--jev-key-env", "VENN_TEST_NO_API_KEY"])
    desk.main()
    result = json.loads((output / "j.json").read_text())
    assert checked == [desk.BANKING_SOURCE, desk.CLINC_SOURCE]
    assert result["dataset"] == desk.CLINC_SOURCE
    assert result["rows"] == 2 and result["intents"] == 2
    assert result["jev"]["status"] == "not run: no API key"
    assert str(clinc) in result["input_provenance"]["files_sha256"]


def test_implementation_fingerprints_detect_edits_without_git(tmp_path):
    source = tmp_path / "venn"
    source.mkdir()
    module = source / "benchmark.py"
    module.write_text("version = 1\n")
    first = implementation_provenance(tmp_path)
    assert first["source_revision"] is None and first["source_dirty"] is None
    module.write_text("version = 2\n")
    second = implementation_provenance(tmp_path)
    assert first["code_sha256"]["venn/benchmark.py"] != second["code_sha256"]["venn/benchmark.py"]
