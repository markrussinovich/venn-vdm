"""Contract tests for choice families: all rows trained, settings chosen by (grouped) cross-validation, class
weighting and a caller-supplied split."""
import numpy as np
import pytest

from test_adapters_mcp import fake_assets
from venn.errors import ContractError
from venn.heads.family import cv_folds, row_weights
from venn.server import DecisionService

FAST = {"folds": 2, "max_epochs": 2}


def no_embedding(texts):
    raise AssertionError("Vector states must not invoke a reader")


def service_for(tmp_path, name="registry.json"):
    if not (tmp_path / "assets").exists():
        fake_assets(tmp_path / "assets")
    return DecisionService(no_embedding, model_dir=tmp_path / "assets", cache_path=tmp_path / "cache", registry_path=tmp_path / name)


def unit(x):
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


def data(per_label=(12, 12, 12), seed=0, groups_per_label=4):
    rng = np.random.default_rng(seed)
    centres = unit(rng.normal(size=(len(per_label), 768)))
    rows, labels, groups = [], [], []
    for label, count in enumerate(per_label):
        for i in range(count):
            rows.append(unit(centres[label] + rng.normal(0, .05, 768)))
            labels.append(str(label))
            groups.append(f"g{label}-{i % groups_per_label}")
    options = [{"label": str(i), "vector": c.tolist()} for i, c in enumerate(centres)]
    question = unit(rng.normal(size=768)).tolist()
    return np.array(rows), labels, groups, options, question


def examples(rows, labels, groups=None):
    return [dict({"state": {"vector": r.tolist()}, "label": l}, **({"group_id": groups[i]} if groups else {}))
            for i, (r, l) in enumerate(zip(rows, labels))]


def test_trains_on_every_row_by_default_and_chooses_settings_by_cross_validation(tmp_path):
    rows, labels, _, options, question = data()
    service = service_for(tmp_path)
    info = service.register_family("f", examples(rows, labels), options, question, cv=FAST)
    assert info["split_counts"] == [len(rows), 0, 0]
    assert set(info["cv_loss_by_weight_decay"]) == {"0.001", "0.01", "0.1"}
    assert all(len(curve) == 2 for curve in info["cv_loss_by_weight_decay"].values())
    assert 1 <= info["settings"]["epochs"] <= 2 and info["settings"]["chosen_by"] == "2-fold cross-validation"
    assert info["temperature_source"] == "out-of-fold logits" and info["report_accuracy"] is None
    assert info["calibrated"] is False and info["additional_report_items_needed"] == 500
    assert 0 <= info["cv_accuracy"] <= 1
    stored = service.families["f"]
    assert len(stored["split_hashes"][0]) == len(rows) and stored["split_hashes"][1] == stored["split_hashes"][2] == []
    answer = service.decide({"state": {"vector": rows[0].tolist()},
                             "questions": {"q": {"type": "Choice", "family": "f", "instructions_vec": question, "options": options}}})
    assert set(answer["answers"]["q"]["probabilities"]) == {"0", "1", "2", "none"}


def test_grouped_cross_validation_keeps_each_group_in_one_fold(tmp_path):
    rows, labels, groups, options, question = data()
    groups = np.array(groups)
    for fold in cv_folds(np.array([int(l) for l in labels]), 3, seed=0, groups=groups):
        others = np.setdiff1d(np.arange(len(groups)), fold)
        assert not set(groups[fold]) & set(groups[others])
    info = service_for(tmp_path).register_family("g", examples(rows, labels, list(groups)), options, question, cv=FAST)
    assert info["settings"]["chosen_by"] == "2-fold cross-validation grouped by group_id"


def test_same_seed_same_head(tmp_path):
    rows, labels, _, options, question = data()
    a = service_for(tmp_path, "a.json").register_family("f", examples(rows, labels), options, question, seed=3, cv=FAST)
    b = service_for(tmp_path, "b.json").register_family("f", examples(rows, labels), options, question, seed=3, cv=FAST)
    for key in ("settings", "cv_loss_by_weight_decay", "temperature", "cv_accuracy"):
        assert a[key] == b[key]


def test_class_weighting(tmp_path):
    labels = np.array([0] * 30 + [1] * 10)
    balanced = row_weights(labels, "balanced")
    assert balanced[0] == pytest.approx(40 / (2 * 30)) and balanced[-1] == pytest.approx(40 / (2 * 10))
    assert np.all(row_weights(labels, None) == 1) and row_weights(labels, {1: 3.0})[-1] == 3.0
    rows, labels_, _, options, question = data(per_label=(20, 6, 6))
    service = service_for(tmp_path)
    assert service.register_family("b", examples(rows, labels_), options, question, cv=FAST, class_weight="balanced")["class_weight"] == "balanced"
    assert service.register_family("d", examples(rows, labels_), options, question, cv=FAST, class_weight={"1": 2.0})["class_weight"] == {"1": 2.0}
    for bad in ("inverse", {"7": 1.0}, {"1": -1.0}):
        with pytest.raises(ContractError, match="class_weight"):
            service.register_family("x", examples(rows, labels_), options, question, cv=FAST, class_weight=bad)


def test_caller_split(tmp_path):
    rows, labels, groups, options, question = data(per_label=(15, 15, 15))
    order = np.arange(len(rows))
    train, selection, report = order[order % 3 == 0], order[order % 3 == 1], order[order % 3 == 2]
    service = service_for(tmp_path)
    info = service.register_family("s", examples(rows, labels), options, question, cv=FAST,
                                   split={"train": train.tolist(), "selection": selection.tolist(), "report": report.tolist()})
    assert info["split_counts"] == [len(train), len(selection), len(report)]
    assert info["temperature_source"] == "selection slice"
    assert info["report_accuracy"] is not None and info["additional_report_items_needed"] == 500 - len(report)
    with pytest.raises(ContractError, match="share rows"):
        service.register_family("x", examples(rows, labels), options, question, cv=FAST, split={"train": [0, 1, 2, 3, 4, 5], "report": [5, 6]})
    with pytest.raises(ContractError, match="out of range"):
        service.register_family("x", examples(rows, labels), options, question, cv=FAST, split={"train": [0, 999]})
    by_group = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    spanning = by_group[groups[0]]
    with pytest.raises(ContractError, match="span"):
        service.register_family("x", examples(rows, labels, groups), options, question, cv=FAST,
                                split={"train": [i for i in range(len(rows)) if i != spanning[-1]], "report": [spanning[-1]]})
    with pytest.raises(ContractError, match="cv must be"):
        service.register_family("x", examples(rows, labels), options, question, cv={"folds": 1})


def test_linear_fit_matches_scikit_learn():
    from sklearn.linear_model import LogisticRegression

    from venn.heads.linear_choice import fit_multinomial, logits

    rows, labels, _, _, _ = data(per_label=(20, 20, 20))
    y = np.array([int(l) for l in labels])
    for C in (0.1, 10.0):
        W, b = fit_multinomial(rows, y, 3, C)
        ours = np.exp(logits(rows, W, b))
        ours /= ours.sum(1, keepdims=True)
        theirs = LogisticRegression(C=C, max_iter=5000, tol=1e-10).fit(rows, y).predict_proba(rows)
        assert np.abs(ours - theirs).max() < 1e-4


def test_linear_family_registers_answers_and_round_trips(tmp_path):
    rows, labels, groups, options, question = data(per_label=(15, 15, 15))
    service = service_for(tmp_path)
    info = service.register_family("lin", examples(rows, labels, groups), options, question, form="linear", cv={"folds": 3})
    assert info["form"] == "linear" and info["split_counts"] == [len(rows), 0, 0]
    assert set(info["cv_log_loss_by_C"]) == {"0.01", "0.1", "1.0", "10.0", "100.0"}
    assert info["settings"]["chosen_by"] == "3-fold cross-validation log loss grouped by group_id"
    ask = {"state": {"vector": rows[0].tolist()}, "questions": {"q": {"type": "Choice", "family": "lin", "instructions_vec": question, "options": options}}}
    answer = service.decide(ask)["answers"]["q"]
    assert answer["answer"] == "0" and answer["probabilities"]["none"] < 1e-6
    subset = dict(ask, questions={"q": {"type": "Choice", "family": "lin", "instructions_vec": question, "options": options[1:]}})
    probabilities = service.decide(subset)["answers"]["q"]["probabilities"]
    assert set(probabilities) == {"1", "2", "none"} and abs(probabilities["1"] + probabilities["2"] - 1) < 1e-5
    reloaded = DecisionService(no_embedding, model_dir=tmp_path / "assets", cache_path=tmp_path / "cache", registry_path=tmp_path / "registry.json")
    assert reloaded.decide(ask)["answers"]["q"] == answer
    reloaded.delete_family("lin")
    assert not list(reloaded.family_store.glob("*.npz"))


def test_linear_family_split_weights_and_form(tmp_path):
    rows, labels, _, options, question = data(per_label=(20, 8, 8))
    order = np.arange(len(rows))
    service = service_for(tmp_path)
    info = service.register_family("lin", examples(rows, labels), options, question, form="linear", class_weight="balanced", cv={"folds": 2},
                                   split={"train": order[order % 3 != 2].tolist(), "report": order[order % 3 == 2].tolist()})
    assert info["class_weight"] == "balanced" and info["report_accuracy"] is not None
    assert info["split_counts"] == [int((order % 3 != 2).sum()), 0, int((order % 3 == 2).sum())]
    with pytest.raises(ContractError, match="form"):
        service.register_family("x", examples(rows, labels), options, question, form="ridge")


def runtime_entry(service, form="linear", bias=0.):
    import copy

    from venn.heads import regression
    from venn.heads.family import FamilyScorer
    from venn.heads.linear_choice import LinearChoiceHead
    from venn.profiles import TEXT_TAG
    from venn.reader import REVISION

    _, _, _, options, question = data()
    vectors = np.array([option["vector"] for option in options], dtype=np.float32)
    arrays = None
    if form == "linear":
        arrays = {"W": np.zeros((3, 768)), "b": np.array([0., 1., bias])}
        head = LinearChoiceHead(arrays["W"], arrays["b"], vectors).eval()
    elif form == "regress":
        arrays = {"coef": np.zeros(768), "intercept": np.float64(bias)}
        head = {**arrays, "offsets": [-1., 1.], "calibrated": False}
    else:
        head = FamilyScorer(10, 768).eval()
    entry = {"name": "f", "form": form, "objective": "regress" if form == "regress" else "choice",
             "head": head, "universal": copy.deepcopy(service.registration_universal) if form == "residual" else None,
             "universal_class": "CosineAbstention", "examples": 10, "options": vectors.tolist(),
             "labels": ["0", "1", "2"], "question": question, "fields": False,
             "temperature": 1., "calibrated": False, "k2_fallback": False,
             "profile": "venn-1", "revision": REVISION, "tag": TEXT_TAG.to_dict()}
    if form == "regress":
        entry.update(extended=False, report={"checksum": regression.checksum(arrays["coef"], arrays["intercept"]),
                                            "calibrated": False, "interval": {"offsets": [-1., 1.]}})
    return entry, arrays


@pytest.mark.parametrize("old_form,new_form", [("linear", "residual"), ("residual", "linear"), ("regress", "regress")])
@pytest.mark.parametrize("phase", ["weights", "manifest"])
def test_failed_family_replacement_preserves_memory_and_reload(tmp_path, monkeypatch, old_form, new_form, phase):
    import json
    import os

    import torch

    service = service_for(tmp_path)
    old, arrays = runtime_entry(service, old_form, bias=2.)
    service._commit_family(old, arrays)
    before = {path.name: path.read_bytes() for path in service.family_store.iterdir()}
    candidate, arrays = runtime_entry(service, new_form, bias=4.)
    if phase == "manifest":
        monkeypatch.setattr(os, "replace", lambda *args: (_ for _ in ()).throw(OSError("interrupted manifest")))
    elif arrays is None:
        real_save = torch.save

        def fail(payload, path):
            real_save(payload, path)
            raise OSError("interrupted weights")

        monkeypatch.setattr(torch, "save", fail)
    else:
        real_save = np.savez

        def fail(path, **values):
            real_save(path, **values)
            raise OSError("interrupted weights")

        monkeypatch.setattr(np, "savez", fail)
    with pytest.raises(OSError, match="interrupted"):
        service._commit_family(candidate, arrays)
    assert service.families["f"] is old
    assert {path.name: path.read_bytes() for path in service.family_store.iterdir()} == before
    reloaded = service_for(tmp_path)
    assert reloaded.families["f"]["form"] == old_form
    metadata = next(service.family_store.glob("*.json"))
    assert json.loads(metadata.read_text())["weights"] == old["weights"]


def test_family_replacement_validates_handles_before_commit_and_refreshes_compatible_ones(tmp_path):
    import copy

    service = service_for(tmp_path)
    old, arrays = runtime_entry(service, bias=2.)
    service._commit_family(old, arrays)
    options = [{"label": label, "vector": vector} for label, vector in zip(old["labels"], old["options"])]
    question = {"type": "Choice", "family": "f", "instructions_vec": old["question"], "options": options}
    handle = service.register_question("q", question)
    request = {"state": {"vector": old["options"][0]}, "questions": {"q": {"question_ref": handle}}}
    before = service.decide(request)["answers"]
    files = {path.name: path.read_bytes() for path in service.family_store.iterdir()}
    prepared = service.questions[handle]
    for change in ("labels", "vectors", "objective"):
        candidate, arrays = runtime_entry(service, bias=4.)
        if change == "labels":
            candidate["labels"] = ["x", "y", "z"]
        elif change == "vectors":
            candidate["options"] = copy.deepcopy(candidate["options"])
            candidate["options"][0][0] += .1
        else:
            candidate["objective"] = "regress"
        with pytest.raises(ContractError) as error:
            service._commit_family(candidate, arrays)
        assert error.value.code in ("family_options", "family_questions")
        assert service.families["f"] is old and service.questions[handle] is prepared
        assert {path.name: path.read_bytes() for path in service.family_store.iterdir()} == files
        assert service.decide(request)["answers"] == before
    candidate, arrays = runtime_entry(service, bias=4.)
    service._commit_family(candidate, arrays)
    assert service.questions[handle] is not prepared
    assert service.decide(request)["answers"] != before
    assert service_for(tmp_path).decide({"state": request["state"], "questions": {"q": question}})["answers"] == service.decide(request)["answers"]
    assert len(list(service.family_store.glob("*.npz"))) == 1


def test_legacy_family_files_reload_and_migrate_on_replacement(tmp_path):
    import json

    service = service_for(tmp_path)
    entry, arrays = runtime_entry(service)
    service._commit_family(entry, arrays)
    metadata = next(service.family_store.glob("*.json"))
    payload = json.loads(metadata.read_text())
    weights = service.family_store / (payload.pop("weights") + ".npz")
    weights.rename(metadata.with_suffix(".npz"))
    metadata.write_text(json.dumps(payload))
    legacy = service_for(tmp_path)
    replacement, arrays = runtime_entry(legacy, bias=4.)
    legacy._commit_family(replacement, arrays)
    assert not metadata.with_suffix(".npz").exists()
    assert service_for(tmp_path).families["f"]["weights"] == replacement["weights"]
    legacy.delete_family("f")
    assert not list(legacy.family_store.iterdir())


def test_duplicate_linear_option_vectors_keep_label_identity_and_cache_order(tmp_path):
    import torch

    from venn.heads.linear_choice import LinearChoiceHead

    service = service_for(tmp_path)
    entry, arrays = runtime_entry(service, bias=3.)
    entry["options"][1] = entry["options"][0]
    entry["head"] = LinearChoiceHead(arrays["W"], arrays["b"], entry["options"]).eval()
    service._commit_family(entry, arrays)
    options = [{"label": label, "vector": vector} for label, vector in zip(entry["labels"], entry["options"])]
    for subset in (options, options[:2], options[:2][::-1], options[1:2]):
        question = {"type": "Choice", "family": "f", "instructions_vec": entry["question"], "options": subset}
        prepared, labels, _, _, _ = service.prepared_question("q", question, "Choice")
        logits = prepared.logits(torch.tensor([entry["options"][0]]))[0, :-1]
        np.testing.assert_allclose(logits.numpy(), [arrays["b"][int(label)] for label in labels])
    reloaded = service_for(tmp_path)
    question["options"] = options[:2][::-1]
    assert reloaded.decide({"state": {"vector": entry["options"][0]}, "questions": {"q": question}})["answers"]["q"]["answer"] == "1"
