"""Contract tests for regression families with form="kernel_ridge", extra numeric inputs and a reader prompt."""
import hashlib

import numpy as np
import pytest

from test_adapters_mcp import fake_assets
from venn.errors import ContractError
from venn.heads import regression
from venn.reader import cache_key
from venn.server import DecisionService

WORDS = "ripe cherry oak tannin bright acid long finish plum spice earthy smoke vanilla structured elegant".split()


def fake_embedder(calls=None):
    """Deterministic text vectors; the prompt shifts them so different prompts give different vectors."""
    def embed(texts, prompt_name="Classification"):
        if calls is not None:
            calls.append(prompt_name)
        out = []
        for t in texts:
            seed = int(hashlib.sha256((prompt_name + "|" + t).encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).normal(size=768)
            out.append(v / np.linalg.norm(v))
        return np.array(out, dtype=np.float32)
    return embed


def service_for(tmp_path, embed=None, name="registry.json"):
    if not (tmp_path / "assets").exists():
        fake_assets(tmp_path / "assets")
    return DecisionService(embed or fake_embedder(), model_dir=tmp_path / "assets", cache_path=tmp_path / "cache", registry_path=tmp_path / name)


def notes(n=120, seed=0):
    rng = np.random.default_rng(seed)
    texts, y = [], []
    for i in range(n):
        k = int(rng.integers(5, 40))
        texts.append(f"note {i}: " + " ".join(rng.choice(WORDS, k)))
        y.append(84 + 0.15 * k + rng.normal(0, 0.5))  # longer notes score higher, as in the wine cookbook
    return texts, np.array(y)


def test_kernel_ridge_matches_scikit_learn():
    from sklearn.kernel_ridge import KernelRidge

    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 20))
    y = rng.normal(size=80) + 88
    dual, mu = regression.fit_kernel(X, y, 0.1, 0.5)
    head = {"form": "kernel_ridge", "X": X, "dual": dual, "mu": mu, "gamma": 0.5, "extra_mean": None}
    ours = regression.predict_head(head, X[:10])
    theirs = KernelRidge(kernel="rbf", alpha=0.1, gamma=0.5).fit(X, y - y.mean()).predict(X[:10]) + y.mean()
    assert np.abs(ours - theirs).max() < 1e-8


def test_conjugate_gradient_solve_recovers_a_known_solution():
    # no LAPACK reference: numpy's small LAPACK solves hang on the Windows ARM64 CI runner
    rng = np.random.default_rng(1)
    for n in (24, 36, 200):
        X = rng.normal(size=(n, 30))
        K = regression.rbf(X, X, 0.05)
        K[np.diag_indices_from(K)] += 0.01
        x = rng.normal(size=n)
        assert np.abs(regression.spd_solve(K, K @ x, "cg") - x).max() < 1e-7


def test_cache_keys_depend_on_the_prompt_and_default_keys_are_unchanged(tmp_path):
    assert cache_key("a note") == cache_key("a note", "Classification")
    assert cache_key("a note", "SentenceSimilarity") != cache_key("a note")
    calls = []
    service = service_for(tmp_path, fake_embedder(calls))
    a = service.embed(["a note"])
    b = service.embed(["a note"], "SentenceSimilarity")
    assert calls == ["Classification", "SentenceSimilarity"] and not np.allclose(a, b)
    with pytest.raises(ContractError, match="prompts"):
        service.embed(["a note"], "Summarise")


def test_text_family_with_prompt_inputs_and_kernel_ridge(tmp_path):
    texts, y = notes()
    service = service_for(tmp_path)
    report = service.register_family("wine", [{"state": t, "points": float(p)} for t, p in zip(texts, y)], objective="regress", target="points",
                                     form="kernel_ridge", prompt="SentenceSimilarity", inputs=["log_words", "log_chars"],
                                     split={"train": list(range(len(texts)))}, cv={"folds": 3, "repeats": 1})
    assert report["form"] == "kernel_ridge" and report["inputs"] == ["log_words", "log_chars"]
    assert report["alpha"] in regression.KERNEL_ALPHAS and report["gamma"] in regression.GAMMAS and report["input_scale"] in regression.INPUT_SCALES
    assert len(report["cv_rmse_by_setting"]) == len(regression.KERNEL_ALPHAS) * len(regression.GAMMAS) * len(regression.INPUT_SCALES)
    question = {"p": {"type": "Score", "family": "wine"}}
    one = service.decide({"state": texts[0], "questions": question})["answers"]["p"]
    many = service.decide({"state": {"texts": texts[:5]}, "questions": question})["answers"]["p"]["rows"]
    assert one["score"] == pytest.approx(many[0]["score"]) and one["source"] == "family_head"
    assert one["interval"][0] < one["score"] < one["interval"][1]
    with pytest.raises(ContractError, match="answers from text"):
        service.decide({"state": {"vector": service.embed([texts[0]])[0].tolist()}, "questions": question})
    handle = service.register_index(service.embed(texts[:3]), ["a", "b", "c"])
    with pytest.raises(ContractError, match="answers from text"):
        list(service.stream_index(handle, {"type": "IndexApply", "family": "wine"}))
    reloaded = service_for(tmp_path)
    assert reloaded.decide({"state": texts[0], "questions": question})["answers"]["p"] == one
    reloaded.delete_family("wine")
    assert not list(reloaded.family_store.glob("*.npz"))


def test_length_inputs_help_when_length_carries_the_target(tmp_path):
    texts, y = notes(160)
    service = service_for(tmp_path)
    examples = [{"state": t, "points": float(p)} for t, p in zip(texts, y)]
    split = {"train": list(range(120)), "report": list(range(120, 160))}
    plain = service.register_family("plain", examples, objective="regress", target="points", split=split, cv={"folds": 3, "repeats": 1})
    with_length = service.register_family("length", examples, objective="regress", target="points", inputs=["log_words"], split=split,
                                          cv={"folds": 3, "repeats": 1})
    assert with_length["form"] == "ridge" and with_length["inputs"] == ["log_words"]
    assert with_length["report"]["rmse"] < plain["report"]["rmse"]


def test_same_seed_same_kernel_family(tmp_path):
    texts, y = notes(60)
    examples = [{"state": t, "points": float(p)} for t, p in zip(texts, y)]
    kw = dict(objective="regress", target="points", form="kernel_ridge", inputs=["log_words"], cv={"folds": 3, "repeats": 1}, seed=2)
    a = service_for(tmp_path, name="a.json").register_family("k", examples, **kw)
    b = service_for(tmp_path, name="b.json").register_family("k", examples, **kw)
    assert a["checksum"] == b["checksum"] and a["alpha"] == b["alpha"] and a["gamma"] == b["gamma"]


def test_refusals(tmp_path):
    texts, y = notes(40)
    examples = [{"state": t, "points": float(p)} for t, p in zip(texts, y)]
    service = service_for(tmp_path)
    with pytest.raises(ContractError, match="form"):
        service.register_family("x", examples, objective="regress", target="points", form="lasso")
    with pytest.raises(ContractError, match="prompts"):
        service.register_family("x", examples, objective="regress", target="points", prompt="Summarise")
    with pytest.raises(ContractError, match="inputs"):
        service.register_family("x", examples, objective="regress", target="points", inputs=["log_bytes"])
    with pytest.raises(ContractError, match="text"):
        service.register_family("x", [{"state": {"vector": service.embed([t])[0].tolist()}, "points": float(p)} for t, p in zip(texts, y)],
                                objective="regress", target="points", inputs=["log_words"])
    plain = service_for(tmp_path, embed=lambda texts: fake_embedder()(texts), name="plain.json")
    with pytest.raises(ContractError, match="prompt_name"):
        plain.register_family("x", examples, objective="regress", target="points", prompt="SentenceSimilarity")
