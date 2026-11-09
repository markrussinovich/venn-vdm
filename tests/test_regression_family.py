import json

import faiss
import numpy as np
import pytest
from fastapi.testclient import TestClient

from test_adapters_mcp import fake_assets
from venn.adapters import FaissOutcomeAdapter
from venn.decisions import decide as decisions_decide
from venn.errors import ContractError
from venn.heads import regression
from venn.profiles import TEXT_TAG, VectorTag
from venn.server import DecisionService, create_app


def no_embedding(texts):
    raise AssertionError("Vector states must not invoke a reader")


def service_for(tmp_path, name="registry.json"):
    if not (tmp_path / "assets").exists():
        fake_assets(tmp_path / "assets")
    return DecisionService(no_embedding, model_dir=tmp_path / "assets", cache_path=tmp_path / "cache",
                           registry_path=tmp_path / name)


def data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 768)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    w = rng.normal(size=768)
    y = 88 + 40 * (X @ w) / np.sqrt(768) + rng.normal(scale=0.5, size=n)
    return X, y


def examples(X, y):
    return [{"state": {"vector": x.tolist()}, "points": float(v)} for x, v in zip(X, y)]


def test_register_answer_batch_and_decisions_shape(tmp_path):
    X, y = data()
    service = service_for(tmp_path)
    report = service.register_family("wine_points", examples(X, y), objective="regress", target="points", seed=0)
    assert report["objective"] == "regress" and report["form"] == "ridge" and report["slices"] == {"train": 180, "selection": 60, "report": 60}
    assert set(report["report"]) == {"rmse", "mae", "spearman"} and report["interval"]["level"] == 0.8
    assert report["calibrated"] is False  # fewer than 500 report items
    assert len(report["checksum"]) == 64 and report["seconds"] >= 0
    one = service.decide({"state": {"vector": X[0].tolist()}, "questions": {"q": {"type": "Score", "family": "wine_points"}}})["answers"]["q"]
    assert set(one) == {"score", "interval", "interval_level", "calibrated", "source"} and one["source"] == "family_head"
    assert one["interval"][0] < one["score"] < one["interval"][1]
    many = service.decide({"state": {"vectors": X[:5].tolist()}, "questions": {"q": {"type": "Score", "family": "wine_points"}}})["answers"]["q"]["rows"]
    assert [round(r["score"], 9) for r in many][0] == round(one["score"], 9) and len(many) == 5
    with TestClient(create_app(service), base_url="http://127.0.0.1") as client:
        body = {"model": "venn-1", "state": {"vector": X[0].tolist()}, "questions": {"points": {"type": "score", "family": "wine_points"}}}
        answer = client.post("/api/alpha/decisions", json=body).json()["answers"]["points"]
    assert answer["type"] == "score" and abs(answer["score"] - one["score"]) < 1e-9 and answer["interval"] == one["interval"]
    assert answer["interval_level"] == one["interval_level"] == 0.8
    assert decisions_decide(service, body)["answers"]["points"]["source"] == "family_head"


def test_deterministic_for_a_fixed_seed_and_persisted(tmp_path):
    X, y = data()
    a = service_for(tmp_path, "a.json").register_family("p", examples(X, y), objective="regress", target="points", seed=3)
    b = service_for(tmp_path, "b.json").register_family("p", examples(X, y), objective="regress", target="points", seed=3)
    assert a["checksum"] == b["checksum"] and a["alpha"] == b["alpha"] and a["report"] == b["report"]
    reloaded = service_for(tmp_path, "a.json")
    assert reloaded.families["p"]["report"]["checksum"] == a["checksum"]
    q = {"q": {"type": "Score", "family": "p"}}
    assert reloaded.decide({"state": {"vector": X[1].tolist()}, "questions": q}) == service_for(tmp_path, "b.json").decide({"state": {"vector": X[1].tolist()}, "questions": q})


def test_no_row_crosses_slices(tmp_path):
    X, y = data()
    service = service_for(tmp_path)
    service.register_family("p", examples(X, y), objective="regress", target="points", seed=1)
    hashes = service.families["p"]["report"]["split_hashes"]
    sets = [set(h) for h in hashes]
    assert sum(len(s) for s in sets) == len(X) and not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    with pytest.raises(ContractError, match="must not share rows"):
        service.register_family("bad", examples(X, y), objective="regress", target="points", split={"train": list(range(200)), "report": list(range(150, 300))})
    custom = service.register_family("all", examples(X, y), objective="regress", target="points", split={"train": list(range(300))})
    assert custom["slices"] == {"train": 300, "selection": 0, "report": 0} and custom["report"] is None
    assert custom["interval"]["source"].startswith("cross-validated")


def test_store_tags_are_pinned_and_other_profiles_refused(tmp_path):
    X, y = data()
    service = service_for(tmp_path)
    other = VectorTag("experimental:other-reader", "a" * 40, "text-classification", 768)
    index = faiss.IndexFlatL2(768)
    index.add(X)
    parts = {"points": {"index": index, "ids": list(range(len(X))), "outcomes": y.tolist()}}
    foreign = service.register_outcome_store(FaissOutcomeAdapter(parts, tag=other))
    with pytest.raises(ContractError, match="venn-1 profile"):
        service.register_family("p", objective="regress", target="points", from_store=foreign)
    pinned = service.register_outcome_store(FaissOutcomeAdapter(parts, tag=TEXT_TAG))
    report = service.register_family("p", objective="regress", target="points", from_store=pinned)
    assert service.families["p"]["tag"] == TEXT_TAG.to_dict() and report["source"]["tag"] == TEXT_TAG.to_dict()
    with pytest.raises(ContractError):
        service.decide({"state": {"vector": X[0].tolist(), "revision": "b" * 40}, "questions": {"q": {"type": "Score", "family": "p"}}})
    with pytest.raises(ContractError):
        service.register_family("q", [{"state": {"vector": x.tolist(), "revision": "b" * 40}, "points": 1.0} for x in X[:20]], objective="regress", target="points")


def test_from_store_round_trips_its_target(tmp_path):
    X, y = data()
    service = service_for(tmp_path)
    direct = service.register_family("from_examples", examples(X, y), objective="regress", target="points", seed=2)
    handle = service.register_index(X, [str(i) for i in range(len(X))], outcomes=y.tolist())
    from_index = service.register_family("from_index", objective="regress", from_store=handle, seed=2)
    index = faiss.IndexFlatL2(768)
    index.add(X)
    store = service.register_outcome_store(FaissOutcomeAdapter({"points": {"index": index, "ids": list(range(len(X))), "outcomes": y.tolist()}}, tag=TEXT_TAG))
    from_outcomes = service.register_family("from_outcomes", objective="regress", target="points", from_store=store, seed=2)
    # The three paths read the same rows and targets; vectors pass through float32 storage, so compare predictions.
    assert direct["alpha"] == from_index["alpha"] == from_outcomes["alpha"]
    assert from_index["source"] == {"kind": "index", "handle": handle, "rows": len(y), "target": "outcome"}
    assert from_outcomes["source"]["rows"] == len(y) and from_outcomes["source"]["target"] == "points"
    q = {"q": {"type": "Score", "family": None}}
    scores = {}
    for name in ("from_examples", "from_index", "from_outcomes"):
        q["q"]["family"] = name
        scores[name] = np.array([r["score"] for r in service.decide({"state": {"vectors": X[:50].tolist()}, "questions": q})["answers"]["q"]["rows"]])
    np.testing.assert_allclose(scores["from_index"], scores["from_examples"], atol=1e-6)
    np.testing.assert_allclose(scores["from_outcomes"], scores["from_examples"], atol=1e-6)
    np.testing.assert_allclose(service.regression_answers(service.families["from_examples"], X[:3])[0]["score"], scores["from_examples"][0])


def test_closed_form_matches_scikit_learn_ridge():
    sklearn = pytest.importorskip("sklearn.linear_model")
    X, y = data(200)
    coef, intercept = regression.fit_ridge(X, y, 0.1)
    reference = sklearn.Ridge(alpha=0.1).fit(X.astype(np.float64), y)
    np.testing.assert_allclose(regression.predict(coef, intercept, X), reference.predict(X.astype(np.float64)), atol=1e-8)


def test_regression_http_stream_respects_row_caps_and_batch_boundaries(tmp_path):
    X, y = data(30)
    service = service_for(tmp_path)
    service.register_family("points", examples(X, y), objective="regress", target="points")
    handle = service.register_index(X[:7], [f"note-{i}" for i in range(7)])
    with TestClient(create_app(service), base_url="http://127.0.0.1") as client:
        for cap in (0, 1, 5, 7, 20):
            response = client.post(f"/index-apply/{handle}", json={
                "type": "index_apply", "family": "points", "row_cap": cap, "batch_size": 2,
            })
            assert response.status_code == 200
            chunks = [json.loads(line) for line in response.text.splitlines()]
            rows = [row for chunk in chunks for row in chunk["rows"]]
            assert [row["id"] for row in rows] == [f"note-{i}" for i in range(min(cap, 7))]
            assert [chunk["offset"] for chunk in chunks] == list(range(0, min(cap, 7), 2))
            assert all(len(chunk["rows"]) <= 2 for chunk in chunks)
        for limit in ({"row_cap": -1}, {"row_cap": True}, {"row_cap": 1000001},
                      {"batch_size": 0}, {"batch_size": True}):
            response = client.post(f"/index-apply/{handle}", json={
                "type": "index_apply", "family": "points", **limit,
            })
            assert response.status_code == 422
            assert "stream_limit" in response.text


def test_text_vectors_are_normalised_in_float64_before_float16_storage(tmp_path):
    """A reader whose own normalisation is coarse (as bf16 is) must still yield unit vectors within float16 rounding."""
    from venn.reader import encode_batches

    rng = np.random.default_rng(4)
    raw = rng.normal(size=(6, 768)) * rng.uniform(0.5, 3.0, size=(6, 1))

    class Reader:
        def encode(self, texts, **kwargs):
            assert kwargs.get("normalize_embeddings") is False
            return raw[: len(texts)].astype(np.float32)

    vectors, _, _ = encode_batches(Reader(), [f"t{i}" for i in range(6)], 6, lambda *a: None)
    assert np.abs(np.linalg.norm(vectors.astype(np.float64), axis=1) - 1).max() < 5e-4
    calls = []

    def embed(texts):
        calls.append(len(texts))
        return raw[: len(texts)] * 1.003  # off unit length, as a bf16 reader's output can be

    fake_assets(tmp_path / "assets")
    service = DecisionService(embed, model_dir=tmp_path / "assets", cache_path=tmp_path / "cache", registry_path=tmp_path / "r.json")
    out = service.embed([f"note {i}" for i in range(6)] + ["note 0"])
    assert calls == [6] and service.embed_count == 6
    assert np.abs(np.linalg.norm(out.astype(np.float64), axis=1) - 1).max() < 1e-6
    np.testing.assert_allclose(out[0], out[6])
    assert service.embed(["note 3"]).shape == (1, 768) and calls == [6]


def test_batch_embedding_raises_threads_then_restores_the_serving_pin(tmp_path, monkeypatch):
    import torch

    seen = []

    def embed(texts):
        seen.append(torch.get_num_threads())
        return np.ones((len(texts), 768))

    fake_assets(tmp_path / "assets")
    service = DecisionService(embed, model_dir=tmp_path / "assets", cache_path=tmp_path / "cache", registry_path=tmp_path / "r.json")
    assert torch.get_num_threads() == 1
    monkeypatch.setenv("VENN_EMBED_THREADS", "3")
    service.embed([f"row {i}" for i in range(40)])
    assert seen == [3] and service.last_batch_threads == 3 and torch.get_num_threads() == 1
    service.embed(["a single request"])
    assert seen[-1] == 1 and torch.get_num_threads() == 1
