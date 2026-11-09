import hashlib
import json
from contextlib import closing

import faiss
import numpy as np
import pytest
from fastapi.testclient import TestClient

from venn.server import DecisionService, create_app
from test_adapters_mcp import fake_assets
from venn import OutcomeService
from venn.adapters import (FaissOutcomeAdapter, LanceDBOutcomeAdapter,
                           PgVectorOutcomeAdapter, QdrantOutcomeAdapter)
from venn.profiles import IMAGE_TAGS, VENN_1, VectorTag


TAG = VectorTag(VENN_1.id, VENN_1.revision, "test-projection", 2)


def partition(vectors=(), outcomes=()):
    index = faiss.IndexFlatL2(2)
    if len(vectors):
        index.add(np.asarray(vectors, dtype=np.float32))
    return {"index": index, "ids": list(range(len(vectors))), "outcomes": list(outcomes),
            "metadata": [{"tenant": "a"} for _ in vectors]}


def question(handle, **kwargs):
    return {"q": {"type": "nearest_outcome", "store": handle, "options": ["a", "b"],
                  "k": 2, **kwargs}}


def state(vector=(1., 0.)):
    return {"vector": list(vector), "tag": TAG.to_dict()}


def test_fresh_outcomes_mean_evidence_min_support_and_minimize():
    parts = {"a": partition([[1, 0], [1, .1]], [2, 4]),
             "b": partition([[1, .2]], [5])}
    service = OutcomeService()
    handle = service.register_store(FaissOutcomeAdapter(parts, tag=TAG))
    result = service.decide_batch([state(), state((1, .1))], question(handle))
    answer = result["results"][0]["answers"]["q"]
    assert answer["answer"] == "b"
    assert answer["options"]["a"]["mean"] == 3
    assert answer["options"]["a"]["spread"] == 1
    assert answer["options"]["a"]["count"] == 2
    assert len(answer["options"]["a"]["evidence_ids"]) == 2
    assert service.decide_batch([state()], question(handle, goal="minimize"))["results"][0]["answers"]["q"]["answer"] == "a"
    assert service.decide_batch([state()], question(handle, min_support=2))["results"][0]["answers"]["q"]["answer"] == "a"
    parts["b"]["index"].add(np.asarray([[1, .05]], dtype=np.float32))
    parts["b"]["ids"].append(1)
    parts["b"]["outcomes"].append(-10)
    parts["b"]["metadata"].append({"tenant": "a"})
    assert service.decide_batch([state()], question(handle))["results"][0]["answers"]["q"]["answer"] == "a"
    parts["a"]["metadata"][0]["tenant"] = "b"
    filtered = service.decide_batch([state()], question(handle, filters={"tenant": "b"}))
    assert filtered["results"][0]["answers"]["q"]["options"]["a"]["mean"] == 2
    assert filtered["results"][0]["answers"]["q"]["options"]["b"]["count"] == 0


def test_empty_store_and_explicit_exploration_prior_are_not_measured_evidence():
    service = OutcomeService()
    handle = service.register_store(FaissOutcomeAdapter({}, tag=TAG))
    answer = service.decide_batch([state()], question(handle, unsupported_value=0),
                                  rng=np.random.default_rng(42))["results"][0]["answers"]["q"]
    assert answer["answer"] == "none"
    assert answer["action"] in ("a", "b") and answer["exploratory"]
    assert answer["options"]["a"]["mean"] is None
    assert answer["probabilities"] is None and not answer["calibrated"]


def test_exact_key_override_is_opt_in_and_does_not_change_default_knn_mean():
    parts = {"a": partition([[1, 0], [1, .1]], [2, 4])}
    parts["a"]["exact_positions"] = {hashlib.sha256(np.asarray([1, 0], np.float16).tobytes()).digest(): 0}
    service = OutcomeService()
    handle = service.register_store(FaissOutcomeAdapter(parts, tag=TAG, query_dtype="float16"))
    normal = service.decide_batch([state()], question(handle))["results"][0]["answers"]["q"]
    exact = service.decide_batch([state()], question(handle, exact_match=True))["results"][0]["answers"]["q"]
    assert normal["options"]["a"]["mean"] == 3
    assert exact["options"]["a"]["mean"] == 2 and exact["options"]["a"]["count"] == 1


def test_tags_limits_and_nonfinite_values_fail_closed():
    service = OutcomeService()
    handle = service.register_store(FaissOutcomeAdapter({}, tag=TAG))
    altered = state()
    altered["tag"]["fields"] = ["other"]
    with pytest.raises(ValueError, match="tag mismatch"):
        service.decide_batch([altered], question(handle))
    for bad in ([{"vector": [1, 0]}], [state((np.nan, 0))], [state((0, 0))], [state()] * 129):
        with pytest.raises(ValueError):
            service.decide_batch(bad, question(handle))
    with pytest.raises(ValueError):
        service.decide_batch([state()], question(handle, k=True))
    with pytest.raises(ValueError):
        service.decide_batch([state()], question(handle), epsilon=[np.nan])
    with pytest.raises(ValueError, match="Unknown"):
        service.decide_batch([state()], question([]))
    with pytest.raises(ValueError, match="uint8"):
        service.decide_batch(np.zeros((1, 3, 3), np.uint8), question(handle))


def test_public_http_outcome_and_binary_uint8_batch(tmp_path):
    fake_assets(tmp_path / "assets")
    outcomes = OutcomeService()
    handle = outcomes.register_store(FaissOutcomeAdapter({"a": partition([[1, 0]], [4])}, tag=TAG))
    service = DecisionService(model_dir=tmp_path / "assets", cache_path=tmp_path / "cache",
                              registry_path=tmp_path / "registry.json", outcome_service=outcomes)
    with TestClient(create_app(service), base_url="http://127.0.0.1") as client:
        response = client.post("/decide-batch", json={"states": [state()], "questions": question(handle)})
        assert response.status_code == 200, response.text
        assert response.json()["results"][0]["answers"]["q"]["answer"] == "a"
        response = client.post("/api/alpha/decisions", json={
            "model": "venn-1", "state": state(), "questions": question(handle)})
        assert response.status_code == 200, response.text
        assert response.json()["venn"]["vector_tag"] == TAG.to_dict()
        wrong = {**state(), "tag": {**state()["tag"], "revision": "0" * 40}}
        response = client.post("/decide-batch", json={"states": [wrong], "questions": question(handle)})
        assert response.status_code in (400, 422), response.text

        class Reader:
            def encode_frames(self, images, representation):
                assert images.dtype == np.uint8 and images.shape == (2, 3, 3, 3)
                assert images[1].mean() == 7
                return np.ones((2, 2304), np.float32) / 48

        outcomes.image_reader = Reader()
        image_store = outcomes.register_store(FaissOutcomeAdapter({}, tag=IMAGE_TAGS["image-thirds-70"]))
        metadata = {"shape": [2, 3, 3, 3], "questions": question(image_store)}
        images = np.stack([np.zeros((3, 3, 3), np.uint8), np.full((3, 3, 3), 7, np.uint8)])
        headers = {"Content-Type": "application/octet-stream", "X-Venn-Batch": json.dumps(metadata)}
        assert client.post("/decide-batch", content=images.tobytes(), headers=headers).status_code == 400
        headers["X-Venn-Experimental"] = "media"
        response = client.post("/decide-batch", content=images.tobytes(), headers=headers)
        assert response.status_code == 200, response.text
        assert len(response.json()["results"]) == 2
        assert response.json()["vector_tag"]["fields"] == ["left", "center", "right"]
        assert client.post("/decide-batch", content=b"bad", headers=headers).status_code == 422


def test_pgvector_filters_are_parameters_and_precede_limit():
    calls = []

    class Cursor:
        def execute(self, query, parameters):
            calls.append((query, parameters))

        def fetchall(self):
            return [("case", 4, .25)]

    adapter = PgVectorOutcomeAdapter(Cursor(), "cases", tag=TAG)
    result = adapter.search(np.asarray([[1, 0]]), ["a"], 11,
                            filters={"tenant": "x' OR 1=1 --"})
    assert result["a"][0][0]["outcome"] == 4
    assert '"tenant" IS NOT DISTINCT FROM %s' in calls[0][0]
    assert "OR 1=1" not in calls[0][0]
    assert calls[0][1][2] == "x' OR 1=1 --"


def test_actual_qdrant_outcome_read_your_writes_and_filter_pushdown():
    qdrant = pytest.importorskip("qdrant_client")
    from qdrant_client import models

    with closing(qdrant.QdrantClient(":memory:")) as client:
        client.create_collection("cases", vectors_config=models.VectorParams(size=2, distance=models.Distance.EUCLID))
        adapter = QdrantOutcomeAdapter(client, "cases", tag=TAG)
        for identifier, tenant in enumerate(["other", "wanted"]):
            client.upsert("cases", [models.PointStruct(
                id=identifier, vector=[1., identifier * .1],
                payload={"option": "a", "outcome": identifier + 3, "tenant": tenant})], wait=True)
        rows = adapter.search(np.asarray([[1, 0]]), ["a"], 1, filters={"tenant": "wanted"})["a"][0]
        assert rows[0]["id"] == 1 and rows[0]["outcome"] == 4


def test_actual_lancedb_fresh_query_and_filter_pushdown(tmp_path):
    lancedb = pytest.importorskip("lancedb")
    table = lancedb.connect(str(tmp_path / "lance")).create_table("cases", [
        {"id": 0, "vector": [1., 0.], "option": "a", "outcome": 1., "tenant": "other"}])
    adapter = LanceDBOutcomeAdapter(table, tag=TAG)
    table.add([{"id": 1, "vector": [1., .1], "option": "a", "outcome": 5., "tenant": "wanted"}])
    rows = adapter.search(np.asarray([[1, 0]], dtype=np.float32), ["a"], 1,
                          filters={"tenant": "wanted"})["a"][0]
    assert rows[0]["id"] == 1 and rows[0]["outcome"] == 5
