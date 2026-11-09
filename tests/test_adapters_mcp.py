import hashlib
import json
from contextlib import closing
from types import SimpleNamespace

import faiss
import numpy as np
import pytest
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from venn.server import DecisionService, create_app
from venn.heads.abstention import CosineAbstention
from venn.adapters import (FaissAdapter, LanceDBAdapter, PgVectorAdapter, QdrantAdapter,
                           WeaviateAdapter)
from venn.profiles import VENN_1


@pytest.fixture
def vectors():
    rows = np.random.default_rng(42).normal(size=(5, 768)).astype(np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def pin():
    return {"profile": VENN_1.id, "revision": VENN_1.revision}


def test_faiss_profile_and_tail_batch(vectors):
    index = faiss.IndexFlatIP(768)
    index.add(vectors)
    adapter = FaissAdapter(index, list(range(5)), **pin())
    batches = list(adapter.batches(3))
    assert [len(row["ids"]) for row in batches] == [3, 2]
    np.testing.assert_array_equal(np.concatenate([row["vectors"] for row in batches]), vectors)
    with pytest.raises(ValueError, match="Profile mismatch"):
        FaissAdapter(index, list(range(5)), profile="venn-1-text", revision=VENN_1.revision)


def test_qdrant_pagination_and_named_vectors(vectors):
    calls = []
    def scroll(**kwargs):
        calls.append(kwargs)
        i = 0 if kwargs["offset"] is None else kwargs["offset"]
        end = min(i + kwargs["limit"], 5)
        return [SimpleNamespace(id=n, vector={"text": vectors[n]}, payload={"label": n % 2})
                for n in range(i, end)], end if end < 5 else None
    adapter = QdrantAdapter(SimpleNamespace(scroll=scroll), "tickets", vector_name="text",
                            outcome_key="label", **pin())
    batches = list(adapter.batches(3))
    assert [i for row in batches for i in row["ids"]] == list(range(5))
    assert [call["offset"] for call in calls] == [None, 3]
    assert all(call["with_vectors"] for call in calls)
    with pytest.raises(ValueError, match="named vector"):
        list(QdrantAdapter(SimpleNamespace(scroll=scroll), "tickets", **pin()).batches())


def test_pgvector_server_cursor_and_identifier_validation(vectors):
    rows = iter([[(i, vectors[i]) for i in range(3)], [(i, vectors[i]) for i in range(3, 5)], []])
    queries = []
    cursor = SimpleNamespace(execute=queries.append, fetchmany=lambda size: next(rows))
    assert sum(len(row["ids"]) for row in PgVectorAdapter(cursor, "tickets", **pin()).batches(3)) == 5
    assert queries == ['SELECT "id", "embedding" FROM "tickets" ORDER BY "id"']
    with pytest.raises(ValueError, match="identifiers"):
        PgVectorAdapter(cursor, "tickets; DROP TABLE tickets", **pin())


def test_weaviate_and_lancedb_batch_readers(vectors):
    items = [SimpleNamespace(uuid=str(i), vector={"default": vector}, properties={"label": i % 2})
             for i, vector in enumerate(vectors)]
    collection = SimpleNamespace(iterator=lambda **kwargs: iter(items))
    batches = list(WeaviateAdapter(collection, outcome_key="label", **pin()).batches(3))
    assert [len(row["ids"]) for row in batches] == [3, 2]
    records = [SimpleNamespace(to_pylist=lambda: [{"id": i, "vector": vectors[i]} for i in range(5)])]
    query = SimpleNamespace(to_batches=lambda **kwargs: iter(records))
    query.select = lambda columns: query
    query.limit = lambda count: query
    table = SimpleNamespace(search=lambda: query)
    assert list(LanceDBAdapter(table, **pin()).batches())[0]["ids"] == list(range(5))


def test_real_local_qdrant_named_vector_scan(vectors):
    qdrant = pytest.importorskip("qdrant_client")
    from qdrant_client import models
    with closing(qdrant.QdrantClient(":memory:")) as client:
        client.create_collection("tickets", vectors_config={
            "text": models.VectorParams(size=768, distance=models.Distance.COSINE)})
        client.upsert("tickets", [
            models.PointStruct(id=i, vector={"text": vector.tolist()}, payload={"label": i % 2})
            for i, vector in enumerate(vectors)])
        batches = list(QdrantAdapter(client, "tickets", vector_name="text",
                                    outcome_key="label", **pin()).batches(3))
    assert [len(row["ids"]) for row in batches] == [3, 2]
    assert [i for row in batches for i in row["ids"]] == list(range(5))
    np.testing.assert_allclose(np.concatenate([row["vectors"] for row in batches]), vectors, atol=1e-6)


def test_real_local_lancedb_scan(tmp_path, vectors):
    lancedb = pytest.importorskip("lancedb")
    database = lancedb.connect(str(tmp_path / "lance"))
    rows = np.tile(vectors, (5, 1))
    table = database.create_table("tickets", [
        {"id": i, "vector": vector.tolist(), "label": i % 2} for i, vector in enumerate(rows)])
    batches = list(LanceDBAdapter(table, outcome_column="label", **pin()).batches(3))
    by_id = {identifier: (vector, outcome)
             for row in batches for identifier, vector, outcome in
             zip(row["ids"], row["vectors"], row["outcomes"])}
    assert set(by_id) == set(range(25))
    for i, vector in enumerate(rows):
        np.testing.assert_allclose(by_id[i][0], vector, atol=1e-6)
        assert by_id[i][1] == i % 2


def fake_assets(folder):
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({
        "reader": {"id": VENN_1.reader, "revision": VENN_1.revision, "dimensions": 768},
        "baseline": {"temperature": .1, "none_logit": 0}, "choice_temperature": 1,
    }))
    save_file(CosineAbstention(.1, 0).state_dict(), str(folder / "universal.safetensors"))
    (folder / "manifest.json").write_text(json.dumps({"sha256": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()
    }}))


def test_loader_only_geometric_membership_and_profile_rejection(tmp_path, vectors):
    assets = tmp_path / "assets"
    fake_assets(assets)
    def no_embedding(texts):
        raise AssertionError("Cached geometry must not invoke a reader")
    service = DecisionService(no_embedding, model_dir=assets, cache_path=tmp_path / "cache",
                              registry_path=tmp_path / "registry.json")
    with TestClient(create_app(service), base_url="http://127.0.0.1") as client:
        request = {"model": "venn-1", "state": {"vector": vectors[0].tolist()},
                   "questions": {"q": {"type": "set_membership", "set": vectors[1:].tolist()}}}
        result = client.post("/api/alpha/decisions", json=request)
        assert result.status_code == 200, result.text
        assert result.json()["answers"]["q"]["probabilities"] is None
        request["state"]["profile"] = "venn-1-text"
        assert client.post("/api/alpha/decisions", json=request).status_code == 400
    assert service.embed_count == 0


def test_mcp_tool_schema():
    pytest.importorskip("mcp")
    import asyncio
    from venn.mcp_server import create_server
    server = create_server()
    tools = asyncio.run(server.list_tools())
    assert len(tools) == 1 and tools[0].name == "decide"
    assert set(tools[0].inputSchema["properties"]) >= {"state", "questions", "model"}
