import hashlib
import json
import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient

from venn.server import DecisionService, create_app
from test_adapters_mcp import fake_assets


@pytest.mark.parametrize("operation", ["initialize", "embed", "invalid-text", "invalid-image"])
def test_cache_connections_close_after_success_and_error(tmp_path, monkeypatch, operation):
    from venn.errors import ContractError

    fake_assets(tmp_path / "assets")
    connections = []
    original = sqlite3.connect

    def connect(*args, **kwargs):
        connection = original(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect)
    try:
        service = DecisionService(lambda texts: np.ones((len(texts), 768), dtype=np.float32),
                                  model_dir=tmp_path / "assets", cache_path=tmp_path / "cache",
                                  registry_path=tmp_path / "registry.json")
        if operation == "embed":
            np.testing.assert_array_equal(service.embed(["ticket"]), service.embed(["ticket"]))
            assert service.embed_count == 1
        elif operation == "invalid-text":
            with pytest.raises(ContractError, match="Text must contain"):
                service.embed([""])
        elif operation == "invalid-image":
            with pytest.raises(ContractError, match="Invalid image"):
                service.media({"image": "AA=="})
        for connection in connections:
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                connection.execute("SELECT 1")
    finally:
        for connection in connections:
            connection.close()


@pytest.fixture
def client(tmp_path):
    fake_assets(tmp_path / "assets")
    def embed(texts):
        return np.array([np.random.default_rng(
            int.from_bytes(hashlib.sha256(text.encode()).digest()[:8])).normal(size=768)
            for text in texts], dtype=np.float32)
    service = DecisionService(embed, model_dir=tmp_path / "assets",
                              cache_path=tmp_path / "cache",
                              registry_path=tmp_path / "registry.json")
    with TestClient(create_app(service), base_url="http://127.0.0.1") as value:
        yield value


@pytest.fixture
def prewarmed_index(client):
    vectors = np.eye(3, 768).tolist()
    definition = {"type": "index_apply", "instructions_vec": vectors[0],
                  "options": [{"label": str(i), "vector": row} for i, row in enumerate(vectors)]}
    ref = client.post("/questions/check", json=definition).json()["question_ref"]
    index = client.post("/indexes", json={"vectors": {"vectors": vectors},
                                          "ids": ["a", "b", "c"]}).json()["index_ref"]
    return ref, index, definition, vectors


@pytest.mark.parametrize("cap,batch", [(1, 1), (3, 2)])
def test_prewarmed_question_matches_direct_bounded_and_streamed_answers(client, prewarmed_index, cap, batch):
    ref, index, definition, _ = prewarmed_index
    limits = {"row_cap": cap, "batch_size": batch, "threshold": .75}
    body = {"model": "venn-1", "state": {"index_ref": index},
            "questions": {"q": {**definition, **limits}}}
    direct = client.post("/api/alpha/decisions", json=body).json()["answers"]["q"]["rows"]
    question = {"type": "index_apply", "question_ref": ref, **limits}
    body["questions"]["q"] = question
    bounded = client.post("/api/alpha/decisions", json=body)
    assert bounded.status_code == 200
    streamed = client.post(f"/index-apply/{index}", json=question)
    assert streamed.status_code == 200
    rows = [row for line in streamed.text.splitlines() for row in json.loads(line)["rows"]]
    assert bounded.json()["answers"]["q"]["rows"] == rows == direct
    assert [row["id"] for row in rows] == ["a", "b", "c"][:cap]


@pytest.mark.parametrize("override", [{"options": []}, {"instructions_vec": []}, {"family": "changed"}])
def test_prewarmed_question_rejects_semantic_overrides_on_both_routes(client, prewarmed_index, override):
    ref, index, _, _ = prewarmed_index
    question = {"type": "index_apply", "question_ref": ref, **override}
    body = {"model": "venn-1", "state": {"index_ref": index}, "questions": {"q": question}}
    for response in (client.post("/api/alpha/decisions", json=body),
                     client.post(f"/index-apply/{index}", json=question)):
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "prewarm_override"


@pytest.mark.parametrize("kind", ["Choice", "Score", "Noul"])
def test_native_prewarmed_index_question_rejects_changed_type(client, prewarmed_index, kind):
    from venn.errors import ContractError

    ref, index, _, _ = prewarmed_index
    question = {"type": kind, "question_ref": ref}
    service = client.app.state.service
    with pytest.raises(ContractError) as bounded:
        service.decide({"state": {"index_ref": index}, "questions": {"q": question}})
    with pytest.raises(ContractError) as streamed:
        list(service.stream_index(index, question))
    assert bounded.value.code == streamed.value.code == "prewarm_override"


@pytest.mark.parametrize("kind", [None, "Choice", "Score", "Noul"])
def test_native_prewarmed_question_accepts_matching_declared_or_default_type(client, prewarmed_index, kind):
    _, _, definition, vectors = prewarmed_index
    definition = {key: value for key, value in definition.items() if key != "type"}
    if kind is not None:
        definition["type"] = kind
    service = client.app.state.service
    ref = service.register_question("single", definition)
    body = {"state": {"vector": vectors[0]}, "questions": {"q": definition}}
    direct = service.decide(body)["answers"]
    body["questions"]["q"] = {"type": kind or "Choice", "question_ref": ref}
    assert service.decide(body)["answers"] == direct


def test_prewarmed_question_cannot_inherit_a_different_request_family(client, prewarmed_index):
    from venn.errors import ContractError

    ref, index, _, _ = prewarmed_index
    with pytest.raises(ContractError) as error:
        client.app.state.service.decide({
            "family": "changed", "state": {"index_ref": index},
            "questions": {"q": {"question_ref": ref}},
        })
    assert error.value.code == "prewarm_override"


def test_prewarmed_question_rejects_unhashable_handle_on_both_routes(client, prewarmed_index):
    _, index, _, _ = prewarmed_index
    question = {"type": "index_apply", "question_ref": []}
    body = {"model": "venn-1", "state": {"index_ref": index}, "questions": {"q": question}}
    for response in (client.post("/api/alpha/decisions", json=body),
                     client.post(f"/index-apply/{index}", json=question)):
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "question_handle"


def test_public_fake_cache_contract_and_replay(client):
    request = {"model": "venn-1", "state": "Cannot log in",
               "questions": {
                   "team": {"type": "choice", "instructions": "Who owns the ticket?",
                            "criteria": {"account": "Login problems", "billing": "Refunds"}},
                   "defect": {"type": "noul", "instructions": "Is it broken?"},
                   "urgency": {"type": "score", "instructions": "How urgent?",
                               "criteria": ["Can wait", "Soon", "Blocked"]},
               }}
    response = client.post("/api/alpha/decisions", json=request)
    assert response.status_code == 200
    result = response.json()
    assert set(result["answers"]) == set(request["questions"])
    assert result["venn"]["vector_tag"]["representation"] == "text-classification"
    assert 0 <= result["answers"]["defect"]["noul"] <= 1
    with_vector = client.post("/api/alpha/decisions", json=request, headers={"X-Venn-Return-Vector": "1"}).json()
    assert "vector" not in result["venn"] and len(with_vector["venn"]["vector"]) == 768
    request["state"] = {"vector": with_vector["venn"]["vector"]}
    assert client.post("/api/alpha/decisions", json=request).json()["answers"] == result["answers"]


def test_public_experimental_and_model_gates(client):
    request = {"model": "venn-1", "state": {"audio": "AA=="},
               "questions": {"q": {"type": "noul", "instructions": "Is it a question?"}}}
    assert client.post("/api/alpha/decisions", json=request).json()["error"]["code"] == "experimental_media"
    request.update(model="venn-1-text", state="Question")
    assert client.post("/api/alpha/decisions", json=request).status_code == 400
    request.update(model="venn-1", state=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}])
    assert client.post("/api/alpha/decisions", json=request).json()["error"]["code"] == "experimental_media"
    response = client.post("/api/alpha/decisions", json=request,
                           headers={"X-Venn-Experimental": "media"})
    assert response.json()["error"]["code"] == "media_parts_unsupported"


@pytest.mark.parametrize("bandwidth", [1e-200, 1e200])
def test_population_shift_extreme_bandwidths_do_not_invent_significance(bandwidth):
    from venn.index.shift import population_shift, rbf_gram

    vectors = np.eye(2)
    with np.errstate(all="raise"):
        kernel = rbf_gram(vectors, bandwidth)
        result = population_shift(vectors, vectors, bandwidth, permutations=19)
    np.testing.assert_array_equal(np.diag(kernel), np.ones(2))
    assert kernel[0, 1] == (0. if bandwidth < 1 else 1.)
    assert result["shifted"] is False
    assert result["p_value"] == 1.
    assert result["effect_size"] == 0.


@pytest.mark.parametrize("bandwidth", [float("nan"), float("inf"), True, "bad"])
def test_population_shift_rejects_invalid_bandwidths(bandwidth):
    from venn.index.shift import population_shift

    with pytest.raises(ValueError):
        population_shift(np.eye(2), np.eye(2), bandwidth, permutations=19)


def test_population_shift_rejects_unrepresentable_kernel():
    from venn.index.shift import population_shift

    with pytest.raises(ValueError, match="finite"):
        population_shift(np.full((2, 2), 1e200), np.eye(2), 1., permutations=19)


@pytest.mark.parametrize("bandwidth", [0, True, "bad"])
def test_population_shift_http_reports_bandwidth_validation_errors(client, bandwidth):
    vectors = np.eye(2, 768).tolist()
    response = client.post("/api/alpha/decisions", json={
        "model": "venn-1", "state": {"vectors": vectors},
        "questions": {"q": {"type": "population_shift", "batch_b": {"vectors": vectors},
                            "bandwidth": bandwidth}},
    })
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "population_input"


def test_native_text_wrappers_and_json_content_keep_their_meaning(client):
    question = {"q": {"type": "noul", "instructions": "Is this broken?"}}
    body = {"model": "venn-1", "state": "Cannot log in", "questions": question}
    plain = client.post("/api/alpha/decisions", json=body).json()["answers"]
    body["state"] = {"text": "Cannot log in"}
    assert client.post("/api/alpha/decisions", json=body).json()["answers"] == plain
    body["state"] = {"message": "Cannot log in"}
    structured = client.post("/api/alpha/decisions", json=body).json()["answers"]
    body["state"] = '{"message": "Cannot log in"}'
    assert client.post("/api/alpha/decisions", json=body).json()["answers"] == structured

    service = client.app.state.service
    service.families["score"] = {"objective": "regress", "text_only": True,
                               "head": {"coef": np.arange(768), "intercept": 1.,
                                        "offsets": [-1., 1.], "calibrated": False}}
    body.update(state={"texts": ["first", "second"]},
                questions={"q": {"type": "score", "family": "score"}})
    response = client.post("/api/alpha/decisions", json=body)
    assert response.status_code == 200
    rows = response.json()["answers"]["q"]["rows"]
    assert len(rows) == 2
    for i, value in enumerate(("first", "second")):
        body["state"] = {"text": value}
        one = client.post("/api/alpha/decisions", json=body).json()["answers"]["q"]
        assert rows[i]["score"] == pytest.approx(one["score"], abs=1e-10)


@pytest.mark.parametrize("operation", ["inference", "family", "batch", "stream"])
def test_slow_work_keeps_health_responsive_and_serializes_mutations(client, monkeypatch, operation):
    import asyncio
    import threading

    import httpx

    service = client.app.state.service
    started, release, mutated = threading.Event(), threading.Event(), threading.Event()
    method = {"inference": "decide", "family": "register_family", "batch": "decide_batch", "stream": "stream_index"}[operation]
    original = getattr(service, method)

    def slow(*args, **kwargs):
        if operation == "stream":
            def chunks():
                started.set()
                assert release.wait(5), "Health request could not run during a stream"
                yield '{"rows":[]}\n'
            return chunks()
        started.set()
        assert release.wait(5), "Health request could not run while the worker was blocked"
        return original(*args, **kwargs) if operation == "inference" else {}

    original_index = service.register_index

    def index(*args, **kwargs):
        mutated.set()
        return original_index(*args, **kwargs)

    monkeypatch.setattr(service, method, slow)
    monkeypatch.setattr(service, "register_index", index)
    path, body = {
        "inference": ("/api/alpha/decisions", {"model": "venn-1", "state": "hello",
                                            "questions": {"q": {"type": "noul", "instructions": "Is it a greeting?"}}}),
        "family": ("/families/f", {"examples": [], "options": [], "instructions_vec": []}),
        "batch": ("/decide-batch", {"states": [], "questions": {}}),
        "stream": ("/index-apply/x", {"type": "index_apply"}),
    }[operation]

    async def run():
        transport = httpx.ASGITransport(app=client.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as http:
            work = asyncio.create_task(http.request("PUT" if operation == "family" else "POST", path, json=body))
            mutation = None
            try:
                assert await asyncio.to_thread(started.wait, 2)
                assert (await asyncio.wait_for(http.get("/health"), 1)).status_code == 200
                assert (await asyncio.wait_for(http.get("/playground"), 1)).status_code == 200
                mutation = asyncio.create_task(http.post("/indexes", json={
                    "vectors": [np.eye(1, 768)[0].tolist()], "ids": ["x"]}))
                await asyncio.sleep(.05)
                assert not mutated.is_set()
            finally:
                release.set()
                assert (await work).status_code == 200
                if mutation is not None:
                    assert (await mutation).status_code == 200
            assert mutated.is_set()

    asyncio.run(run())


def test_local_playground_is_packaged_without_external_scripts(client):
    response = client.get("/playground")
    assert response.status_code == 200
    assert "Venn local playground" in response.text
    assert 'fetch("/api/alpha/decisions"' in response.text
    assert response.headers["referrer-policy"] == "no-referrer"


@pytest.mark.parametrize("headers,status", [
    ({"Host": "attacker.example"}, 400),
    ({"Host": "localhost.attacker.example"}, 400),
    ({"Host": "attacker.example", "X-Forwarded-Host": "localhost"}, 400),
    ({"Host": "user@localhost"}, 400),
    ({"Host": "localhost:invalid"}, 400),
    ([("Host", "localhost"), ("Host", "attacker.example")], 400),
    ({"Origin": "http://attacker.example"}, 403),
    ({"Origin": "null"}, 403),
    ({"Origin": "not-an-origin"}, 403),
    ({"Origin": "http://127.0.0.1:8001"}, 403),
    ({"Origin": "https://127.0.0.1"}, 403),
    ([("Origin", "http://127.0.0.1"), ("Origin", "http://attacker.example")], 403),
    ({"Origin": "http://127.0.0.1:80"}, 200),
    ({"Host": "[::1]:8000", "Origin": "http://[::1]:8000"}, 200),
])
def test_local_api_validates_host_and_browser_origin(client, headers, status):
    assert client.get("/health", headers=headers).status_code == status


def test_foreign_requests_cannot_invoke_mutations_or_raw_json_scoring(client, monkeypatch):
    calls = []
    service = client.app.state.service
    monkeypatch.setattr(service, "delete_family", lambda name: calls.append(name))
    monkeypatch.setattr(service, "decide_batch", lambda *args, **kwargs: calls.append("batch") or {})
    assert client.delete("/families/sentinel", headers={"Host": "attacker.example"}).status_code == 400
    assert client.delete("/families/sentinel", headers={"Origin": "null"}).status_code == 403
    body = '{"states": [], "questions": {}}'
    assert client.post("/decide-batch", content=body, headers={
        "Content-Type": "text/plain", "Origin": "http://attacker.example"}).status_code == 403
    assert calls == []
    assert client.delete("/families/sentinel").status_code == 200
    assert client.post("/decide-batch", content=body, headers={
        "Content-Type": "application/json", "Origin": "http://127.0.0.1"}).status_code == 200
    assert calls == ["sentinel", "batch"]


def test_nonlocal_hosts_require_explicit_configuration(client):
    app = create_app(client.app.state.service, allowed_hosts=["api.example"])
    with TestClient(app, base_url="https://api.example") as remote:
        assert remote.get("/health", headers={"Origin": "https://api.example:443"}).status_code == 200
        assert remote.get("/health", headers={"Host": "127.0.0.1"}).status_code == 400
        assert remote.get("/health", headers={"Origin": "https://attacker.example"}).status_code == 403


def test_public_space_density_extension_matches_known_outlier(client):
    rng = np.random.default_rng(42)
    vectors = np.zeros((8, 768), dtype=np.float32)
    vectors[:7, 0] = 1
    vectors[:7] += rng.normal(0, .001, (7, 768))
    vectors[7, 1] = 1
    request = {"model": "venn-1", "state": {"vectors": vectors.tolist()},
               "questions": {"odd": {"type": "set_membership", "method": "knn5"}}}
    response = client.post("/api/alpha/decisions", json=request)
    assert response.status_code == 200, response.text
    assert response.json()["answers"]["odd"]["answer"] == 7
    assert response.json()["answers"]["odd"]["probabilities"] is None
    for invalid in (True, 0, len(vectors)):
        request["questions"]["odd"]["outliers"] = invalid
        assert client.post("/api/alpha/decisions", json=request).status_code == 422
    request["questions"]["odd"].pop("outliers")
    request["questions"]["odd"]["threshold"] = .9
    assert client.post("/api/alpha/decisions", json=request).status_code == 422


def test_generated_site_can_be_explicitly_mounted(client, tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<h1>Venn rendered docs</h1>")
    with TestClient(create_app(client.app.state.service, site_dir=site), base_url="http://127.0.0.1") as mounted:
        assert mounted.get("/guide/").text == "<h1>Venn rendered docs</h1>"
        assert mounted.get("/playground").status_code == 200


@pytest.mark.parametrize("family,experimental", [(None, True), ("ordinal", False)])
def test_score_registration_status_does_not_imply_calibration(client, monkeypatch, family, experimental):
    rubric = {"type": "score", "instructions": "Rate satisfaction", "criteria": ["Low", "Mid", "High"]}
    if family:
        rubric["family"] = family
    labels = ["0", "1", "2"] if family else ["option-0", "option-1", "option-2"]
    def answer(request):
        assert request["questions"]["q"].get("family") == family
        return {"answers": {"q": {"probabilities": dict(zip(labels + ["none"], [.1, .2, .5, .2])),
                                 "calibrated": False, "answer": labels[2]}},
                "model": {}, "vector": None}
    monkeypatch.setattr(client.app.state.service, "decide", answer)
    result = client.post("/api/alpha/decisions", json={
        "model": "venn-1", "state": "review", "questions": {"q": rubric}}).json()
    score = result["answers"]["q"]
    assert score["score"] == pytest.approx(1.5)
    assert score["venn"]["experimental"] is experimental
    assert not score["venn"]["calibrated"]


def test_public_family_fitting_replay_and_profile_migration_rejection(client, tmp_path):
    service = client.app.state.service
    criteria = ["Low satisfaction", "Average satisfaction", "High satisfaction"]
    instruction = "Rate satisfaction"
    options = [{"label": str(i), "vector": service.vector(value).tolist()}
               for i, value in enumerate(criteria)]
    rng = np.random.default_rng(42)
    rows = [np.asarray(option["vector"]) + rng.normal(0, .01, 768)
            for option in options for _ in range(10)]
    response = client.put("/families/ordinal", json={
        "options": options, "instructions_vec": service.vector(instruction).tolist(), "cv": {"folds": 2, "max_epochs": 3},
        "examples": [{"state": row.tolist(), "label": str(i // 10)}
                     for i, row in enumerate(rows)]})
    assert response.status_code == 200, response.text
    request = {"model": "venn-1", "state": {"vector": rows[0].tolist()}, "questions": {
        "score": {"type": "score", "family": "ordinal", "instructions": instruction, "criteria": criteria}}}
    result = client.post("/api/alpha/decisions", json=request)
    assert result.status_code == 200, result.text
    answer = result.json()["answers"]["score"]
    assert not answer["venn"]["experimental"]
    def no_embedding(texts):
        raise AssertionError("Persisted replay must use the exact fake cache")
    reloaded = DecisionService(no_embedding, model_dir=tmp_path / "assets",
                               cache_path=tmp_path / "cache",
                               registry_path=tmp_path / "registry.json")
    with TestClient(create_app(reloaded), base_url="http://127.0.0.1") as replay:
        assert replay.post("/api/alpha/decisions", json=request).json()["answers"]["score"] == answer
    metadata = next(service.family_store.glob("*.json"))
    import json
    saved = json.loads(metadata.read_text())
    saved["profile"] = "venn-1-text"
    metadata.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="profile mismatch"):
        DecisionService(no_embedding, model_dir=tmp_path / "assets",
                        cache_path=tmp_path / "cache", registry_path=tmp_path / "registry.json")
