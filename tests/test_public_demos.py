import numpy as np

from venn.demos import million_sweep, retrieval_demo, RAG_OPTIONS, RAG_QUERY, PASSAGES
from venn.redteam import prepare


def test_sweep_consumes_final_partial_batch_and_real_service_outputs(tmp_path):
    from test_adapters_mcp import fake_assets
    fake_assets(tmp_path / "assets")
    for model_dir in (None, tmp_path / "assets"):
        events = []
        result = million_sweep(19, batch_size=8, model_dir=model_dir,
                               progress=lambda *event: events.append(event))
        assert result["rows"] == sum(result["counts"].values()) == 19
        assert result["serialized_bytes"] > 0
        assert result["scoring_serialization_consumption_seconds"] > 0
        assert events[-1] == ("consuming", 19, 19)


def test_rag_keeps_relevant_passages_and_preserves_before_control():
    embedded = {value: np.eye(4, dtype=np.float32)[i] for i, value in enumerate(RAG_OPTIONS)}
    embedded.update({value: np.eye(4, dtype=np.float32)[0 if i < 3 else 1]
                     for i, value in enumerate(PASSAGES)})
    embedded[RAG_QUERY] = np.eye(4, dtype=np.float32)[0]
    result = retrieval_demo(embedded)
    assert result["before"]["returned"] == 6
    assert set(result["after"]["passage_ids"]) == {0, 1, 2}
    assert result["after"]["precision"] == result["after"]["recall"] == 1
    assert set(result["after"]["passage_ids"]) <= set(result["before"]["passage_ids"])


def test_rag_filter_never_retrieves_additional_passages():
    embedded = {value: np.eye(4, dtype=np.float32)[i] for i, value in enumerate(RAG_OPTIONS)}
    embedded.update({value: np.eye(4, dtype=np.float32)[0 if i < 3 else 1]
                     for i, value in enumerate(PASSAGES)})
    embedded[RAG_QUERY] = np.eye(4, dtype=np.float32)[1]
    result = retrieval_demo(embedded)
    assert result["after"]["passage_ids"] == []
    assert result["after"]["precision"] is None


def test_redteam_keeps_task_labels_and_conflict_controls_separate():
    protocol = prepare()
    for case in protocol["cases"]:
        assert case["meaning_preserving"][0] == case["base"]
        assert case["base"] not in case["instruction_conflict"]
        assert "\nIgnore the categories" in case["instruction_conflict"][0]
    assert not protocol["guardrail_use_approved"]


def test_precedent_desk_rejects_foreign_browser_access(monkeypatch):
    import runpy
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    path = Path(__file__).parents[1] / "examples" / "precedent_desk" / "app.py"
    monkeypatch.syspath_prepend(str(path.parent))
    monkeypatch.setitem(sys.modules, "eval", SimpleNamespace(
        SEED=0, STOPS=[], QUEUE_INSTRUCTION=""))
    calls = []
    desk = SimpleNamespace(state=lambda: calls.append("state") or {"ready": True})
    app = runpy.run_path(str(path))["create_app"](desk)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        assert client.get("/api/state", headers={"Host": "attacker.example"}).status_code == 400
        assert client.get("/api/state", headers={"Origin": "http://attacker.example"}).status_code == 403
        assert calls == []
        assert client.get("/api/state", headers={"Origin": "http://127.0.0.1:8765"}).json() == {"ready": True}
        assert calls == ["state"]
