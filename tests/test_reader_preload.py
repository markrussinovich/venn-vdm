"""The text reader loads once in the background at `venn serve` start; requests wait for that load, never load twice."""
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from test_adapters_mcp import fake_assets
from venn import cli
from venn.errors import ContractError
from venn.server import DecisionService, create_app


def service_for(tmp_path):
    fake_assets(tmp_path / "assets")
    return DecisionService(model_dir=tmp_path / "assets", cache_path=tmp_path / "cache", registry_path=tmp_path / "registry.json")


def fake_embedder(texts, prompt_name="Classification"):
    return np.array([np.random.default_rng(len(text)).normal(size=768) for text in texts], dtype=np.float32)


class SlowLoader:
    """Stands in for loading the reader: counts loads and blocks until released."""

    def __init__(self):
        self.calls, self.started, self.release = 0, threading.Event(), threading.Event()

    def __call__(self):
        self.calls += 1
        self.started.set()
        assert self.release.wait(10)
        return fake_embedder


def test_preload_runs_once_and_a_concurrent_embed_waits_for_it(tmp_path, monkeypatch):
    service = service_for(tmp_path)
    loader = SlowLoader()
    monkeypatch.setattr(service, "_load_text_reader", loader)
    assert service.reader_state == "idle"
    preload = service.preload_reader()
    assert loader.started.wait(10) and service.reader_state == "loading"
    result = {}
    request = threading.Thread(target=lambda: result.update(vectors=service.embed(["a new ticket"])))
    request.start()
    time.sleep(0.2)
    assert request.is_alive() and "vectors" not in result  # waiting for the background load
    loader.release.set()
    preload.join(10)
    request.join(10)
    assert result["vectors"].shape == (1, 768)
    assert loader.calls == 1 and service.reader_state == "ready"
    service.embed(["another"])
    assert loader.calls == 1


def test_absent_reader_still_serves_vectors_and_reports_it(tmp_path, monkeypatch, caplog):
    service = service_for(tmp_path)

    def missing():
        raise ContractError("reader_missing", "The EmbeddingGemma 2 reader is not downloaded. Run 'venn download-embedder'", 503)

    monkeypatch.setattr(service, "_load_text_reader", missing)
    service.preload_reader().join(10)
    assert service.reader_state == "absent"
    assert sum("download-embedder" in record.getMessage() for record in caplog.records) == 1
    vectors = np.eye(768, dtype=np.float32)[:6]
    request = {"model": "venn-1", "state": {"vector": vectors[0].tolist()},
               "questions": {"q": {"type": "set_membership", "set": vectors[1:].tolist()}}}
    with TestClient(create_app(service), base_url="http://127.0.0.1") as client:
        assert client.get("/health").json()["reader"] == "absent"
        assert client.post("/api/alpha/decisions", json=request).status_code == 200
        text = client.post("/api/alpha/decisions", json={**request, "state": "a new ticket"})
    assert text.status_code == 503, text.text
    assert text.json()["error"]["code"] == "reader_missing" and "download-embedder" in text.json()["error"]["message"]


@pytest.mark.parametrize("flags,expected", [([], 1), (["--no-preload"], 0)])
def test_serve_preloads_unless_told_not_to(tmp_path, monkeypatch, flags, expected):
    fake_assets(tmp_path / "assets")
    monkeypatch.setenv("VENN_HOME", str(tmp_path / "home"))
    calls = []
    monkeypatch.setattr(DecisionService, "preload_reader", lambda self: calls.append(self))
    import uvicorn
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: None)
    cli.main(["serve", "--model-dir", str(tmp_path / "assets"), *flags])
    assert len(calls) == expected


def test_serve_rejects_wildcard_hosts_before_starting(capsys):
    with pytest.raises(SystemExit):
        cli.main(["serve", "--allowed-host", "*.test"])
    assert "without wildcards" in capsys.readouterr().err
    with pytest.raises(ValueError, match="without wildcards"):
        create_app(object(), allowed_hosts=["*.test"])
