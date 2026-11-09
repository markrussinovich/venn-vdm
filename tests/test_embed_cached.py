"""embed_cached merges GPU shards into the cache and leaves no shard files or telemetry sidecars behind."""
import json
from concurrent.futures import Future
from pathlib import Path

import huggingface_hub
import numpy as np

from venn import reader


class InlinePool:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, function, *args):
        future = Future()
        future.set_result(function(*args))
        return future


def fake_worker(rank, texts, output, batch_size, cache_path, keys):
    np.save(output, np.ones((len(texts), 768), dtype=np.float16))
    telemetry = {"gpu": rank, "texts": len(texts), "status": "complete", "seconds": 1.0}
    Path(str(output) + ".123.telemetry.json").write_text(json.dumps(telemetry))
    return telemetry


def test_merged_shards_leave_no_sidecars(tmp_path, monkeypatch):
    monkeypatch.setattr(reader, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(reader, "embed_worker", fake_worker)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args, **kwargs: None)
    run = tmp_path / "run"
    run.mkdir()
    vectors, telemetry = reader.embed_cached(["a", "b", "c"], tmp_path / "cache.sqlite", run, gpu_count=2)
    assert vectors.shape == (3, 768) and telemetry["cache_misses"] == 3 and len(telemetry["workers"]) == 2
    assert list(run.iterdir()) == []
