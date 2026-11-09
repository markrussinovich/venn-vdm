"""Embed texts with the pinned reader, caching vectors by content in SQLite."""

import hashlib
import json
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np


MODEL_ID = "google/embeddinggemma-2"
REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"


PROMPTS = ("Classification", "SentenceSimilarity", "Clustering")  # the reader's documented task prompts Venn accepts


def cache_key(text, prompt="Classification"):
    """Return the SHA-256 cache key for ``text`` under the pinned model, revision, prompt and dimension."""
    payload = json.dumps(
        [MODEL_ID, REVISION, "text-only", prompt, 768, text],
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def unit(values):
    """Normalise rows in float64. A bf16 reader's own normalisation leaves up to 0.4% length error
    (an 8-bit mantissa); this leaves only the float16 storage rounding, about 0.03%."""
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def encode_batches(model, texts, batch_size, consume):
    """Encode ``texts`` longest-first with ``model``, halving ``batch_size`` on CUDA out-of-memory errors.

    Calls ``consume(indices, values)`` after each batch. Returns ``(float16 vectors, final batch size,
    retry count)``; raises ``ValueError`` if the model returns a wrong shape or nonfinite values.
    """
    import torch

    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    vectors = np.empty((len(texts), 768), dtype=np.float16)
    position, retries = 0, 0
    while position < len(order):
        indices = order[position:position + batch_size]
        try:
            values = model.encode(
                [texts[i] for i in indices], prompt_name="Classification",
                batch_size=len(indices), normalize_embeddings=False, show_progress_bar=False,
                convert_to_numpy=True,
            )
        except torch.cuda.OutOfMemoryError:
            if batch_size <= 1:
                raise
            batch_size = max(1, batch_size // 2)
            retries += 1
            torch.cuda.empty_cache()
            continue
        values = unit(values)
        if values.shape != (len(indices), 768) or not np.isfinite(values).all():
            raise ValueError("Invalid embedding output")
        values = values.astype(np.float16)
        consume(indices, values)
        vectors[indices] = values
        position += len(indices)
    return vectors, batch_size, retries


def embed_worker(rank, texts, output, batch_size=64, cache_path=None, keys=None):
    """Embed ``texts`` on GPU ``rank``, save the vectors to ``output`` (``.npy``) and return a telemetry dict.

    When ``cache_path`` and ``keys`` are given, each batch is also written to the SQLite cache. A
    ``.telemetry.json`` file is written next to ``output`` even if encoding fails.
    """
    import os
    import torch
    from sentence_transformers import SentenceTransformer

    os.environ.setdefault("RAYON_NUM_THREADS", "2")
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    start = time.monotonic()
    model = SentenceTransformer(
        MODEL_ID, revision=REVISION, device=f"cuda:{rank}",
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"dtype": torch.bfloat16}, local_files_only=True,
    )
    telemetry = {
        "gpu": rank, "texts": len(texts),
        "parameters": sum(p.numel() for p in model.parameters()),
        "completed_texts": 0, "status": "failed",
    }
    db = sqlite3.connect(cache_path, timeout=120) if cache_path is not None else None
    try:
        def consume(indices, values):
            if db is not None:
                db.executemany(
                    "INSERT OR IGNORE INTO vectors VALUES (?, ?)",
                    [(keys[i], value.tobytes()) for i, value in zip(indices, values)],
                )
                db.commit()
            telemetry["completed_texts"] += len(indices)

        vectors, final_batch, retries = encode_batches(model, texts, batch_size, consume)
        np.save(output, vectors)
        telemetry.update(status="complete", final_batch_size=final_batch, oom_retries=retries)
    finally:
        if db is not None:
            db.close()
        telemetry["seconds"] = time.monotonic() - start
        Path(str(output) + f".{time.time_ns()}.telemetry.json").write_text(
            json.dumps(telemetry, indent=2) + "\n"
        )
    return telemetry


def embed_cached(texts, cache_path, run_path, gpu_count=8, batch_size=64):
    """Return unit-norm float32 vectors for ``texts``, embedding only cache misses across ``gpu_count`` GPUs.

    Returns ``(vectors, telemetry)``; telemetry holds cache hits and misses, per-worker stats and GPU hours.
    Shard files and their telemetry sidecars are written under ``run_path`` and removed once merged into
    ``cache_path``; a failed shard's sidecar stays for diagnosis.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    run_path = Path(run_path)
    keys = [cache_key(text) for text in texts]
    unique = dict(zip(keys, texts))
    with sqlite3.connect(cache_path) as db:
        db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, value BLOB)")
        db.commit()
        cached = {}
        for key in unique:
            row = db.execute("SELECT value FROM vectors WHERE key = ?", (key,)).fetchone()
            if row:
                cached[key] = np.frombuffer(row[0], dtype=np.float16).copy()
        missing = [key for key in unique if key not in cached]
        telemetry = []
        if missing:
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import LocalEntryNotFoundError

            try:
                snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)
            except LocalEntryNotFoundError:
                snapshot_download(MODEL_ID, revision=REVISION)
            partitions = [missing[rank::gpu_count] for rank in range(gpu_count)]
            with ProcessPoolExecutor(
                max_workers=gpu_count, mp_context=get_context("spawn")
            ) as pool:
                jobs = []
                for rank, part in enumerate(partitions):
                    if part:
                        output = run_path / f"embedding-shard-{rank}.npy"
                        jobs.append((
                            part, output,
                            pool.submit(
                                embed_worker, rank, [unique[k] for k in part], output,
                                batch_size, str(cache_path), part,
                            ),
                        ))
                for part, output, job in jobs:
                    telemetry.append(job.result())
                    values = np.load(output)
                    db.executemany(
                        "INSERT OR IGNORE INTO vectors VALUES (?, ?)",
                        [(key, value.tobytes()) for key, value in zip(part, values)],
                    )
                    db.commit()
                    cached.update(zip(part, values))
                    output.unlink()
                    # The worker's sidecar is kept only for a failed shard; a merged shard's telemetry is returned.
                    for sidecar in run_path.glob(f"{output.name}.*.telemetry.json"):
                        sidecar.unlink()
        vectors = np.stack([cached[key] for key in keys]).astype(np.float32)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return vectors, {
        "cache_hits": len(unique) - len(missing), "cache_misses": len(missing),
        "workers": telemetry,
        "gpu_hours": sum(worker["seconds"] for worker in telemetry) / 3600,
    }
