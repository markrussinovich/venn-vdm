"""Decision service and FastAPI application for the Venn HTTP API."""

import hashlib
import os
import threading
import time
import base64
import io
import json
import logging
import sqlite3
import copy
import uuid
import asyncio
from contextlib import closing
from pathlib import Path

import faiss
import numpy as np
import orjson
import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from torch.nn import functional as F

from venn.errors import ContractError
from venn.http_security import LocalHTTPGuard, allowed_host_set
from venn.index.density import knn_outlier_scores
from venn.outcomes import OutcomeService
from venn.profiles import TEXT_TAG, VENN_1, VectorTag
from venn.reader import MODEL_ID, REVISION, cache_key
from venn.index.apply import PreparedQuestion
from venn.index.shift import population_shift
from venn.heads.abstention import CosineAbstention
from venn.heads.calibration import fit_temperature
from venn.heads.family import ContextAbstention, FamilyScorer, FamilyQuestion, fit_family
from venn.heads.linear_choice import LinearChoiceHead, fit_linear_family
from venn.heads import regression
from venn.index.decision import leave_one_out_distances, nearest_decision
from venn.runtime import load_weights, model_directory, state_directory

logger = logging.getLogger("venn.server")
READER_MISSING = ("The EmbeddingGemma 2 reader is not downloaded. Run 'venn download-embedder' and retry; "
                  "vector states need no reader.")


class ORJSONResponse(JSONResponse):
    """JSON response serialized with orjson (NumPy-aware)."""

    def render(self, content):
        return orjson.dumps(content, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY)




class DecisionService:
    """Loads the pinned decision head and answers vector-native questions.

    Holds the embedding cache, registered indexes, prepared questions, calibrations and families, and
    raises ``ContractError`` for invalid input. ``embed`` may supply a custom text embedder; ``model_dir``
    overrides the head bundle location.
    """
    def __init__(self, embed=None, dimensions=768, cache_path=None, registry_path=None, model_dir=None,
                 outcome_service=None):
        torch.set_num_threads(1)
        self._thread_lock = threading.RLock()  # re-entrant: registration holds it while embedding option texts
        self.last_batch_threads = None
        assets = model_directory(model_dir)
        config = json.loads((assets / "config.json").read_text())
        if config["reader"] != {"id": MODEL_ID, "revision": REVISION, "dimensions": 768}:
            raise ValueError("Venn heads require the pinned 768-dimensional reader")
        baseline = config["baseline"]
        self.head = CosineAbstention(baseline["temperature"], baseline["none_logit"]).eval()
        self.head.load_state_dict(load_weights(assets / "universal.safetensors"))
        self.registration_universal = copy.deepcopy(self.head)
        self.generic = config["choice_temperature"]
        self.embedder, self.dimensions = embed, dimensions
        self._reader_lock = threading.Lock()  # one text-reader load, shared by the background preload and first use
        self.reader_state = "idle" if embed is None else "ready"  # idle, loading, ready, absent or error
        self.prepared, self.calibrations, self.indexes, self.questions = {}, {}, {}, {}
        self.cache_path = Path(cache_path or state_directory() / "cache.sqlite")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.cache_path)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS vectors(key TEXT PRIMARY KEY,value BLOB)")
        self.embed_count, self.prewarm_count = 0, 0
        self.media_model = None
        self.outcome_service = outcome_service or OutcomeService()
        self.store = state_directory() / "registry"
        self.store.mkdir(parents=True, exist_ok=True)
        self.registry_path = Path(registry_path or self.store / "calibrations.json")
        if self.registry_path.exists():
            self.calibrations = json.loads(self.registry_path.read_text())
        self.family_store = self.registry_path.parent / "families"
        self.family_store.mkdir(exist_ok=True)
        self.families = {}
        for metadata in self.family_store.glob("*.json"):
            entry = json.loads(metadata.read_text())
            weights_path = self.family_store / entry.get("weights", metadata.stem)
            if entry.get("profile", "venn-1") != "venn-1" or entry.get("revision") != REVISION:
                raise ValueError("Registered family profile mismatch; re-embed and re-register")
            if entry.get("objective") == "regress" and entry.get("extended"):
                weights = np.load(weights_path.with_suffix(".npz"))
                head = {**entry["head_scalars"], **{k: weights[k] for k in regression.HEAD_ARRAYS if k in weights}}
                head.setdefault("extra_mean", None)
                if regression.extended_checksum(head) != entry["report"]["checksum"]:
                    raise ValueError("Registered regression family checksum mismatch")
                entry["head"] = head
                self.families[entry["name"]] = entry
                continue
            if entry.get("objective") == "regress":
                weights = np.load(weights_path.with_suffix(".npz"))
                entry["head"] = {"coef": weights["coef"], "intercept": float(weights["intercept"]), "offsets": entry["report"]["interval"]["offsets"],
                                 "calibrated": entry["report"]["calibrated"]}
                if regression.checksum(entry["head"]["coef"], entry["head"]["intercept"]) != entry["report"]["checksum"]:
                    raise ValueError("Registered regression family checksum mismatch")
                self.families[entry["name"]] = entry
                continue
            if entry.get("form") == "linear":
                weights = np.load(weights_path.with_suffix(".npz"))
                entry["head"] = LinearChoiceHead(weights["W"], weights["b"], np.array(entry["options"], dtype=np.float32)).eval()
                entry["universal"] = None
                self.families[entry["name"]] = entry
                continue
            scorer = FamilyScorer(entry["examples"], dimensions, fields=entry.get("fields", False)).eval()
            scorer.load_state_dict(torch.load(weights_path.with_suffix(".pt"), weights_only=True, map_location="cpu"))
            entry["head"] = scorer
            if entry.get("universal_class") == "ContextAbstention":
                entry["universal"] = ContextAbstention(baseline["temperature"], baseline["none_logit"] / baseline["temperature"], dimensions).eval()
            else:
                entry["universal"] = CosineAbstention(baseline["temperature"], baseline["none_logit"], dimensions=dimensions).eval()
            entry["universal"].load_state_dict(torch.load(weights_path.with_suffix(".universal.pt"), weights_only=True, map_location="cpu"))
            self.families[entry["name"]] = entry

    def embed(self, texts, prompt="Classification"):
        """Return unit-norm float32 vectors for ``texts``, reading and filling the SQLite cache.

        ``prompt`` is one of the reader's documented task prompts in ``venn.reader.PROMPTS`` (default Classification,
        the ``venn-1`` profile's). Loads the pinned reader on CPU at the first miss unless an embedder was supplied; a
        supplied embedder must accept ``prompt_name=`` to embed with a non-default prompt. Raises ``ContractError``
        for empty or over-long text, an unknown prompt or an invalid embedding.
        """
        from venn.reader import PROMPTS
        if prompt not in PROMPTS:
            raise ContractError("embed_prompt", f"Use one of the reader's prompts: {', '.join(PROMPTS)}")
        for text in texts:
            if not isinstance(text, str) or not text or len(text) > 32768:
                raise ContractError("invalid_text", "Text must contain 1..32768 characters")
        keys = [cache_key(text, prompt) for text in texts]
        found = {}
        with closing(sqlite3.connect(self.cache_path)) as db, db:
            for key in set(keys):
                row = db.execute("SELECT value FROM vectors WHERE key=?", (key,)).fetchone()
                if row:
                    found[key] = np.frombuffer(row[0], dtype=np.float16).astype(np.float32)
            misses = list(dict.fromkeys(key for key in keys if key not in found))
            if misses:
                if self.embedder is None:
                    self.load_reader()
                text_of = dict(zip(keys, texts))
                with self.batch_threads(len(misses)):
                    self._embed_misses(db, misses, text_of, found, prompt)
        values = [found[key] / max(np.linalg.norm(found[key]), 1e-12) for key in keys]
        return np.array(values, dtype=np.float32)

    @staticmethod
    def _require_downloaded_reader():
        """Raise ``ContractError`` (503 ``reader_missing``) unless the pinned reader is in the local Hugging Face cache."""
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)
        except LocalEntryNotFoundError:
            raise ContractError("reader_missing", READER_MISSING, 503) from None

    def _load_text_reader(self):
        """Load the pinned text reader on CPU from the local cache and return an embedder callable."""
        self._require_downloaded_reader()
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_ID, revision=REVISION, device="cpu",
            config_kwargs={"vision_config": None, "audio_config": None},
            model_kwargs={"dtype": torch.float32}, local_files_only=True)
        return lambda xs, prompt_name="Classification": model.encode(xs, prompt_name=prompt_name, normalize_embeddings=False, batch_size=32)

    def load_reader(self, warm=False):
        """Load the text reader once and return the embedder; a caller arriving mid-load waits for that load.

        ``warm`` also encodes one short text before releasing waiting callers, so the first real encode is not slower.

        Raises ``ContractError`` (503 ``reader_missing``) when the reader is not downloaded; the next call checks again,
        so ``venn download-embedder`` takes effect without a restart.
        """
        with self._reader_lock:
            if self.embedder is None:
                self.reader_state = "loading"
                try:
                    embedder = self._load_text_reader()
                    if warm:
                        embedder(["warm-up"])
                    self.embedder = embedder
                except ContractError:
                    self.reader_state = "absent"
                    raise
                except Exception:
                    self.reader_state = "error"
                    raise
                self.reader_state = "ready"
            return self.embedder

    def preload_reader(self):
        """Start a daemon thread that loads the text reader, so the first text request need not; returns the thread.

        A missing reader is recorded in ``reader_state`` and logged once; requests then get the ``reader_missing`` error.
        """
        def run():
            try:
                self.load_reader(warm=True)
            except ContractError:
                logger.warning("Text reader not preloaded: %s", READER_MISSING)
            except Exception:
                logger.exception("Text reader preload failed; text requests will retry the load")

        thread = threading.Thread(target=run, name="venn-reader-preload", daemon=True)
        thread.start()
        return thread

    @staticmethod
    def default_batch_threads(cap=24):
        available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        return max(1, min(available, cap))

    def batch_threads(self, rows):
        """Let a large embedding batch use many cores, then restore the one-thread serving pin.

        ``VENN_EMBED_THREADS`` sets the batch thread count. The default is the CPUs this process may use, capped at
        24: on a 2 x 48-core host the reader embedded 128 notes in 47 s on 1 thread, 4.8 s on 24, 4.9 s on 48 and
        71 s on 96, because threads that span both sockets stall on each other. Batches of fewer than
        32 texts keep the serving setting, so single requests behave as before.
        """
        import contextlib

        if rows < 32:
            return contextlib.nullcontext()

        @contextlib.contextmanager
        def raised():
            previous = torch.get_num_threads()
            with self._thread_lock:
                torch.set_num_threads(int(os.environ.get("VENN_EMBED_THREADS", 0)) or self.default_batch_threads())
                self.last_batch_threads = torch.get_num_threads()
                try:
                    yield
                finally:
                    torch.set_num_threads(previous)
        return raised()

    def _embed_misses(self, db, misses, text_of, found, prompt="Classification"):
        """Embed cache misses in batches of 256, normalise in float64, store as float16 and fill ``found``."""
        for start in range(0, len(misses), 256):
            batch = misses[start:start + 256]
            texts = [text_of[k] for k in batch]
            if prompt == "Classification":
                computed = self.embedder(texts)
            else:
                try:
                    computed = self.embedder(texts, prompt_name=prompt)
                except TypeError:
                    raise ContractError("embed_prompt", "This embedder does not accept prompt_name; it embeds with the default prompt only")
            computed = np.asarray(computed, dtype=np.float64)
            if computed.shape != (len(batch), self.dimensions) or not np.isfinite(computed).all():
                raise ContractError("invalid_embedding", "Embedder returned invalid vector")
            # normalise in float64 before float16 storage, whatever precision the reader ran in
            computed /= np.maximum(np.linalg.norm(computed, axis=1, keepdims=True), 1e-12)
            for key, value in zip(batch, computed.astype(np.float16)):
                db.execute("INSERT OR IGNORE INTO vectors VALUES (?,?)", (key, value.tobytes()))
                found[key] = value.astype(np.float32)
            db.commit()
            self.embed_count += len(batch)

    def vector(self, value, declared_revision=REVISION, max_rows=512):
        """Resolve ``value`` to a finite float32 vector or ``(rows, dim)`` matrix.

        Accepts raw text, ``{"text"}``, ``{"image"|"audio"}`` media, ``{"vector"|"vectors", "revision"}`` or a
        numeric array, and the plural forms ``{"texts"|"images"|"audios": [...]}``, which embed in bulk and return
        a matrix. Raises ``ContractError`` for a revision mismatch, wrong shape, nonfinite or zero vectors, or more
        than ``max_rows`` rows.
        """
        if isinstance(value, str):
            if declared_revision != REVISION:
                raise ContractError("embedder_revision", "Raw text uses only the pinned model")
            return self.embed([value])[0]
        revision = declared_revision
        if isinstance(value, dict):
            plural = [key for key in ("texts", "images", "audios") if key in value]
            if plural:
                if len(plural) > 1 or any(k in value for k in ("text", "image", "audio", "vector", "vectors")):
                    raise ContractError("state_shape", "Provide one kind of state")
                items = value[plural[0]]
                if value.get("revision", declared_revision) != REVISION:
                    raise ContractError("embedder_revision", "Raw text and media use only the pinned model")
                if not isinstance(items, list) or not 1 <= len(items) <= max_rows:
                    raise ContractError("state_limit", f"Provide 1..{max_rows} {plural[0]}")
                if plural[0] == "texts":
                    return self.embed(items)
                return self.embed_media(items, "image" if plural[0] == "images" else "audio")
            if "image" in value or "audio" in value:
                if value.get("revision", declared_revision) != REVISION:
                    raise ContractError("embedder_revision", "Raw media uses only the pinned model")
                return self.media(value)
            if "text" in value:
                if value.get("revision", declared_revision) != REVISION:
                    raise ContractError("embedder_revision", "Raw text is embedded only by the pinned model")
                return self.embed([value["text"]])[0]
            revision = value.get("revision", revision)
            value = value.get("vector", value.get("vectors"))
        try:
            result = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError):
            raise ContractError("invalid_vector", "Vector must be numeric")
        if revision != REVISION:
            raise ContractError("embedder_revision", "Use the pinned reader revision; re-embed incompatible vectors")
        if result.ndim not in (1, 2) or result.shape[-1] != self.dimensions or not np.isfinite(result).all():
            raise ContractError("invalid_vector", f"Expected finite {self.dimensions}-dimensional vectors")
        if result.ndim == 2 and not 1 <= len(result) <= max_rows:
            raise ContractError("state_limit", f"At most {max_rows} state vectors")
        if np.any(np.linalg.norm(result, axis=-1) <= 1e-12):
            raise ContractError("zero_vector", "Zero vectors are not valid embeddings")
        return result

    def media(self, value):
        """Embed one base64 ``image`` or ``audio`` payload with the pinned reader and return a unit-norm vector.

        Payloads are capped at 6 MB; images at 4M pixels; audio must be mono 16 kHz WAV of at most 30
        seconds. Results are cached. Raises ``ContractError`` for invalid media. See ``embed_media`` for many at once.
        """
        if "image" in value and "audio" in value:
            raise ContractError("media_format", "Provide one raw modality")
        if value.get("revision", REVISION) != REVISION:
            raise ContractError("embedder_revision", "Raw media uses only the pinned model")
        kind = "image" if "image" in value else "audio"
        return self.embed_media([value[kind]], kind)[0]

    MEDIA_BATCH = {"image": 32, "audio": 8}

    def _decode_media(self, raw, kind):
        """Validate one decoded payload and return the reader's input item; raises ``ContractError`` if invalid."""
        if kind == "image":
            from PIL import Image
            try:
                image = Image.open(io.BytesIO(raw))
                if image.width * image.height > 4000000:
                    raise ContractError("image_size", "Image exceeds 4M pixels")
                image = image.convert("RGB")
            except (OSError, ValueError, Image.DecompressionBombError):
                raise ContractError("image_encoding", "Invalid image")
            return {"text": "<|image|>", "image": [image]}
        import soundfile as sf
        try:
            with sf.SoundFile(io.BytesIO(raw)) as audio:
                rate = audio.samplerate
                if rate != 16000 or audio.channels != 1 or not 1 <= audio.frames <= 480000:
                    raise ContractError("audio_format", "Audio must be mono 16 kHz and at most 30 seconds")
                samples = audio.read(dtype="float32")
        except (RuntimeError, OSError):
            raise ContractError("audio_encoding", "Invalid WAV audio")
        if rate != 16000 or samples.ndim != 1 or not 1 <= len(samples) <= 480000 or not np.isfinite(samples).all():
            raise ContractError("audio_format", "Audio must be mono 16 kHz and at most 30 seconds")
        return {"text": "<|audio|>", "audio": [samples]}

    def embed_media(self, payloads, kind="image", batch_size=None):
        """Embed many base64 ``image`` (or ``audio``) payloads and return a ``(n, 768)`` array of unit vectors.

        Each payload gets the same checks and cache key as ``media``. Cache hits are read in one pass; misses are
        decoded, embedded by the pinned reader in batches of ``batch_size`` (default 32 images or 8 audio clips) with
        the batch thread count, and written in one transaction. Raises ``ContractError`` for any invalid payload, before
        anything is embedded.
        """
        if kind not in ("image", "audio"):
            raise ContractError("media_format", "kind must be image or audio")
        if not isinstance(payloads, (list, tuple)) or not payloads:
            raise ContractError("media_format", "Provide a non-empty list of base64 payloads")
        raws, keys = [], []
        for payload in payloads:
            try:
                raw = base64.b64decode(payload, validate=True)
            except (TypeError, ValueError):
                raise ContractError("media_encoding", "Media must be base64 encoded")
            if not 0 < len(raw) <= 6000000:
                raise ContractError("media_size", "Media exceeds 6 MB", 413)
            raws.append(raw)
            keys.append(hashlib.sha256(json.dumps([MODEL_ID, REVISION, "full", kind]).encode() + raw).hexdigest())
        found = {}
        with closing(sqlite3.connect(self.cache_path)) as db, db:
            for key in set(keys):
                row = db.execute("SELECT value FROM vectors WHERE key=?", (key,)).fetchone()
                if row:
                    found[key] = np.frombuffer(row[0], dtype=np.float16).astype(np.float32)
            misses = list(dict.fromkeys(k for k in keys if k not in found))
            if misses:
                raw_of = dict(zip(keys, raws))
                items = {k: self._decode_media(raw_of[k], kind) for k in misses}  # validate everything first
                if self.media_model is None:
                    self._require_downloaded_reader()
                    from sentence_transformers import SentenceTransformer
                    self.media_model = SentenceTransformer(MODEL_ID, revision=REVISION, device="cpu",
                        model_kwargs={"dtype": torch.float32}, local_files_only=True)
                size = batch_size or self.MEDIA_BATCH[kind]
                rows = []
                with self.batch_threads(len(misses)):
                    for start in range(0, len(misses), size):
                        batch = misses[start:start + size]
                        vectors = np.asarray(self.media_model.encode([items[k] for k in batch], batch_size=len(batch),
                                                                     normalize_embeddings=True), dtype=np.float16)
                        if vectors.shape != (len(batch), 768) or not np.isfinite(vectors).all() or \
                                (np.linalg.norm(vectors.astype(np.float32), axis=1) <= 0).any():
                            raise ContractError("media_embedding", "Invalid media embedding")
                        for key, vector in zip(batch, vectors):
                            found[key] = vector.astype(np.float32)
                            rows.append((key, vector.tobytes()))
                db.executemany("INSERT OR IGNORE INTO vectors VALUES(?,?)", rows)
                db.commit()
                self.embed_count += len(misses)
        out = np.array([found[k] for k in keys], dtype=np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def register_index(self, vectors, ids, outcomes=None):
        """Register unit-normalized ``vectors`` with distinct ``ids`` (and optional ``outcomes``) in a FAISS index.

        Returns the index handle; raises ``ContractError`` if IDs repeat or counts do not match.
        """
        values = self.vector(vectors, max_rows=1000000)
        if values.ndim != 2 or len(ids) != len(values) or len(set(map(str, ids))) != len(ids):
            raise ContractError("invalid_index", "Distinct IDs must match index rows")
        if outcomes is not None and len(outcomes) != len(ids):
            raise ContractError("invalid_index", "Outcomes must match rows")
        values = values / np.linalg.norm(values, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(values.shape[1])
        index.add(np.ascontiguousarray(values))
        handle = hashlib.sha256(values.tobytes() + json.dumps([ids, outcomes]).encode()).hexdigest()
        self.indexes[handle] = {"vectors": values, "ids": list(ids), "outcomes": outcomes, "index": index,
                               "revision": REVISION, "profile": "venn-1"}
        return handle

    def register_outcome_store(self, adapter):
        """Register an app-owned outcome store adapter and return its handle."""
        return self.outcome_service.register_store(adapter)

    def decide_batch(self, states, questions, **kwargs):
        """Answer ``nearest_outcome`` questions for a batch of states through the outcome service.

        Re-raises the outcome service's ``ValueError`` as ``ContractError``.
        """
        try:
            return self.outcome_service.decide_batch(states, questions, **kwargs)
        except ValueError as error:
            raise ContractError("outcome_request", str(error)) from error

    def index(self, handle):
        """Return the registered index for ``handle``; raises ``ContractError`` (404) if it is unknown."""
        if handle not in self.indexes:
            raise ContractError("index_handle", "Missing or stale index handle", 404)
        return self.indexes[handle]

    def register_calibration(self, family, logits, targets, kind="Choice"):
        """Fit and persist a softmax temperature for ``family``/``kind`` from 200..10000 labeled logit rows.

        Returns the temperature; raises ``ContractError`` for wrong sizes, nonfinite logits or bad labels.
        """
        values, labels = np.asarray(logits, dtype=np.float32), np.asarray(targets)
        if values.ndim != 2 or not 200 <= len(values) <= 10000 or len(labels) != len(values):
            raise ContractError("calibration_size", "Register 200..10000 labeled logits")
        if not np.isfinite(values).all() or labels.ndim != 1 or not np.all(labels == labels.astype(int)) or not np.all((labels >= 0) & (labels < values.shape[1])):
            raise ContractError("calibration_labels", "Invalid calibration labels or logits")
        key = family + "/" + kind
        self.calibrations[key] = fit_temperature(values, labels.astype(int))
        self.registry_path.write_text(json.dumps(self.calibrations, indent=2) + "\n")
        return self.calibrations[key]

    def delete_calibration(self, family, kind="Choice"):
        """Remove the stored temperature for ``family``/``kind``, if any, and persist the registry."""
        self.calibrations.pop(family + "/" + kind, None)
        self.registry_path.write_text(json.dumps(self.calibrations, indent=2) + "\n")

    def register_family(self, family, examples=None, options=None, instructions=None, seed=0, *, objective="choice", target=None,
                        from_store=None, form=None, split=None, cv=None, alphas=None, class_weight=None, prompt=None, inputs=None):
        """Train and persist a family head.

        ``objective="choice"`` (default) trains a Choice head from labeled ``examples`` over labeled option vectors
        (see ``_register_choice``). ``objective="regress"`` fits a numeric predictor (see ``_register_regression``).
        Both take an optional caller ``split`` and ``cv`` settings; ``class_weight`` applies to choice families. ``form`` is
        ``"ridge"`` for regression and ``"residual"`` (default) or ``"linear"`` for choice families.
        Replacement validates existing question handles before atomically switching the saved family manifest.
        """
        if objective == "regress":
            return self._register_regression(family, examples=examples, target=target, from_store=from_store, form=form or "ridge",
                                             split=split, cv=cv, alphas=alphas, seed=seed, prompt=prompt, inputs=inputs)
        if objective != "choice":
            raise ContractError("family_objective", "Use objective choice or regress")
        if examples is None or options is None or instructions is None:
            raise ContractError("family_fields", "Choice families need examples, options and instructions")
        if form not in (None, "residual", "linear"):
            raise ContractError("family_form", "Choice families use form residual or linear")
        with self.batch_threads(len(examples) if isinstance(examples, list) else 0):  # a batch job: use many cores
            return self._register_choice(family, examples, options, instructions, seed, split=split, cv=cv, class_weight=class_weight,
                                         form=form or "residual")

    def _register_regression(self, family, *, examples=None, target=None, from_store=None, form="ridge", split=None, cv=None,
                             alphas=None, seed=0, prompt=None, inputs=None):
        """Fit a ridge regression family from numeric targets and persist it.

        Rows come from ``examples`` (each ``{"state": vector or text, <target>: number}``) or ``from_store``: an index
        handle registered with numeric outcomes, or an outcome-store handle whose ``export()`` partition named by
        ``target`` supplies (vectors, outcomes). ``split`` may give ``{"train", "selection", "report"}`` row indices;
        otherwise a seeded 60/20/20 split by target is used. Alpha is chosen by ``cv`` (folds x repeats, default 5 x 3)
        inside the train slice. ``form="kernel_ridge"`` fits an RBF kernel ridge instead, choosing alpha and gamma by
        the same cross-validation. ``prompt`` embeds text states with another of the reader's documented prompts, and
        ``inputs`` (``["log_words", "log_chars"]``) appends numeric inputs computed from the text, their scale chosen by
        the same cross-validation; both need text states, and the family then answers from text only. Returns the
        training report. Raises ``ContractError`` for invalid input.
        """
        started = time.perf_counter()
        if not isinstance(family, str) or not 1 <= len(family) <= 128:
            raise ContractError("family_name", "Family name must contain 1..128 characters")
        from venn.reader import PROMPTS
        if form not in ("ridge", "kernel_ridge"):
            raise ContractError("family_form", "Regression families support form ridge or kernel_ridge")
        if prompt is not None and prompt not in PROMPTS:
            raise ContractError("embed_prompt", f"Use one of the reader's prompts: {', '.join(PROMPTS)}")
        inputs = list(inputs or [])
        if any(name not in regression.INPUTS for name in inputs) or len(set(inputs)) != len(inputs):
            raise ContractError("family_inputs", f"inputs must be distinct names from {', '.join(regression.INPUTS)}")
        text_only = prompt not in (None, "Classification") or bool(inputs)
        if text_only and from_store is not None:
            raise ContractError("family_inputs", "prompt and inputs need text states; register from examples")
        if (examples is None) == (from_store is None):
            raise ContractError("family_examples", "Provide either examples or from_store")
        tag = TEXT_TAG
        if examples is not None:
            if not isinstance(target, str) or not target:
                raise ContractError("family_target", "Name the numeric target field")
            if not isinstance(examples, list) or not 10 <= len(examples) <= 1000000:
                raise ContractError("family_examples", "Register 10..1000000 examples")
            try:
                texts = None
                if text_only:
                    texts = [self.state_text(example["state"]) for example in examples]
                    values = np.asarray(self.embed(texts, prompt or "Classification"), dtype=np.float64)
                else:
                    values = np.array([self.vector(example["state"], max_rows=1) for example in examples], dtype=np.float64)
                y = np.array([float(example[target]) for example in examples], dtype=np.float64)
            except (KeyError, TypeError, ValueError) as exc:
                raise ContractError("family_examples", f"Each example needs a state and a numeric {target!r}: {exc}")
            source = {"kind": "examples", "rows": len(y), "target": target}
        else:
            if from_store in self.indexes:
                store = self.indexes[from_store]
                if store["outcomes"] is None:
                    raise ContractError("family_store", "The index has no outcomes")
                try:
                    y = np.array([float(v) for v in store["outcomes"]], dtype=np.float64)
                except (TypeError, ValueError):
                    raise ContractError("family_store", "Index outcomes must be numeric for a regression family")
                values = np.asarray(store["vectors"], dtype=np.float64)
                source = {"kind": "index", "handle": from_store, "rows": len(y), "target": "outcome"}
            elif from_store in self.outcome_service.stores:
                adapter = self.outcome_service.stores[from_store]
                tag = VectorTag.parse(adapter.tag)
                if tag.profile != VENN_1.id or tag.revision != REVISION or tag.dimensions != self.dimensions:
                    raise ContractError("embedder_revision", "The store's vectors are not from the venn-1 profile; re-embed them")
                export = getattr(adapter, "export", None)
                if not callable(export):
                    raise ContractError("family_store", "The outcome store adapter cannot export its cases")
                exported = export()
                if target not in exported:
                    raise ContractError("family_target", f"The store has no partition {target!r}")
                vectors_, outcomes = exported[target]
                values = np.asarray(vectors_, dtype=np.float64)
                y = np.asarray(outcomes, dtype=np.float64)
                source = {"kind": "outcome_store", "handle": from_store, "rows": len(y), "target": target, "tag": tag.to_dict()}
            else:
                raise ContractError("family_store", "Unknown index or outcome-store handle", 404)
        if values.ndim != 2 or values.shape[1] != self.dimensions or len(values) != len(y) or not np.isfinite(values).all() or not np.isfinite(y).all():
            raise ContractError("family_examples", "Rows must be finite single vectors with finite targets")
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        parts = None
        if split is not None:
            try:
                parts = tuple(np.asarray(split.get(key, []), dtype=np.int64) for key in ("train", "selection", "report"))
            except (AttributeError, TypeError, ValueError):
                raise ContractError("family_split", "Split must map train/selection/report to row indices")
            if any(len(p) and (p.min() < 0 or p.max() >= len(y)) for p in parts):
                raise ContractError("family_split", "Split indices are out of range")
        cv = cv or {}
        extended = form == "kernel_ridge" or bool(inputs)
        try:
            if extended:
                extra = regression.input_features(texts, inputs) if inputs else None
                head, report, parts = regression.register_extended(values, y, form=form, extra=extra, input_names=inputs, seed=seed, split=parts,
                                                                   folds=int(cv.get("folds", 5)), repeats=int(cv.get("repeats", 3)), alphas=alphas)
            else:
                head, report, parts = regression.register(values, y, seed=seed, split=parts, folds=int(cv.get("folds", 5)),
                                                          repeats=int(cv.get("repeats", 3)), alphas=tuple(alphas or regression.ALPHAS))
        except ValueError as exc:
            raise ContractError("family_examples", str(exc))
        report["seconds"] = time.perf_counter() - started
        report["source"] = source
        report["split_hashes"] = [[hashlib.sha256(values[i].tobytes()).hexdigest() for i in part] for part in parts]
        entry = {"name": family, "objective": "regress", "form": form, "report": report, "tag": tag.to_dict(),
                 "revision": REVISION, "profile": "venn-1", "fields": False, "head": head, "extended": extended,
                 "prompt": prompt or "Classification", "inputs": inputs, "text_only": text_only}
        if extended:
            arrays = {k: np.asarray(head[k], dtype=np.float64) for k in regression.HEAD_ARRAYS if head.get(k) is not None}
            entry["head_scalars"] = {k: head.get(k) for k in regression.HEAD_SCALARS}
        else:
            arrays = {"coef": head["coef"], "intercept": np.float64(head["intercept"])}
        self._commit_family(entry, arrays)
        return {k: v for k, v in report.items() if k != "split_hashes"}

    @staticmethod
    def state_text(state):
        """The text of a state given as a string or ``{"text": ...}``; raises ``ContractError`` otherwise."""
        if isinstance(state, str):
            return state
        if isinstance(state, dict) and isinstance(state.get("text"), str):
            return state["text"]
        raise ContractError("family_state", "This family answers from text; send the text so Venn can embed it with the family's prompt")

    def regression_answers(self, entry, values, texts=None):
        """Score rows of ``values`` (or of ``texts``, for families that answer from text) with a regression family:
        score, 80% interval, calibration flag."""
        head = entry["head"]
        if entry.get("text_only"):
            if texts is None:
                raise ContractError("family_state", "This family answers from text; send the text so Venn can embed it with the family's prompt")
            values = self.embed(texts, entry.get("prompt", "Classification"))
        values = np.atleast_2d(np.asarray(values, dtype=np.float64))
        if values.shape[1] != self.dimensions:
            raise ContractError("family_state", "Vectors do not match the family's profile")
        values = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        if entry.get("extended"):
            extra = regression.input_features(texts, entry["inputs"]) if entry.get("inputs") else None
            scores = regression.predict_head(head, values, extra)
        else:
            scores = regression.predict(head["coef"], head["intercept"], values)
        lo, hi = head["offsets"]
        return [{"score": float(s), "interval": [float(s + lo), float(s + hi)], "interval_level": regression.LEVEL,
                 "calibrated": bool(head["calibrated"]), "source": "family_head"} for s in scores]

    def _register_choice(self, family, examples, options, instructions, seed=0, *, split=None, cv=None, class_weight=None, form="residual"):
        """Train and persist a family head from labeled ``examples`` over labeled option vectors.

        Each example is ``{"state", "label"}`` with an optional ``group_id``; needs 10..100000 distinct examples,
        2..256 options and at least five examples per label. The head trains on every example, or on the ``split``
        train slice when the caller gives ``{"train", "selection", "report"}`` row indices (a group may not span
        slices). Weight decay and epochs are chosen by ``cv["folds"]``-fold cross-validation (default 5) inside the
        train rows, grouped by ``group_id`` when given; ``cv["max_epochs"]`` caps the epochs tried (default 24). ``class_weight`` is ``None``, ``"balanced"`` or
        ``{label: weight}``. ``form="linear"`` fits a multinomial logistic regression on the state vectors instead of the
        residual head (penalty chosen by the same cross-validation; single-vector states only). Returns the training
        report; raises ``ContractError`` for invalid input.
        """
        if not isinstance(family, str) or not 1 <= len(family) <= 128:
            raise ContractError("family_name", "Family name must contain 1..128 characters")
        if not isinstance(examples, list) or not 10 <= len(examples) <= 100000:
            raise ContractError("family_examples", "Register 10..100000 distinct labeled examples")
        if not isinstance(options, list) or not 2 <= len(options) <= 256:
            raise ContractError("family_options", "Provide 2..256 labeled option vectors")
        try:
            labels = [str(option["label"]) for option in options]
            if len(set(labels)) != len(labels) or "none" in labels:
                raise ValueError("Option labels must be distinct and cannot be none")
            option_vectors = np.array([self.vector(option["vector"]) for option in options])
            lookup = {label: i for i, label in enumerate(labels)}
            values = np.array([self.vector(example["state"]) for example in examples])
            target = np.array([lookup[str(example["label"])] for example in examples])
            if np.bincount(target, minlength=len(labels)).min() < 5:
                raise ValueError("Each option label needs at least five distinct examples")
            if values.ndim not in (2, 3) or option_vectors.ndim != 2:
                raise ValueError("Use consistent single-vector or multi-vector states")
            values /= np.linalg.norm(values, axis=-1, keepdims=True)
            if len({state.tobytes() for state in values}) != len(values):
                raise ValueError("Repeated states cannot inflate registration size")
            q = self.vector(instructions)
            if q.ndim != 1:
                raise ValueError("Instructions must be one vector")
            parts = None
            groups = None
            if any("group_id" in example for example in examples):
                if not all(isinstance(example.get("group_id"), str) for example in examples):
                    raise ValueError("Provide a string group_id for every grouped example")
                groups = np.array([example["group_id"] for example in examples])
            if split is not None:
                if not isinstance(split, dict) or not set(split) <= {"train", "selection", "report"}:
                    raise ContractError("family_split", "Split must map train/selection/report to row indices")
                parts = [np.asarray(split.get(key, []), dtype=np.int64).reshape(-1) for key in ("train", "selection", "report")]
                together = np.concatenate(parts)
                if len(together) and (together.min() < 0 or together.max() >= len(values)):
                    raise ContractError("family_split", "Split indices are out of range")
                if len(set(together.tolist())) != len(together):
                    raise ContractError("family_split", "Split slices must not share rows")
                if groups is not None and len({g for part in parts for g in set(groups[part].tolist())}) != sum(len(set(groups[part].tolist())) for part in parts):
                    raise ContractError("family_split", "A group may not span split slices")
            if cv is not None and (not isinstance(cv, dict) or not set(cv) <= {"folds", "max_epochs"}
                                   or not isinstance(cv.get("folds", 5), int) or not 2 <= cv.get("folds", 5) <= 20
                                   or not isinstance(cv.get("max_epochs", 24), int) or not 1 <= cv.get("max_epochs", 24) <= 100):
                raise ContractError("family_cv", "cv must be {\"folds\": 2..20, \"max_epochs\": 1..100}")
            weights = None
            if class_weight is not None:
                if class_weight == "balanced":
                    weights = "balanced"
                elif isinstance(class_weight, dict) and all(str(k) in lookup and isinstance(v, (int, float)) and v > 0 for k, v in class_weight.items()):
                    weights = {lookup[str(k)]: float(v) for k, v in class_weight.items()}
                else:
                    raise ContractError("family_class_weight", "class_weight must be \"balanced\" or {label: positive weight}")
            arrays = None
            if form == "linear":
                model, info, parts, arrays = fit_linear_family(values, target, option_vectors, seed=seed, partitions=parts, groups=groups,
                                                               folds=(cv or {}).get("folds", 5), class_weight=weights)
            else:
                model, info, parts = fit_family(values, target, option_vectors, q,
                                                copy.deepcopy(self.registration_universal), seed=seed, partitions=parts, groups=groups,
                                                folds=(cv or {}).get("folds", 5), epochs=(cv or {}).get("max_epochs", 24), class_weight=weights)
            if weights is not None and weights != "balanced":
                info["class_weight"] = {labels[i]: w for i, w in weights.items()}
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("family_examples", str(exc))
        entry = {**info, "name": family, "labels": labels, "options": option_vectors.tolist(),
                 "question": q.tolist(), "fields": values.ndim == 3,
                 "universal_class": type(self.registration_universal).__name__,
                 "split_hashes": [[hashlib.sha256(values[i].tobytes()).hexdigest() for i in part] for part in parts],
                 "revision": REVISION, "profile": "venn-1",
                 "head": model, "universal": None if form == "linear" else copy.deepcopy(self.registration_universal), "form": form}
        self._commit_family(entry, arrays)
        return info

    def _commit_family(self, entry, arrays):
        family = entry["name"]
        rebound, cache = {}, {}
        for handle, (definition, _, _, _) in self.questions.items():
            if definition.get("family") == family:
                if entry.get("objective") == "regress":
                    raise ContractError("family_questions", "Prepared option questions cannot use a regression family; use a new family name")
                prepared, names, q, _, _ = self.prepared_question(
                    "", definition, definition.get("type", "Choice"), family_entry=entry, cache=cache)
                rebound[handle] = definition, prepared, names, q
        key = hashlib.sha256(family.encode()).hexdigest()
        metadata = self.family_store / f"{key}.json"
        previous = json.loads(metadata.read_text()) if metadata.exists() else {}
        base = self.family_store / f"{key}-{uuid.uuid4().hex}"
        entry["weights"] = base.name
        pending = base.with_suffix(".pending")
        paths = [base.with_suffix(suffix) for suffix in (".npz", ".pt", ".universal.pt")]
        try:
            if arrays is not None:
                np.savez(paths[0], **arrays)
            else:
                torch.save(entry["head"].state_dict(), paths[1])
                torch.save(entry["universal"].state_dict(), paths[2])
            pending.write_text(json.dumps({k: v for k, v in entry.items() if k not in ("head", "universal")}, indent=2))
            # Immutable weights first, then one atomic manifest switch; the old generation stays usable.
            os.replace(pending, metadata)
        except BaseException:
            for path in [pending, *paths]:
                path.unlink(missing_ok=True)
            raise
        self.families[family] = entry
        self.prepared.clear()
        self.prepared.update(cache)
        self.questions.update(rebound)
        self.prewarm_count += len(cache)
        old = self.family_store / previous.get("weights", key)
        for suffix in (".npz", ".pt", ".universal.pt"):
            try:
                old.with_suffix(suffix).unlink(missing_ok=True)
            except OSError:
                pass  # The new manifest is committed; old weights can be removed by delete_family.

    def delete_family(self, family):
        """Delete a registered family and its saved weights; raises ``ContractError`` (404) if it is not registered."""
        if family not in self.families:
            raise ContractError("family_name", "Family is not registered", 404)
        del self.families[family]
        metadata = self.family_store / f"{hashlib.sha256(family.encode()).hexdigest()}.json"
        key = metadata.stem
        for path in [metadata, *self.family_store.glob(f"{key}-*"),
                     metadata.with_suffix(".pt"), metadata.with_suffix(".universal.pt"), metadata.with_suffix(".npz")]:
            path.unlink(missing_ok=True)
        self.prepared.clear()
        for handle, (definition, _, _, _) in list(self.questions.items()):
            if definition.get("family") == family:
                prepared, names, q, _, _ = self.prepared_question("", definition, definition.get("type", "Choice"))
                self.questions[handle] = definition, prepared, names, q

    def prepared_question(self, name, question, kind, *, family_entry=None, cache=None):
        """Resolve ``question`` to a prepared scorer, caching by family, instructions, labels and options.

        Returns ``(prepared, labels, question_vector, temperature, calibrated)``; honours ``question_ref``
        handles and uses the family head when the family is registered. Raises ``ContractError`` for bad
        options or handles.
        """
        family = question.get("family", "")
        entry = family_entry if family_entry is not None else self.families.get(family)
        prepared_cache = self.prepared if cache is None else cache
        calibration_kind = "Choice" if kind == "IndexApply" else kind
        temperature = self.calibrations.get(family + "/" + calibration_kind, self.generic)
        calibrated = False
        if "question_ref" in question:
            handle = question["question_ref"]
            if handle not in self.questions:
                raise ContractError("question_handle", "Missing or stale question handle", 404)
            _, prepared, labels, q = self.questions[handle]
            if entry is not None and isinstance(prepared, FamilyQuestion):
                temperature = entry["temperature"]
                calibrated = bool(entry["calibrated"]) and kind in ("Choice", "IndexApply")
            return prepared, labels, q, temperature, calibrated
        instruction = question.get("instructions_vec", question.get("instructions", "What is the topic?"))
        q = self.vector(instruction)
        if q.ndim != 1:
            raise ContractError("instructions", "Instructions must be one vector")
        options = question.get("options", [])
        if not 1 <= len(options) <= 256:
            raise ContractError("option_limit", "Provide 1..256 options")
        vectors = np.array([self.vector(option if "vector" in option else option.get("label")) if isinstance(option, dict) else self.vector(option) for option in options])
        if vectors.ndim != 2:
            raise ContractError("options", "Each option must be one vector")
        labels = [str(option.get("label", i)) if isinstance(option, dict) else (option if isinstance(option, str) else str(i)) for i, option in enumerate(options)]
        if len(set(labels)) != len(labels) or "none" in labels:
            raise ContractError("option_labels", "Option labels must be distinct and cannot be none")
        if entry is not None:
            if any(label not in entry["labels"] for label in labels):
                raise ContractError("family_options", "Options must belong to the named family")
            registered = np.array(entry["options"], dtype=np.float32)[[entry["labels"].index(label) for label in labels]]
            if not np.allclose(vectors, registered, atol=1e-6):
                raise ContractError("family_options", "Re-register changed family option vectors")
        key = hashlib.sha256(family.encode() + json.dumps(labels).encode() + q.tobytes() + vectors.tobytes()).hexdigest()
        if key not in prepared_cache:
            if entry is not None and (len(labels) != 2 or not entry["k2_fallback"]):
                bound = entry
                if entry.get("form") == "linear":
                    # Labels, not embedding distances, identify rows when option vectors coincide.
                    indices = [entry["labels"].index(label) for label in labels]
                    head = entry["head"]
                    bound = {**entry, "head": LinearChoiceHead(head.W[indices].cpu().numpy(), head.b[indices].cpu().numpy(), vectors).eval()}
                prepared_cache[key] = FamilyQuestion(bound, q, vectors)
            else:
                prepared_cache[key] = PreparedQuestion(self.head, torch.tensor(q), torch.tensor(vectors))
            if cache is None:
                self.prewarm_count += 1
        if isinstance(prepared_cache[key], FamilyQuestion):
            temperature = entry["temperature"]
            calibrated = bool(entry["calibrated"]) and kind in ("Choice", "IndexApply")
        return prepared_cache[key], labels, q, temperature, calibrated

    def register_question(self, name, question):
        """Prepare a Choice, Score, Noul or IndexApply question and return a reusable ``question_ref`` handle.

        Raises ``ContractError`` for other question types.
        """
        kind = question.get("type", "Choice")
        if kind not in ("Choice", "Score", "Noul", "IndexApply"):
            raise ContractError("prewarm_type", "This type has no option prewarm handle")
        prepared, labels, q, _, _ = self.prepared_question(name, question, kind)
        definition = json.loads(json.dumps(question))
        definition.pop("threshold", None)
        handle = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
        self.questions[handle] = definition, prepared, labels, q
        return handle

    def stream_index(self, handle, question):
        """Return a generator of NDJSON lines applying ``question`` to every row of the index ``handle``.

        Each line is ``{"offset", "rows"}`` with one answer per row; honours ``row_cap``, ``batch_size`` and
        ``threshold``. Raises ``ContractError`` for bad limits or handles.
        """
        index = self.index(handle)
        if "question_ref" in question:
            ref = question["question_ref"]
            if ref not in self.questions:
                raise ContractError("question_handle", "Missing question handle", 404)
            question = {**self.questions[ref][0], **question}
        cap, batch = question.get("row_cap", len(index["vectors"])), question.get("batch_size", 65536)
        if type(cap) is not int or not 0 <= cap <= 1000000 or type(batch) is not int or not 1 <= batch <= 65536:
            raise ContractError("stream_limit", "Invalid row cap or batch size")
        threshold = question.get("threshold")
        if threshold is not None and (not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1):
            raise ContractError("threshold", "Threshold must be in [0,1]")
        entry = self.families.get(question.get("family", ""))
        if entry is not None and entry.get("objective") == "regress":
            if entry.get("text_only"):
                raise ContractError("family_state", "This family answers from text; index rows carry vectors only")
            def regression_batches():
                for offset in range(0, min(cap, len(index["vectors"])), batch):
                    rows = self.regression_answers(entry, index["vectors"][offset:min(offset + batch, cap)])
                    for i, row in enumerate(rows):
                        row["id"] = index["ids"][offset + i]
                    yield json.dumps({"offset": offset, "rows": rows}) + "\n"
            return regression_batches()
        prepared, labels, _, temperature, calibrated = self.prepared_question("stream", question, "IndexApply")
        if isinstance(prepared, FamilyQuestion) and prepared.entry["fields"]:
            raise ContractError("family_state", "Multi-vector families cannot consume single-vector index rows")
        def batches():
            for offset, p in prepared.stream(index["vectors"][:cap], batch, temperature=temperature):
                rows = []
                for i, distribution in enumerate(p):
                    row = self.answer(distribution, labels, calibrated, threshold)
                    row["id"] = index["ids"][offset + i]
                    rows.append(row)
                yield json.dumps({"offset": offset, "rows": rows}) + "\n"
        return batches()

    @torch.inference_mode()
    def decide(self, request):
        """Answer the named questions in ``request`` about its ``state``.

        Supports Choice, Score, Noul (yes/no), IndexApply, NearestReference, SetMembership, PopulationShift
        and nearest_outcome. Returns ``answers``, the state ``vector`` (when embedded on demand), ``vector_tag``,
        ``model`` and ``index_ref``. Raises ``ContractError`` for invalid requests.
        """
        if not isinstance(request, dict) or not isinstance(request.get("questions"), dict):
            raise ContractError("request_shape", "Provide state and named questions")
        if not 1 <= len(request["questions"]) <= 32:
            raise ContractError("question_limit", "Provide 1..32 questions")
        model = request.get("model", {"id": MODEL_ID, "revision": REVISION})
        revision = model.get("revision", REVISION) if isinstance(model, dict) else REVISION
        if isinstance(model, str) and model != MODEL_ID:
            raise ContractError("embedder_revision", f"Unknown model; use {MODEL_ID}")
        if isinstance(model, dict) and model.get("id", MODEL_ID) != MODEL_ID:
            raise ContractError("embedder_revision", "Unsupported reader; re-embed with the pinned venn-1 profile")
        if any(isinstance(question, dict) and question.get("type") == "nearest_outcome"
               for question in request["questions"].values()):
            if revision != REVISION:
                raise ContractError("embedder_revision", "Outcome stores require the pinned reader revision")
            result = self.decide_batch([request.get("state")], request["questions"])["results"][0]
            return {**result, "model": {"id": MODEL_ID, "revision": REVISION}, "index_ref": None}
        state = request.get("state")
        index = self.index(state["index_ref"]) if isinstance(state, dict) and "index_ref" in state else None
        texts = None
        if isinstance(state, str):
            texts = [state]
        elif isinstance(state, dict) and isinstance(state.get("text"), str):
            texts = [state["text"]]
        elif isinstance(state, dict) and isinstance(state.get("texts"), list):
            texts = state["texts"]
            if not 1 <= len(texts) <= 512 or not all(isinstance(t, str) for t in texts):
                raise ContractError("state_texts", "Provide 1..512 texts")
        values = index["vectors"] if index else self.vector(state, revision)
        answers = {}
        for name, question in request["questions"].items():
            if not isinstance(question, dict):
                raise ContractError("question_shape", "Each question must be an object")
            if "question_ref" in question:
                handle = question["question_ref"]
                if handle not in self.questions:
                    raise ContractError("question_handle", "Missing or stale question handle", 404)
                if set(question) - {"question_ref", "threshold", "row_cap", "batch_size"}:
                    raise ContractError("prewarm_override", "Re-register changed options or instructions")
                question = {**self.questions[handle][0], **question}
            if "family" not in question and request.get("family"):
                question = {**question, "family": request["family"]}
            kind = question.get("type", "Choice")
            threshold = question.get("threshold")
            if threshold is not None and (not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1):
                raise ContractError("threshold", "Threshold must be in [0,1]")
            if kind == "NearestReference":
                store = self.index(question.get("index_ref"))
                if store["outcomes"] is None or values.ndim != 1:
                    raise ContractError("nearest_input", "NearestReference needs one vector and labeled store")
                x = values / np.linalg.norm(values)
                scores = x[None] @ store["vectors"].T
                labels = np.asarray(store["outcomes"])
                if len(np.unique(labels)) < 2:
                    raise ContractError("nearest_labels", "Need at least two reference outcomes")
                predicted, margin = nearest_decision(scores, labels)
                family = question.get("family", "") + "/" + kind
                t = self.calibrations.get(family, self.generic)
                probability = float(torch.sigmoid(torch.tensor(float(margin[0]) / t)))
                k = question.get("k", 5)
                if not isinstance(k, int) or not 1 <= k <= 100:
                    raise ContractError("nearest_k", "Reference count must be in 1..100")
                top = np.argsort(scores[0])[::-1][:k]
                label = str(predicted[0])
                answers[name] = {"answer": label if threshold is None or probability >= threshold else "none",
                    "probabilities": {label: probability, "none": 1 - probability}, "confidence": probability,
                    "calibrated": False, "evidence_ids": [store["ids"][i] for i in top],
                    "margin": float(margin[0]), "none_semantics": "cannot affirm nearest outcome; not calibrated corpus-wide OOD"}
                continue
            if kind == "PopulationShift":
                if "batch_b" not in question:
                    raise ContractError("population_input", "Provide two populations")
                other = self.vector(question["batch_b"], revision)
                if values.ndim != 2 or other.ndim != 2 or min(len(values), len(other)) < 2:
                    raise ContractError("population_input", "Provide two populations")
                result = population_shift(values, other, bandwidth=question.get("bandwidth", 1), permutations=200)
                answers[name] = {**result, "answer": result["shifted"], "probabilities": None,
                                 "confidence": None, "calibrated": False, "p_value_is_posterior": False}
                continue
            if kind == "SetMembership":
                if "members" in question:
                    members = self.vector(question["members"])
                    if values.ndim != 1 or members.ndim != 2:
                        raise ContractError("set_input", "Provide candidate and members")
                    if threshold is not None:
                        raise ContractError("threshold", "Geometric membership has no probability threshold")
                    candidate = values / np.linalg.norm(values)
                    members = members / np.linalg.norm(members, axis=1, keepdims=True)
                    k = min(10, len(members) - 1)
                    if k < 1:
                        raise ContractError("set_input", "Membership needs at least two members")
                    distances = np.maximum(0, 1 - members @ members.T)
                    np.fill_diagonal(distances, np.inf)
                    reference = np.partition(distances, k - 1, axis=1)[:, k - 1]
                    distance = float(np.partition(1 - members @ candidate, k - 1)[k - 1])
                    cutoff = float(np.quantile(reference, .95))
                    answers[name] = {"answer": distance <= cutoff, "distance": distance,
                        "reference_cutoff": cutoff, "probabilities": None, "confidence": None,
                        "calibrated": False, "experimental": True,
                        "scope": "geometric k-NN density; not a posterior"}
                else:
                    if values.ndim != 2 or len(values) < 2:
                        raise ContractError("set_input", "Odd-one-out requires multiple vectors")
                    if question.get("method") == "knn5":
                        if threshold is not None:
                            raise ContractError("threshold", "Density ranking has no probability threshold")
                        count = question.get("outliers", 1)
                        if type(count) is not int or not 1 <= count < len(values):
                            raise ContractError("set_input", "Outlier count must be in 1..row_count-1")
                        try:
                            scores = knn_outlier_scores(values)
                        except ValueError as error:
                            raise ContractError("set_input", str(error))
                        top = np.argsort(scores)[-count:][::-1].tolist()
                        answers[name] = {"answer": top[0] if count == 1 else top,
                            "distances": scores.tolist(), "probabilities": None, "confidence": None,
                            "calibrated": False, "scope": "k-NN density ranking; not a posterior"}
                        continue
                    if question.get("method", "centroid") != "centroid":
                        raise ContractError("set_input", "Use centroid or knn5")
                    distance = leave_one_out_distances(values)
                    p = torch.tensor(distance).softmax(0).tolist()
                    answers[name] = {"answer": int(np.argmax(distance)), "probabilities": dict(enumerate(p)), "confidence": max(p), "calibrated": False, "distances": distance.tolist()}
                continue
            if kind not in ("Choice", "Score", "Noul", "IndexApply"):
                raise ContractError("question_type", f"Unsupported question type {kind}")
            regress = self.families.get(question.get("family", request.get("family", "")))
            if regress is not None and regress.get("objective") == "regress":
                if kind not in ("Score", "IndexApply"):
                    raise ContractError("family_kind", "A regression family answers Score questions")
                rows = self.regression_answers(regress, values, texts)
                single = values.ndim == 1
                if single:
                    answers[name] = rows[0]
                else:
                    if len(rows) > 512:
                        raise ContractError("stream_required", "Use /index-apply for responses above 512 rows", 413)
                    for i, row in enumerate(rows):
                        row["id"] = index["ids"][i] if index else i
                    answers[name] = {"rows": rows}
                continue
            prepared, labels, q, temperature, calibrated = self.prepared_question(name, question, kind)
            entry = self.families.get(question.get("family", request.get("family", "")))
            if kind == "IndexApply":
                if entry is not None and entry["fields"]:
                    raise ContractError("family_state", "Multi-vector families cannot consume single-vector index rows")
                if values.ndim != 2:
                    raise ContractError("index_input", "IndexApply requires multiple rows")
                cap = question.get("row_cap", len(values))
                batch = question.get("batch_size", 65536)
                if not isinstance(cap, int) or not 0 <= cap <= 1000000 or not isinstance(batch, int) or not 1 <= batch <= 65536:
                    raise ContractError("stream_limit", "Invalid row cap or batch size")
                if cap > 512:
                    raise ContractError("stream_required", "Use /index-apply for responses above 512 rows", 413)
                rows = []
                for offset, p in prepared.stream(values[:cap], batch, temperature=temperature):
                    for i, distribution in enumerate(p):
                        result = self.answer(distribution, labels, calibrated, threshold)
                        result["id"] = index["ids"][offset + i] if index else offset + i
                        rows.append(result)
                answers[name] = {"rows": rows, "batches": (len(rows) + batch - 1) // batch}
                continue
            state_vector = torch.tensor(values)
            if entry is not None:
                if state_vector.ndim == 2 and not entry["fields"]:
                    raise ContractError("family_state", "This family was registered with single-vector states")
                if state_vector.ndim == 1 and entry["fields"]:
                    raise ContractError("family_state", "This family requires multi-vector states")
                if entry["fields"] and not isinstance(prepared, FamilyQuestion):
                    state_vector = state_vector.mean(0)
                if not isinstance(prepared, FamilyQuestion):
                    state_vector = F.normalize(state_vector, dim=-1)
                logits = prepared.logits(state_vector[None])[0]
            elif state_vector.ndim == 2:
                raise ContractError("unsupported_input", f"{kind} answers one item; ask an IndexApply question to answer each row "
                                    "of a plural state, or name a registered field family for one multi-field item")
            if entry is None:
                state_vector = F.normalize(state_vector, dim=-1)
                logits = prepared.logits(state_vector[None])[0]
            p = (logits / temperature).softmax(-1)
            if kind == "Noul":
                none = torch.sigmoid((logits[-1] - torch.logsumexp(logits[:-1], 0)) / temperature)
                p = torch.cat([(logits[:-1] / self.generic).softmax(0) * (1 - none), none[None]])
            answer = self.answer(p, labels, calibrated, threshold)
            if kind == "Score":
                scores = question.get("scores", list(range(len(labels))))
                try:
                    scores = np.asarray(scores, dtype=np.float64)
                except (TypeError, ValueError):
                    raise ContractError("rubric", "Scores must be numeric")
                if scores.ndim != 1 or len(scores) != len(labels) or not np.isfinite(scores).all():
                    raise ContractError("rubric", "Finite scores must match ordered rubric")
                mass = float(p[:-1].double().sum())
                answer["expected_score"] = float((p[:-1].double() * torch.tensor(scores)).sum()) / mass if mass else None
                answer["score_conditioned_on_not_none"] = True
            if kind == "Noul":
                answer["answer"] = "none" if float(p[-1]) >= (threshold if threshold is not None else .5) else "in_scope"
            answers[name] = answer
        on_demand = isinstance(state, str) or (isinstance(state, dict) and any(k in state for k in ("text", "image", "audio", "texts", "images", "audios")))
        tag = TEXT_TAG
        if isinstance(state, dict) and ("image" in state or "images" in state):
            tag = VectorTag(VENN_1.id, REVISION, "image-default", 768, ("image",))
        elif isinstance(state, dict) and ("audio" in state or "audios" in state):
            tag = VectorTag(VENN_1.id, REVISION, "audio-default", 768, ("audio",))
        elif not on_demand:
            tag = VectorTag(VENN_1.id, REVISION, "untagged", values.shape[-1])
        return {"answers": answers, "vector": values.tolist() if on_demand else None,
                "vector_tag": tag.to_dict(), "model": {"id": MODEL_ID, "revision": REVISION},
                "index_ref": state["index_ref"] if index else None}

    @staticmethod
    def answer(p, labels, calibrated, threshold):
        """Turn probabilities over ``labels + ["none"]`` into an answer dict.

        Picks "none" when the best probability is below ``threshold``.
        """
        values = p.tolist()
        best = int(np.argmax(values))
        confidence = values[best]
        answer = (labels + ["none"])[best]
        if threshold is not None and confidence < threshold:
            answer = "none"
        return {"answer": answer, "probabilities": dict(zip(labels + ["none"], values)),
                "confidence": confidence, "calibrated": calibrated}


def create_app(service=None, site_dir=None, allowed_hosts=None):
    """Build the FastAPI app around ``service`` (a new ``DecisionService`` by default).

    ``ContractError`` becomes a JSON ``{"error": {"code", "message"}}`` response. ``site_dir`` mounts static
    documentation at ``/guide``. Hosts default to loopback names; ``allowed_hosts`` explicitly
    configures other hostnames. Browser requests must be same-origin. Service calls and stream batches are
    serialized in the ASGI runtime's worker pool; health and static routes do not wait for that work.
    """
    from venn.decisions import decide as compatible_decide

    allowed_host_set(allowed_hosts)  # fail here, not on the first request: Starlette builds middleware lazily
    app = FastAPI(default_response_class=ORJSONResponse)
    app.add_middleware(LocalHTTPGuard, allowed_hosts=allowed_hosts)
    app.state.service = service or DecisionService()
    work_lock = asyncio.Lock()

    async def call_service(function, *args, **kwargs):
        def invoke():
            # Keep service mutations and global torch thread settings serialized, off the event loop.
            with app.state.service._thread_lock:
                return function(*args, **kwargs)
        async with work_lock:
            return await run_in_threadpool(invoke)

    async def serialized_stream(stream):
        finished = object()
        while True:
            chunk = await call_service(next, stream, finished)
            if chunk is finished:
                return
            yield chunk

    @app.exception_handler(ContractError)
    async def typed_error(request, exc):
        """Render a ``ContractError`` as a JSON error response with its status code."""
        return ORJSONResponse({"error": {"code": exc.code, "message": exc.message}}, status_code=exc.status)

    async def read_request(request):
        if int(request.headers.get("content-length", 0)) > 8000000:
            raise ContractError("request_size", "Request exceeds 8MB", 413)
        body = await request.body()
        if len(body) > 8000000:
            raise ContractError("request_size", "Request exceeds 8MB", 413)
        try:
            value = json.loads(body)
        except ValueError:
            raise ContractError("request_json", "Invalid JSON")
        return value

    @app.post("/decide-batch")
    async def decide_batch(request: Request):
        """POST /decide-batch: answer ``nearest_outcome`` questions for tagged vector states or a raw image batch.

        Tagged states are sent as JSON; uint8 image batches as ``application/octet-stream`` with ``X-Venn-Batch``
        metadata (``shape``, ``questions``, ...) and the experimental media header.
        """
        if request.headers.get("content-type", "").split(";")[0] == "application/octet-stream":
            if request.headers.get("x-venn-experimental") != "media":
                raise ContractError("experimental_media", "Set X-Venn-Experimental: media to opt in", 400)
            try:
                metadata = json.loads(request.headers.get("x-venn-batch", ""))
                shape = metadata["shape"]
                if (not isinstance(shape, list) or len(shape) != 4
                        or any(type(size) is not int or size < 1 for size in shape)
                        or not 1 <= shape[0] <= 128 or shape[-1] != 3
                        or np.prod(shape, dtype=object) > 64000000):
                    raise ValueError("Invalid uint8 batch shape")
                body = bytearray()
                async for chunk in request.stream():
                    if len(body) + len(chunk) > 64000000:
                        raise ContractError("request_size", "Batch exceeds 64 MB", 413)
                    body.extend(chunk)
                if len(body) != int(np.prod(shape, dtype=object)):
                    raise ValueError("Body length must match the uint8 shape")
                states = np.frombuffer(body, dtype=np.uint8).reshape(shape)
            except (ValueError, KeyError, TypeError) as error:
                raise ContractError("outcome_request", str(error)) from error
        else:
            metadata = await read_request(request)
            if not isinstance(metadata, dict):
                raise ContractError("outcome_request", "Provide states and questions")
            states = metadata.get("states")
        if metadata.get("model", "venn-1") != "venn-1":
            raise ContractError("decision_model", "Use venn-1", 400)
        return ORJSONResponse(await call_service(app.state.service.decide_batch,
            states, metadata.get("questions"),
            representation=metadata.get("representation", "image-thirds-70"),
            epsilon=metadata.get("epsilon")))

    @app.post("/api/alpha/decisions")
    async def decisions(request: Request):
        """POST /api/alpha/decisions: answer a Decisions API request with choice, score, noul and Venn extension questions."""
        result = await call_service(compatible_decide,
            app.state.service, await read_request(request),
            experimental_media=request.headers.get("x-venn-experimental") == "media",
        )
        if request.headers.get("x-venn-return-vector", "").lower() not in ("1", "true"):
            result["venn"].pop("vector", None)  # 768 floats; opt in with X-Venn-Return-Vector: 1
        return ORJSONResponse(result)

    @app.put("/families/{family}")
    async def register_family(family: str, value: dict):
        """PUT /families/{family}: train a family head from ``examples``, ``options`` and ``instructions_vec``.

        Returns the training report.
        """
        if value.get("objective") == "regress":
            return await call_service(app.state.service.register_family, family, value.get("examples"), seed=value.get("seed", 0), objective="regress",
                                                     target=value.get("target"), from_store=value.get("from_store"),
                                                     form=value.get("form", "ridge"), split=value.get("split"), cv=value.get("cv"),
                                                     prompt=value.get("prompt"), inputs=value.get("inputs"))
        if not {"examples", "options", "instructions_vec"}.issubset(value):
            raise ContractError("family_fields", "Provide examples, options and instructions_vec")
        return await call_service(app.state.service.register_family, family, value["examples"], value["options"],
                                                 value["instructions_vec"], value.get("seed", 0), split=value.get("split"),
                                                 cv=value.get("cv"), class_weight=value.get("class_weight"), form=value.get("form"))

    @app.delete("/families/{family}")
    async def delete_family(family: str):
        """DELETE /families/{family}: remove a registered family."""
        await call_service(app.state.service.delete_family, family)
        return {"deleted": True}

    @app.post("/indexes")
    async def indexes(value: dict):
        """POST /indexes: register ``vectors`` and ``ids`` (optionally ``outcomes``); returns ``{"index_ref"}``."""
        if not {"vectors", "ids"}.issubset(value):
            raise ContractError("index_fields", "Provide vectors and IDs")
        return {"index_ref": await call_service(app.state.service.register_index, value["vectors"], value["ids"], value.get("outcomes"))}

    def streaming_question(value):
        """Validate the public ``index_apply`` question used by the streaming routes."""
        if not isinstance(value, dict):
            raise ContractError("question_shape", "Provide an index_apply question")
        if value.get("type", "index_apply" if "question_ref" in value else None) != "index_apply":
            raise ContractError("question_type", "Streaming routes take index_apply questions")
        return {**value, "type": "IndexApply"}

    @app.post("/questions/{name}")
    async def prewarm_question(name: str, value: dict):
        """POST /questions/{name}: prepare an ``index_apply`` question once; returns ``{"question_ref"}`` to reuse across streams."""
        return {"question_ref": await call_service(app.state.service.register_question, name, streaming_question(value))}

    @app.post("/index-apply/{handle}")
    async def stream_index(handle: str, value: dict):
        """POST /index-apply/{handle}: stream NDJSON answers for every row of a registered index."""
        stream = await call_service(app.state.service.stream_index, handle, streaming_question(value))
        return StreamingResponse(serialized_stream(stream),
                                 media_type="application/x-ndjson")

    @app.delete("/indexes/{handle}")
    async def delete_index(handle: str):
        """DELETE /indexes/{handle}: drop a registered index."""
        def remove():
            app.state.service.index(handle)
            del app.state.service.indexes[handle]
        await call_service(remove)
        return {"deleted": True}

    @app.put("/calibrations/{family}")
    async def calibrate(family: str, value: dict):
        """PUT /calibrations/{family}: fit a temperature from ``logits`` and ``targets``; returns ``{"temperature"}``."""
        if not {"logits", "targets"}.issubset(value):
            raise ContractError("calibration_fields", "Provide logits and targets")
        t = await call_service(app.state.service.register_calibration, family, value["logits"], value["targets"], value.get("type", "Choice"))
        return {"temperature": t}

    @app.delete("/calibrations/{family}")
    async def uncalibrate(family: str, type: str = "Choice"):
        """DELETE /calibrations/{family}: remove the stored temperature for ``type`` (default Choice)."""
        await call_service(app.state.service.delete_calibration, family, type)
        return {"deleted": True}

    @app.get("/health")
    async def health():
        """GET /health: service status, pinned reader revision, and the text reader's state."""
        service = app.state.service
        return {"status": "ok", "revision": REVISION, "reader_loaded": service.embedder is not None,
                "reader": service.reader_state,
                "note": "The reader loads in the background at 'venn serve' start, or on the first text request "
                        "with --no-preload; loading can take several seconds."}

    @app.get("/playground", response_class=HTMLResponse)
    async def playground():
        """GET /playground: serve the bundled interactive HTML playground."""
        from importlib.resources import files

        return HTMLResponse(files("venn").joinpath("playground.html").read_text(),
                            headers={"Referrer-Policy": "no-referrer",
                                     "X-Content-Type-Options": "nosniff"})

    if site_dir is not None:
        from fastapi.staticfiles import StaticFiles

        app.mount("/guide", StaticFiles(directory=site_dir, html=True), name="guide")

    return app
