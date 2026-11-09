"""Contract tests for bulk embedding: embed_media and the plural states {"texts"|"images"|"audios": [...]}."""
import base64
import io

import numpy as np
import pytest
from PIL import Image

from test_adapters_mcp import fake_assets
from venn.errors import ContractError
from venn.server import DecisionService


class FakeMediaReader:
    """Stands in for the pinned reader: a deterministic vector per image, recording every batch it is given."""

    def __init__(self):
        self.batches = []

    def encode(self, items, batch_size=32, normalize_embeddings=True):
        self.batches.append(len(items))
        out = []
        for item in items:
            pixels = np.asarray(item["image"][0], dtype=np.float64) if "image" in item else np.asarray(item["audio"][0], dtype=np.float64)
            rng = np.random.default_rng(int(pixels.sum() * 1000) % (2 ** 32))
            v = rng.normal(size=768)
            out.append(v / np.linalg.norm(v))
        return np.array(out, dtype=np.float32)


def fake_text(texts, prompt_name="Classification"):
    return np.array([np.random.default_rng(abs(hash(t)) % (2 ** 32)).normal(size=768) for t in texts], dtype=np.float32)


def service_for(tmp_path):
    if not (tmp_path / "assets").exists():
        fake_assets(tmp_path / "assets")
    service = DecisionService(fake_text, model_dir=tmp_path / "assets", cache_path=tmp_path / "cache", registry_path=tmp_path / "registry.json")
    service.media_model = FakeMediaReader()
    return service


def png(value):
    buffer = io.BytesIO()
    Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def test_bulk_matches_single_and_batches_only_misses(tmp_path):
    service = service_for(tmp_path)
    images = [png(v) for v in (10, 20, 30, 20, 40)]  # one repeat
    bulk = service.embed_media(images, "image", batch_size=2)
    assert bulk.shape == (5, 768) and np.allclose(np.linalg.norm(bulk, axis=1), 1, atol=1e-5)
    assert np.allclose(bulk[1], bulk[3])
    assert service.media_model.batches == [2, 2]  # four distinct images, batches of two
    single = np.array([service.media({"image": image}) for image in images])
    assert np.allclose(single, bulk) and service.media_model.batches == [2, 2]  # all from the cache now
    other = service_for(tmp_path)  # a new service over the same cache embeds nothing
    assert np.allclose(other.embed_media(images, "image"), bulk) and other.media_model.batches == []


def test_one_bad_payload_rejects_the_batch_before_embedding(tmp_path):
    service = service_for(tmp_path)
    with pytest.raises(ContractError, match="Invalid image"):
        service.embed_media([png(1), base64.b64encode(b"not an image").decode()], "image")
    assert service.media_model.batches == []
    with pytest.raises(ContractError, match="base64"):
        service.embed_media(["***"], "image")
    with pytest.raises(ContractError, match="image or audio"):
        service.embed_media([png(1)], "video")


def test_plural_states_follow_the_singular_shapes(tmp_path):
    service = service_for(tmp_path)
    images = [png(v) for v in (5, 6, 7)]
    matrix = service.vector({"images": images})
    assert matrix.shape == (3, 768) and np.allclose(matrix[0], service.vector({"image": images[0]}))
    texts = service.vector({"texts": ["a", "b"]})
    assert texts.shape == (2, 768) and np.allclose(texts[1], service.vector({"text": "b"}))
    with pytest.raises(ContractError, match="one kind"):
        service.vector({"images": images, "image": images[0]})
    with pytest.raises(ContractError, match="1..2"):
        service.vector({"images": images}, max_rows=2)


def test_bulk_states_answer_a_family_in_one_call(tmp_path):
    service = service_for(tmp_path)
    images = [png(v) for v in range(0, 240, 6)]
    vectors = service.embed_media(images, "image")
    y = vectors[:, 0] * 10 + 50
    service.register_family("brightness", [{"state": {"vector": v.tolist()}, "score": float(t)} for v, t in zip(vectors, y)],
                            objective="regress", target="score", split={"train": list(range(len(images)))}, cv={"folds": 3, "repeats": 1})
    rows = service.decide({"state": {"images": images[:5]}, "questions": {"s": {"type": "Score", "family": "brightness"}}})["answers"]["s"]["rows"]
    one = service.decide({"state": {"image": images[0]}, "questions": {"s": {"type": "Score", "family": "brightness"}}})["answers"]["s"]
    assert len(rows) == 5 and rows[0]["score"] == pytest.approx(one["score"])


def test_plural_states_answer_index_apply_and_name_the_type_otherwise(tmp_path):
    service = service_for(tmp_path)
    options = [{"label": "a"}, {"label": "b"}]
    rows = service.decide({"state": {"texts": ["x", "y", "z"]}, "questions": {
        "q": {"type": "IndexApply", "options": options}}})["answers"]["q"]["rows"]
    assert [row["id"] for row in rows] == [0, 1, 2]
    for kind in ("Choice", "Noul"):
        with pytest.raises(ContractError, match=f"{kind} answers one item; ask an IndexApply"):
            service.decide({"state": {"texts": ["x", "y"]}, "questions": {"q": {"type": kind, "options": options}}})
