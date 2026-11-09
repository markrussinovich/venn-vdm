import importlib.util
from pathlib import Path

import numpy as np
import pytest

from venn import OutcomeService
from venn.profiles import IMAGE_TAGS, VENN_1, VectorTag


spec = importlib.util.spec_from_file_location(
    "arena", Path(__file__).resolve().parents[1] / "examples/arena.py")
arena = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arena)
TAG = VectorTag("experimental:arena-test", "a" * 40, "test", 2)


def test_terminal_returns_merge_per_action_and_live_api_sees_updates(tmp_path):
    memory = arena.Memory(TAG, ["0", "1"])
    vector = np.array([1., 0.], np.float32)
    memory.observe([vector, vector], [0, 1], [1., 2.], .5)
    service = OutcomeService()
    handle = service.register_store(memory.adapter)
    response = arena.decide(service, handle, [vector], TAG, np.random.default_rng(1),
                            [0.], 11, True)["results"][0]["answers"]["move"]
    assert response["options"]["0"]["mean"] == 2.
    assert response["options"]["1"]["mean"] == 2.
    memory.observe([vector], [1], [5.], .5)
    response = arena.decide(service, handle, [vector], TAG, np.random.default_rng(1),
                            [0.], 11, True)["results"][0]["answers"]["move"]
    assert response["answer"] == "1"
    memory.observe([vector], [1], [3.], .5)
    assert memory.parts["1"]["outcomes"] == [5.]
    memory.save(tmp_path, {})
    restored = arena.Memory(TAG, ["0", "1"])
    restored.load(tmp_path)
    assert restored.size == 2
    assert restored.parts["1"]["outcomes"] == [5.]
    wrong = arena.Memory(VectorTag("experimental:other", "a" * 40, "test", 2), ["0", "1"])
    with pytest.raises(ValueError, match="mismatch"):
        wrong.load(tmp_path)


def test_experimental_tag_never_relabels_retained_profile():
    with pytest.raises(ValueError, match="mismatch"):
        VectorTag(VENN_1.id, "a" * 40, "test", 2)
    with pytest.raises(ValueError, match="mismatch"):
        VectorTag("venn-1-text", VENN_1.revision, "test", 2)
    with pytest.raises(ValueError, match="immutable"):
        VectorTag("experimental:reader", "main", "test", 2)
    service = OutcomeService()
    handle = service.register_store(arena.Memory(TAG, ["0"]).adapter)
    with pytest.raises(ValueError, match="mismatch"):
        arena.decide(service, handle, [np.ones(2)], VectorTag(
            "experimental:arena-test", "b" * 40, "test", 2), np.random.default_rng(1), [0.], 11, True)


def test_native_path_sends_raw_frames_to_venn_not_portable_reader():
    class Reader:
        def encode_frames(self, frames, representation):
            assert frames.dtype == np.uint8
            assert representation == "image-thirds-70"
            return np.ones((len(frames), 2304), np.float32) / 48

    tag = IMAGE_TAGS["image-thirds-70"]
    memory = arena.Memory(tag, ["0", "1"])
    service = OutcomeService(image_reader=Reader())
    handle = service.register_store(memory.adapter)
    response = arena.decide(service, handle, np.ones((2, 3, 3, 3), np.uint8),
                            tag, np.random.default_rng(1), [0., 0.], 11, True, native=True)
    assert len(response["results"]) == 2
    assert response["vector_tag"] == tag.to_dict()


def test_episode_returns_do_not_cross_episode_boundaries():
    np.testing.assert_allclose(arena.returns([1, 2], .5), [2, 2])
    np.testing.assert_allclose(arena.returns([4], .5), [4])


def test_portable_reader_preserves_ordered_thirds_and_normalizes():
    class Model:
        def encode(self, images, **kwargs):
            assert [image.size for image in images] == [(106, 240), (107, 240), (107, 240)]
            assert [np.asarray(image)[0, 0, 0] for image in images] == [1, 2, 3]
            return np.eye(3, dtype=np.float32)

    reader = arena.PortableReader.__new__(arena.PortableReader)
    reader.model, reader.layout, reader.gemma = Model(), "thirds", False
    frames = np.ones((1, 240, 320, 3), np.uint8)
    frames[:, :, 106:213] = 2
    frames[:, :, 213:] = 3
    vectors = reader.encode(frames)
    assert vectors.shape == (1, 9)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-6)
    assert np.argmax(vectors.reshape(3, 3), axis=1).tolist() == [0, 1, 2]


def test_head_is_compiled_from_the_memory_and_answers_the_same_question(tmp_path):
    rng = np.random.default_rng(3)
    memory = arena.Memory(TAG, ["0", "1"])
    centres = np.asarray([[1., 0.], [0., 1.]], np.float32)
    for i in range(400):
        best = i % 2
        vector = (centres[best] + .1 * rng.normal(size=2)).astype(np.float32)
        action = best if rng.random() < .8 else 1 - best
        memory.observe([vector], [action], [1. if action == best else -1.], .99)
    service = OutcomeService(head_device="cpu")
    handle = service.register_store(memory.adapter)
    args = type("Args", (), {"head": "awr", "k": 5, "exact_match": True, "seed": 0})()
    summary = arena.register_head(service, handle, ["0", "1"], None, args, warm=False)
    assert summary["parameters"] == 2 * 16 + 2 * 16 + 1 and summary["cases"] == memory.size
    picks = [arena.decide(service, handle, [c], TAG, np.random.default_rng(0), [0.], 5, True, family="policy")
             ["results"][0]["answers"]["move"] for c in centres]
    assert [p["answer"] for p in picks] == ["0", "1"] and all(p["options"] is None for p in picks)
    again = arena.register_head(service, handle, ["0", "1"], None, args, warm=True)
    assert again["report"]["warm_start"] is True
    assert service.save_family("policy", tmp_path / "head.pt") == again["sha256"]
