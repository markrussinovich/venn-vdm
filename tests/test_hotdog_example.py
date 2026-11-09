import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def hotdog():
    path = Path(__file__).parents[1] / "examples" / "hotdog" / "hotdog.py"
    spec = importlib.util.spec_from_file_location("hotdog_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_photos_and_inventory_stay_local(hotdog, tmp_path):
    from fastapi.testclient import TestClient

    data = tmp_path / "photos"
    data.mkdir()
    (data / "food.png").write_bytes(b"photo")
    (data / "notes.txt").write_text("not a photo")
    demo = object.__new__(hotdog.Demo)
    demo.data = data
    demo.images = {"0": data / "food.png"}
    client = TestClient(hotdog.create_app(demo), base_url="http://127.0.0.1")
    assert hotdog.photos(data) == [data / "food.png"]
    assert client.get("/image/0").content == b"photo"
    assert client.get("/image/unknown").status_code == 404
    assert client.post("/action/judge", json={"id": "../secret"}).status_code == 400
    assert client.get("/image/0", headers={"Host": "attacker.example"}).status_code == 400
    assert client.get("/image/0", headers={"Origin": "http://attacker.example"}).status_code == 403


def test_training_rejects_empty_data(hotdog, tmp_path):
    demo = object.__new__(hotdog.Demo)
    demo.data = tmp_path
    with pytest.raises(ValueError, match="five distinct"):
        demo.train()


def test_evaluation_rejects_training_leakage(hotdog, tmp_path):
    import hashlib

    demo = object.__new__(hotdog.Demo)
    demo.data = tmp_path
    for label in hotdog.LABELS:
        folder = tmp_path / "test" / label
        folder.mkdir(parents=True)
        (folder / "photo.jpg").write_bytes(label.encode())
    demo.training = {"training_hashes": [hashlib.sha256(b"hotdog").hexdigest()]}
    with pytest.raises(ValueError, match="not used in training"):
        demo.evaluate()


def test_compare_loads_the_reader_on_the_chosen_device(hotdog, tmp_path, monkeypatch):
    import venn.decisions

    demo = object.__new__(hotdog.Demo)
    demo.device = "cuda:7"
    demo.service = type("Service", (), {"media_model": None})()
    loaded = []

    def reader():
        loaded.append(demo.device)
        demo.service.media_model = object()
        return 1.5

    def decide(service, prompt, experimental_media=False):
        assert service.media_model is not None, "the service would load its own CPU reader"
        return {"answers": {"food": {"choice": "hamburger", "probabilities": {}, "venn": {"calibrated": False}}}}

    monkeypatch.setattr(demo, "reader", reader)
    monkeypatch.setattr(venn.decisions, "decide", decide)
    image = tmp_path / "lunch.png"
    image.write_bytes(b"photo")
    result = demo.compare_descriptions(image)
    assert loaded == ["cuda:7"] and result["reader_load_seconds"] == 1.5
