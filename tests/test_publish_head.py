import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest

from scripts import publish_head
from scripts.check_release_ci import check as check_release_ci
from test_adapters_mcp import fake_assets


@pytest.mark.parametrize("change", [
    {"head_sha": "other"},
    {"event": "pull_request"},
    {"status": "in_progress", "conclusion": None},
    {"status": "queued", "conclusion": None},
    {"conclusion": "failure"},
    {"conclusion": "cancelled"},
])
def test_release_gate_rejects_unverified_revisions(change):
    run = {"id": 1, "head_sha": "release", "event": "push",
           "status": "completed", "conclusion": "success", **change}
    with pytest.raises(ValueError):
        check_release_ci({"workflow_runs": [run]}, "release")


def test_release_gate_requires_latest_exact_revision_success():
    run = {"id": 1, "head_sha": "release", "event": "push",
           "status": "completed", "conclusion": "success"}
    assert check_release_ci({"workflow_runs": [run]}, "release") == 1
    with pytest.raises(ValueError, match="No push CI"):
        check_release_ci({"workflow_runs": []}, "release")
    with pytest.raises(ValueError):
        check_release_ci({"workflow_runs": [{**run, "conclusion": "failure"}, run]}, "release")


def retained_like(tmp_path, monkeypatch):
    """Fake assets whose checksums are pinned for the duration of a test."""
    model = tmp_path / "model"
    fake_assets(model)
    config = json.loads((model / "config.json").read_text())
    config.update(format_version=1, default_head="CosineAbstention", source_runs=["internal"])
    (model / "config.json").write_text(json.dumps(config))
    public = {key: config[key] for key in publish_head.PUBLIC_CONFIG_KEYS}
    monkeypatch.setattr(publish_head, "HEAD_SHA256", publish_head.sha256(model / "universal.safetensors"))
    monkeypatch.setattr(publish_head, "HEAD_CONFIG_SHA256",
                        hashlib.sha256(json.dumps(public, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
    card = tmp_path / "card.md"
    card.write_text("# Card\nSee [limits](limitations.md) and [license](../LICENSE).\n")
    return model, card


def test_stage_writes_loader_files_and_absolute_card_links(tmp_path, monkeypatch):
    model, card = retained_like(tmp_path, monkeypatch)
    output = publish_head.stage(model, tmp_path / "bundle", card)
    assert {p.name for p in (output / "assets").iterdir()} == {"config.json", "manifest.json", "universal.safetensors"}
    assert "source_runs" not in json.loads((output / "assets/config.json").read_text())
    readme = (output / "README.md").read_text()
    assert readme.startswith("---\nbase_model: google/embeddinggemma-2")
    assert "license: other\n" in readme
    assert "license_name: venn-1-redistribution-permission\n" in readme
    assert "blob/main/docs/limitations.md" in readme and "blob/main/LICENSE" in readme
    assert "docs-navigation" not in readme
    with pytest.raises(ValueError, match="new output"):
        publish_head.stage(model, tmp_path / "bundle", card)


def test_stage_rejects_other_heads_and_changed_configuration(tmp_path, monkeypatch):
    model, card = retained_like(tmp_path, monkeypatch)
    (model / "universal.safetensors").write_bytes(b"different weights")
    with pytest.raises(ValueError, match="not the pinned"):
        publish_head.stage(model, tmp_path / "a", card)
    assert not (tmp_path / "a").exists()
    other = tmp_path / "other"
    fake_assets(other)
    config = json.loads((other / "config.json").read_text())
    config.update(format_version=1, default_head="CosineAbstention", choice_temperature=2.)
    (other / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(publish_head, "HEAD_SHA256", publish_head.sha256(other / "universal.safetensors"))
    model = other
    with pytest.raises(ValueError, match="configuration differs"):
        publish_head.stage(model, tmp_path / "b", card)


def test_publish_is_private_by_default_and_dry_run_makes_no_network_calls(tmp_path, monkeypatch):
    model, card = retained_like(tmp_path, monkeypatch)
    output = publish_head.stage(model, tmp_path / "bundle", card)
    with patch("huggingface_hub.HfApi") as constructor:
        assert publish_head.publish(output, "example/venn-1", dry_run=True) is None
        constructor.assert_not_called()
        api = constructor.return_value
        api.upload_folder.return_value = MagicMock(oid="abc123")
        assert publish_head.publish(output, "example/venn-1", token="TEST_ONLY") == "abc123"
        assert api.create_repo.call_args.kwargs["private"] is True
        publish_head.publish(output, "example/venn-1", public=True, token="TEST_ONLY")
        assert api.create_repo.call_args.kwargs["private"] is False
    (output / "extra.txt").write_text("not allowed")
    with pytest.raises(ValueError, match="exactly"):
        publish_head.publish(output, "example/venn-1", dry_run=True)
