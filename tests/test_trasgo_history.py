"""§1-shaped projections require complete external canonical evidence."""

import json

import pytest

from venn.history import EventHistory
from venn.trasgo_history import project_history, resolve_history


def _sample():
    history = EventHistory()
    provenance = {"source": "synthetic-evaluation", "revision": "test-1"}
    root = history.append(branch="original", kind="state", observed=True,
                          payload={"request": "synthetic", "vector": [0.1, 0.2]},
                          provenance=provenance)
    observed = history.append(parent=root, branch="observed", kind="reward", observed=True,
                              payload={"outcome": 4.5, "evidence_ids": ["case-3"]}, provenance=provenance)
    hypothetical = history.append(parent=root, branch="counterfactual", kind="reward", observed=False,
                                  payload={"outcome": 9., "evidence_ids": []}, provenance=provenance)
    return history, root, observed, hypothetical


def test_projection_rehydrates_exact_branch_only_with_canonical_store():
    history, root, actual, imagined = _sample()
    packet = project_history(history, imagined)
    assert packet["§"] == 1
    assert set(packet) == {"§", "E", "S", "R", "Δ", "μ", "ρ"}
    assert packet["μ"]["cert"] is None
    assert [e["id"] for e in resolve_history(packet, history)] == [root, imagined]
    assert resolve_history(packet, history)[-1]["observed"] is False
    assert resolve_history(project_history(history, actual), history)[-1]["observed"] is True
    assert "9.0" not in json.dumps(packet)  # Numerical outcome exists only in the canonical store.


def test_tampered_ids_and_metadata_cannot_be_promoted():
    history, root, actual, imagined = _sample()
    packet = project_history(history, actual)
    packet["Δ"][0] = "0" * 64
    with pytest.raises(ValueError, match="mismatch"):
        resolve_history(packet, history)
    packet = project_history(history, actual)
    packet["μ"]["cert"] = 1.
    with pytest.raises(ValueError, match="mismatch"):
        resolve_history(packet, history)
    packet = project_history(history, actual)
    packet["E"]["N"] = [imagined]
    with pytest.raises(ValueError, match="mismatch"):
        resolve_history(packet, history)


def test_missing_external_store_prevents_rehydration():
    history, root, actual, imagined = _sample()
    packet = project_history(history, actual)
    with pytest.raises(ValueError, match="Unknown"):
        resolve_history(packet, EventHistory())
    assert packet["ρ"]["lineage_sha256"] == project_history(history, actual)["ρ"]["lineage_sha256"]


def test_missing_payload_is_not_recreated_from_plausibility():
    history, root, actual, imagined = _sample()
    p = project_history(history, actual)
    assert "evidence_ids" not in json.dumps(p)
    restored = resolve_history(p, history)
    assert restored[-1]["payload"]["evidence_ids"] == ["case-3"]


def test_empty_history_has_no_projectable_head():
    with pytest.raises(ValueError, match="empty"):
        project_history(EventHistory(), None)


def test_tampered_external_event_bytes_rejected_even_with_valid_packet_hash():
    history, _, actual, _ = _sample()
    packet = project_history(history, actual)
    corrupted = json.loads(history._events[actual])
    corrupted["payload"]["outcome"] = 9999.
    history._events[actual] = json.dumps(corrupted, sort_keys=True, ensure_ascii=False,
                                        separators=(",", ":"), allow_nan=False).encode("utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        resolve_history(packet, history)
