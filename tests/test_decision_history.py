"""Structural tests for content-addressed decision history; no model semantics."""

import pytest

from venn.history import EventHistory


def _put(log, parent=None, branch="main", kind="decision", observed=True, payload=None):
    return log.append(parent=parent, branch=branch, kind=kind,
                      payload={"option": "a", "value": 1.} if payload is None else payload,
                      provenance={"source": "test", "revision": "synthetic-v1"},
                      observed=observed)


def test_forks_preserve_shared_parent_and_isolate_siblings():
    log = EventHistory()
    root = _put(log, kind="state")
    measured = _put(log, root, branch="actual", kind="reward", payload={"reward": 3.})
    imagined = _put(log, root, branch="candidate", kind="reward", observed=False,
                   payload={"reward": 5.})
    assert measured != imagined
    assert [e["id"] for e in log.lineage(measured)] == [root, measured]
    assert [e["id"] for e in log.lineage(imagined)] == [root, imagined]
    assert log.get(measured)["observed"] is True
    assert log.get(imagined)["observed"] is False
    assert len(log) == 3


def test_canonical_json_keeps_id_order_independent_and_payload_detached():
    log = EventHistory()
    payload = {"a": 1, "b": [1, 2]}
    first = _put(log, payload=payload)
    second = _put(log, payload={"b": [1, 2], "a": 1})
    assert first == second and len(log) == 1
    payload["b"].append(999)
    event = log.get(first)
    assert event["payload"] == {"a": 1, "b": [1, 2]}
    event["payload"]["b"].append(999)
    assert log.get(first)["payload"] == {"a": 1, "b": [1, 2]}


def test_observation_flag_and_provenance_affect_identity():
    log = EventHistory()
    actual = _put(log)
    imagined = _put(log, observed=False)
    assert actual != imagined
    assert log.get(actual)["provenance"]["revision"] == "synthetic-v1"


@pytest.mark.parametrize("kwargs", [
    {"parent": "bad"},
    {"observed": "yes"},
    {"branch": ""},
    {"kind": ""},
    {"payload": {"x": float("nan")}},
    {"payload": {"x": object()}},
    {"payload": {7: "coerced-key"}},
    {"payload": {"x": (1, 2)}},
    {"provenance": {}},
    {"provenance": {"source": "test", "revision": ""}},
])
def test_invalid_or_unattested_history_is_rejected(kwargs):
    history = EventHistory()
    default = dict(parent=None, branch="a", kind="decision", payload={"x": 1},
                   provenance={"source": "test", "revision": "v1"}, observed=True)
    default.update(kwargs)
    with pytest.raises(ValueError):
        history.append(**default)


def test_root_and_byte_limits_and_unknown_event():
    history = EventHistory(max_event_bytes=180)
    with pytest.raises(ValueError, match="limit"):
        _put(history, payload={"raw": "x" * 500})
    root = _put(history, payload={"x": 1})
    assert len(history.lineage(root)) == 1
    with pytest.raises(ValueError, match="Unknown"):
        history.get("0" * 64)
    with pytest.raises(ValueError):
        EventHistory(max_event_bytes=True)


def test_stored_event_mutation_is_detected_on_every_read():
    import json

    history = EventHistory()
    event_id = _put(history, payload={"outcome": 1.})
    record = json.loads(history._events[event_id])
    record["payload"]["outcome"] = 1000.
    history._events[event_id] = json.dumps(record, sort_keys=True, ensure_ascii=False,
                                           separators=(",", ":"), allow_nan=False).encode("utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        history.get(event_id)
    with pytest.raises(ValueError, match="hash mismatch"):
        history.lineage(event_id)
