"""Minimal append-only, content-addressed decision-event histories.

This helper records observed and hypothetical decisions without confusing them.
It is not a persistent database, encryption boundary, or causal state machine.
The caller controls secrets, timestamps, retention, and external storage.
"""

import hashlib
import json


def _validate_json(value):
    """Reject lossy Python-to-JSON coercions before assigning a content hash."""
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise ValueError("History JSON object keys must be strings")
        for item in value.values():
            _validate_json(item)
    elif isinstance(value, list):
        for item in value:
            _validate_json(item)
    elif value is None or type(value) in (str, bool, int, float):
        return
    else:
        raise ValueError("History requires native JSON values, not Python-specific containers")


class EventHistory:
    """In-memory immutable-event DAG with single-parent branching.

    Each event is canonical JSON, hashed with SHA-256 and indexed by digest.
    Snapshot reconstruction returns a path of events, not an inferred state.
    """

    def __init__(self, *, max_event_bytes=65536):
        if type(max_event_bytes) is not int or not 1 <= max_event_bytes <= 1048576:
            raise ValueError("max_event_bytes must be an integer between 1 and 1048576")
        self._events = {}
        self.max_event_bytes = max_event_bytes

    def append(self, *, parent=None, branch, kind, payload, provenance, observed):
        """Append an event and return its stable content-addressed ID.

        A parent must already exist. ``observed`` distinguishes measured outcomes from
        proposed or simulated events, without judging their semantic validity.
        The provenance must identify a nonempty source and revision.
        """
        if parent is not None and parent not in self._events:
            raise ValueError("Parent event does not exist")
        if any(not isinstance(s, str) or not s for s in (branch, kind)):
            raise ValueError("Branch and event kind must be nonempty strings")
        if type(observed) is not bool:
            raise ValueError("observed must be boolean")
        if (not isinstance(provenance, dict)
                or any(not isinstance(provenance.get(key), str) or not provenance.get(key)
                       for key in ("source", "revision"))):
            raise ValueError("Provenance must provide nonempty source and revision")
        record = {"parent": parent, "branch": branch, "kind": kind,
                  "payload": payload, "provenance": provenance, "observed": observed}
        try:
            _validate_json(record)
            encoded = json.dumps(record, sort_keys=True, ensure_ascii=False,
                                 separators=(",", ":"), allow_nan=False).encode("utf-8")
        except (TypeError, ValueError, RecursionError) as error:
            raise ValueError("History events require finite JSON data") from error
        if len(encoded) > self.max_event_bytes:
            raise ValueError("History event exceeds the configured byte limit")
        event_id = hashlib.sha256(encoded).hexdigest()
        if event_id in self._events and self._events[event_id] != encoded:
            raise ValueError("Digest collision for different event content")
        self._events[event_id] = encoded
        return event_id

    def get(self, event_id):
        """Return a detached event record with its ID, rejecting unknown IDs."""
        if event_id not in self._events:
            raise ValueError("Unknown event ID")
        encoded = self._events[event_id]
        if hashlib.sha256(encoded).hexdigest() != event_id:
            raise ValueError("Stored event content hash mismatch")
        return {"id": event_id, **json.loads(encoded)}

    def lineage(self, event_id):
        """Return root-to-head events; branches never import sibling events."""
        path = []
        seen = set()
        while event_id is not None:
            if event_id in seen:
                raise ValueError("Cycle in history store")
            seen.add(event_id)
            event = self.get(event_id)
            path.append(event)
            event_id = event["parent"]
        return list(reversed(path))

    def __len__(self):
        return len(self._events)
