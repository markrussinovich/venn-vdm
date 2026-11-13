"""Optional, experimental §1-shaped projection of canonical decision history.

The packet is only a reference/index view, not standalone history, compression
proof, or verified Trasgo language-model conformance. Exact reconstruction
requires the external EventHistory's canonical event store.
"""

import hashlib
import json


def _digest(ids):
    data = json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def project_history(history, head):
    """Export a deterministic, external-reference-only §1-inspired packet."""
    events = history.lineage(head)
    if not events:
        raise ValueError("Cannot project an empty event lineage")
    ids = [event["id"] for event in events]
    return {
        "§": 1,
        "E": {"N": [head]},
        "S": {"N.count": len(ids), "N.root": ids[0]},
        "R": ["N→Δ:refs"],
        "Δ": ids,
        "μ": {"scope": "venn-events", "res": "external-ids", "cert": None},
        "ρ": {"lineage_sha256": _digest(ids), "ref_format": "sha256"},
    }


def resolve_history(packet, history):
    """Rehydrate exact event records only after verifying references and path.

    Without the external event store the packet cannot reconstruct any payload,
    source citation, numerical outcome, or observed/hypothetical distinction.
    """
    try:
        ids = packet["Δ"]
        head = packet["E"]["N"][0]
        if (packet["§"] != 1 or not isinstance(ids, list) or not ids
                or not all(isinstance(x, str) and len(x) == 64 for x in ids)
                or head != ids[-1] or packet["S"]["N.count"] != len(ids)
                or packet["S"]["N.root"] != ids[0]
                or packet["R"] != ["N→Δ:refs"]
                or packet["μ"] != {"scope": "venn-events", "res": "external-ids", "cert": None}
                or packet["ρ"]["ref_format"] != "sha256"
                or packet["ρ"]["lineage_sha256"] != _digest(ids)):
            raise ValueError("History packet metadata or lineage fingerprint mismatch")
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Malformed external-reference history packet") from error
    actual = history.lineage(head)
    if [event["id"] for event in actual] != ids:
        raise ValueError("External event lineage disagrees with packet")
    return actual
