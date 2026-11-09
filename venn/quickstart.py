"""A CPU quickstart: register an outcome store, ask one question, read the evidence.

Nothing is downloaded. The vectors are hand-written coordinates rather than text
embeddings, so this demonstrates the API, not semantic accuracy.
"""
import json

import faiss
import numpy as np

from venn.adapters import FaissOutcomeAdapter
from venn.outcomes import OutcomeService
from venn.profiles import TEXT_TAG

# Past tickets were handled by two teams. Each case records how well it went (0..1).
PAST_CASES = {
    "account": [("acct-101", .9), ("acct-102", .8), ("acct-103", 1.)],
    "billing": [("bill-201", .4), ("bill-202", .5), ("bill-203", .3)],
}


def build_store():
    """Return one FAISS index per team holding its past ticket vectors and outcomes."""
    rng = np.random.default_rng(0)
    parts = {}
    for team, cases in PAST_CASES.items():
        vectors = rng.normal(0, .02, (len(cases), TEXT_TAG.dimensions)).astype(np.float32)
        vectors[:, 0] += 1  # All past tickets sit in one region of the space.
        index = faiss.IndexFlatL2(TEXT_TAG.dimensions)
        index.add(vectors)
        parts[team] = {"index": index, "ids": [case_id for case_id, _ in cases],
                       "outcomes": [outcome for _, outcome in cases]}
    return parts


def run():
    """Route a new ticket to the team whose similar past tickets had the best outcomes."""
    service = OutcomeService()
    store = service.register_store(FaissOutcomeAdapter(build_store(), tag=TEXT_TAG))
    ticket = np.zeros(TEXT_TAG.dimensions, dtype=np.float32)
    ticket[0] = 1  # A new ticket in the same region as the past cases.
    response = service.decide_batch(
        [{"vector": ticket, "tag": TEXT_TAG.to_dict()}],
        {"team": {"type": "nearest_outcome", "store": store, "options": list(PAST_CASES),
                  "k": 3, "min_support": 1, "goal": "maximize"}},
        rng=np.random.default_rng(0),
    )
    answer = response["results"][0]["answers"]["team"]
    result = {
        "answer": answer["answer"],
        # These are the per-option fields the API returns: mean outcome, case count and evidence IDs.
        "options": {team: {"mean": round(stats["mean"], 2), "count": stats["count"], "evidence_ids": stats["evidence_ids"]}
                    for team, stats in answer["options"].items()},
        "note": "authored vectors, not text embeddings; nothing was downloaded",
    }
    if result["answer"] != "account":
        raise RuntimeError("Unexpected quickstart result")
    return result


def main():
    """Print the quickstart result as JSON."""
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
