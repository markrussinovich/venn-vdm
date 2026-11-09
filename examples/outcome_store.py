"""A synthetic CPU outcome-store example without fitted heads or a reader."""
import faiss
import numpy as np

from venn import OutcomeService
from venn.adapters import FaissOutcomeAdapter
from venn.profiles import TEXT_TAG


def main():
    index = faiss.IndexFlatL2(768)
    vector = np.zeros(768, dtype=np.float32)
    vector[0] = 1  # Authored coordinates, not a semantic embedding.
    index.add(vector[None])
    parts = {"support": {"index": index, "ids": ["resolved-1"], "outcomes": [1.]}}
    service = OutcomeService()
    store = service.register_store(FaissOutcomeAdapter(parts, tag=TEXT_TAG))
    result = service.decide_batch(
        [{"vector": vector, "tag": TEXT_TAG.to_dict()}],
        {"queue": {"type": "nearest_outcome", "store": store,
                   "options": ["support", "billing"], "k": 11,
                   "min_support": 1, "goal": "maximize"}},
    )
    answer = result["results"][0]["answers"]["queue"]["answer"]
    if answer != "support":
        raise RuntimeError("Unexpected outcome-store result")
    print(answer)


if __name__ == "__main__":
    main()
