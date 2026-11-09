"""Fit, reload and delete a family on synthetic vectors.

Needs the venn-1 head (downloaded automatically unless --model-dir is given) but no reader.
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from venn import DecisionService


def run(model_dir):
    rng = np.random.default_rng(42)
    centers = np.eye(3, 768, dtype=np.float32)
    options = [{"label": str(i), "vector": value.tolist()} for i, value in enumerate(centers)]
    examples = [{"state": (center + rng.normal(0, .01, 768)).tolist(), "label": str(i)}
                for i, center in enumerate(centers) for _ in range(10)]
    with tempfile.TemporaryDirectory(prefix="venn-family-") as folder:
        kwargs = {"model_dir": model_dir, "cache_path": Path(folder) / "cache.sqlite",
                  "registry_path": Path(folder) / "registry.json"}
        service = DecisionService(**kwargs)
        info = service.register_family("example", examples, options, centers[0].tolist(), seed=42)
        query = {"state": {"vector": centers[0].tolist()}, "questions": {
            "category": {"type": "Choice", "family": "example",
                         "instructions_vec": centers[0].tolist(), "options": options}}}
        before = service.decide(query)["answers"]
        reloaded = DecisionService(**kwargs)
        after = reloaded.decide(query)["answers"]
        if before != after:
            raise RuntimeError("Family response changed after reload")
        reloaded.delete_family("example")
        return {"scope": "synthetic family lifecycle; not ordinal or semantic validation",
                "split_counts": info["split_counts"], "calibrated": info["calibrated"],
                "reloaded_identically": True, "answer": after}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, help="Head assets; defaults to downloading the published venn-1 head")
    args = parser.parse_args()
    if args.model_dir is None:
        from venn.cli import download_head

        args.model_dir = download_head()
    print(json.dumps(run(args.model_dir), indent=2))
