"""Execute the registration shown in the film, using an existing embedding cache.

Inputs stay local. The output contains execution provenance, not tasting notes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time


def register_head(service, examples):
    return service.register_family(
        "wine_points", examples,
        objective="regress",
        target="points",
        form="kernel_ridge",
        prompt="SentenceSimilarity",
        inputs=["log_words", "log_chars"],
        split={"train": list(range(len(examples)))},
    )


def run(examples, model_dir, cache, output):
    from venn import DecisionService

    def cache_only(_texts, **_kwargs):
        raise RuntimeError("Missing SentenceSimilarity embeddings; prepare the cache first")

    with tempfile.TemporaryDirectory(prefix="venn-video-head-") as home:
        home = Path(home)
        shutil.copy2(cache, home / "cache.sqlite")
        service = DecisionService(
            embed=cache_only, model_dir=model_dir, cache_path=home / "cache.sqlite",
            registry_path=home / "registry.json",
        )
        started = time.perf_counter()
        report = register_head(service, examples)
        record = {
            "status": "executed",
            "scope": "API execution replay on the existing fitting notes; not a new held-out evaluation or timing comparison.",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "examples": len(examples),
            "embedding": "Existing pinned SentenceSimilarity cache; no new embedding or API calls.",
            "register_family_calls": 1,
            "elapsed_seconds": time.perf_counter() - started,
            "form": report["form"],
            "alpha": report["alpha"],
            "gamma": report["gamma"],
            "input_scale": report["input_scale"],
            "head_checksum": report["checksum"],
            "events": ["register_family started", "register_family returned"],
        }
        output.write_text(json.dumps(record, indent=2) + "\n")
        return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True, help="Local JSON list of state-text/points fitting examples")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True, help="Existing Venn SQLite embedding cache")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.examples.read_text()), args.model_dir, args.cache, args.output), indent=2))
