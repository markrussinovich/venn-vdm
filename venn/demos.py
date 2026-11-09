"""Small reproducible demos: conditional retrieval filtering, drift detection and a million-row sweep."""

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from venn.reader import embed_cached
from venn.index.prewarm import prewarm
from venn.index.shift import population_shift
from venn.benchmark import write


PASSAGES = [
    "Reset a forgotten password using the password reset link on the sign-in page.",
    "If you cannot log in, request a password reset email and follow its recovery link.",
    "To unlock an account, verify your identity with the account access team.",
    "A duplicate card charge can be refunded by the billing team.",
    "Track a parcel using its shipping tracking number.",
    "Change the delivery address before the parcel leaves the warehouse.",
    "Refunds are returned to the original payment method.",
    "A garden needs enough water and sunlight for its plants.",
    "Pasta cooks in boiling water.",
    "The office closes at six in the evening.",
    "The basketball game starts on Saturday.",
    "Music lessons take place in the studio.",
]
RAG_QUERY = "I forgot my password and cannot sign in. How can I recover my account?"
RAG_OPTIONS = ["Account sign-in, password recovery and account access",
               "Billing, payments and refunds", "Parcel delivery and shipping", "Other topics"]
DRIFT_A = ["I cannot sign in.", "I forgot my password.", "My account is locked.", "The login fails."]
DRIFT_B = ["Where is my parcel?", "My shipment is late.", "Change the delivery address.", "Track my package."]


def conditional_scores(states, options):
    """Return cosine scores of ``states`` against ``options`` centered on their centroid, shape ``(states, options)``."""
    import torch
    with torch.inference_mode():
        _, centered, _ = prewarm(torch.zeros(1, options.shape[1]),
                                 torch.tensor(options)[None],
                                 torch.ones(1, len(options), dtype=torch.bool))
        return (torch.tensor(states) @ centered[0].T).numpy()


def retrieval_demo(embedded):
    """Rank the demo passages for the account-recovery query, then keep only those whose topic is account access.

    ``embedded`` maps each demo text to its vector. Returns precision and recall before and after filtering.
    """
    vectors = np.stack([embedded[value] for value in PASSAGES])
    query = embedded[RAG_QUERY]
    labels = conditional_scores(vectors, np.stack([embedded[value] for value in RAG_OPTIONS])).argmax(1)
    ranked = np.argsort(vectors @ query)[::-1]
    before = ranked[:6].tolist()
    after = [int(i) for i in before if labels[i] == 0]
    relevant = {0, 1, 2}
    def measure(ids):
        hits = len(set(ids) & relevant)
        return {"passage_ids": ids, "precision": hits / len(ids) if ids else None,
                "recall": hits / len(relevant), "returned": len(ids)}
    return {"query": RAG_QUERY, "passages": PASSAGES, "before": measure(before), "after": measure(after),
            "policy": "Freeze four topic descriptions; retain conditional account-topic argmax, no tuned probability threshold.",
            "ground_truth_relevant_ids": sorted(relevant),
            "scope": "Small authored illustrative retrieval, not a held-out RAG quality benchmark or generated-answer evaluation."}


def drift_demo(embedded):
    """Run the population-shift test on eight windows against a fixed account-topic reference.

    The last four windows switch to shipping texts. Returns per-window results and the test configuration.
    """
    rng = np.random.default_rng(20261102)
    reference = np.stack([embedded[value] for value in DRIFT_A])[np.arange(48) % 4]
    rows = []
    for window in range(8):
        pool = DRIFT_A if window < 4 else DRIFT_B
        current = np.stack([embedded[value] for value in pool])[rng.permutation(np.arange(48) % 4)]
        result = population_shift(reference, current, bandwidth=1., permutations=199,
                                  seed=20261102 + window, alpha=.05 / 8)
        rows.append({"window": window, "source": "account" if window < 4 else "shipping", **result})
    return {"windows": rows, "window_size": 48, "reference": "Fixed 48-row account reference",
            "alpha_per_window": .05 / 8, "horizon": 8, "familywise_alpha": .05,
            "scope": "Authored finite-support illustrative stream with repeats; not the distinct-row public benchmark, "
                     "not an unlimited sequential false-alarm guarantee. No posterior interpretation."}


def million_sweep(rows=1000000, batch_size=8192, model_dir=None, progress=None):
    """Score ``rows`` synthetic unit vectors against four random options and time generation, registration and consumption.

    With ``model_dir``, runs a real ``DecisionService`` IndexApply stream; otherwise a headless centered-cosine
    sweep. ``progress(stage, done, total)`` is called per batch. Returns a timing and count report.
    """
    rng = np.random.default_rng(20261102)
    options = rng.normal(size=(4, 768)).astype(np.float32)
    options /= np.linalg.norm(options, axis=1, keepdims=True)
    generation_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="venn-sweep-") as directory:
        vectors = np.memmap(Path(directory) / "generated.f16", mode="w+", dtype="float16", shape=(rows, 768))
        for offset in range(0, rows, batch_size):
            batch = rng.normal(size=(min(batch_size, rows - offset), 768)).astype(np.float32)
            batch /= np.linalg.norm(batch, axis=1, keepdims=True)
            vectors[offset:offset + len(batch)] = batch
            if progress:
                progress("generating", offset + len(batch), rows)
        vectors.flush()
        generation_seconds = time.perf_counter() - generation_start
        counts, consumed, serialized_bytes = {}, 0, 0
        digest = hashlib.sha256()
        if model_dir:
            from venn.server import DecisionService
            service = DecisionService(model_dir=model_dir, cache_path=Path(directory) / "cache",
                                      registry_path=Path(directory) / "registry.json")
            start = time.perf_counter()
            handle = service.register_index({"vectors": vectors}, list(range(rows)))
            registration_seconds = time.perf_counter() - start
            question = {"type": "IndexApply", "instructions_vec": options[0].tolist(),
                        "options": [{"label": str(i), "vector": vector.tolist()} for i, vector in enumerate(options)],
                        "batch_size": batch_size}
            start = time.perf_counter()
            for line in service.stream_index(handle, question):
                encoded = line.encode()
                digest.update(encoded)
                serialized_bytes += len(encoded)
                result = json.loads(line)
                for row in result["rows"]:
                    counts[row["answer"]] = counts.get(row["answer"], 0) + 1
                consumed += len(result["rows"])
                if progress:
                    progress("consuming", consumed, rows)
            scope = "Actual CPU DecisionService register_index + IndexApply NDJSON, supplied trained head, fully consumed."
        else:
            registration_seconds = 0.
            start = time.perf_counter()
            for offset in range(0, rows, batch_size):
                batch = np.asarray(vectors[offset:offset + batch_size], dtype=np.float32)
                batch /= np.linalg.norm(batch, axis=1, keepdims=True)
                predicted = conditional_scores(batch, options).argmax(1)
                line = json.dumps({"offset": offset, "choices": predicted.tolist()}).encode()
                digest.update(line)
                serialized_bytes += len(line)
                consumed += len(predicted)
                if progress:
                    progress("consuming", consumed, rows)
                for label, count in zip(*np.unique(predicted, return_counts=True)):
                    counts[str(label)] = counts.get(str(label), 0) + int(count)
            scope = "Headless CPU conditional centered-cosine sweep, not trained-head abstention or HTTP IndexApply."
        seconds = time.perf_counter() - start
        del vectors
    if consumed != rows or sum(counts.values()) != rows:
        raise RuntimeError("Sweep did not consume every output")
    return {"rows": rows, "dimensions": 768, "data": "Seeded synthetic normalized Gaussian vectors, not Gemma text embeddings",
            "scope": scope, "generation_seconds": generation_seconds,
            "index_registration_seconds": registration_seconds, "scoring_serialization_consumption_seconds": seconds,
            "rows_per_second": rows / seconds, "end_to_end_seconds": generation_seconds + registration_seconds + seconds,
            "counts": counts, "serialized_bytes": serialized_bytes, "consumed_sha256": digest.hexdigest(),
            "timing_excludes": "Model/service loading and final report writing; generation and registration reported separately.",
            "device": "CPU; no GPU kernel or semantic accuracy claim"}


def main():
    """Command-line entry point: embed the demo texts, run all three demos and write ``results.json`` under ``--output``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--rows", type=int, default=1000000)
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    if not 1 <= args.rows <= 1000000:
        parser.error("--rows must be in 1..1000000")
    from venn.reproduction import preflight
    preflight(args.gpus, args.output, args.cache)
    args.output.mkdir(parents=True, exist_ok=True)
    texts = PASSAGES + [RAG_QUERY] + RAG_OPTIONS + DRIFT_A + DRIFT_B
    write(args.output / "protocol.json", {"texts": texts, "seed": 20261102, "rows": args.rows,
                                        "model_mode": "head" if args.model_dir else "headless"})
    vectors, telemetry = embed_cached(texts, args.cache, args.output, args.gpus)
    embedded = dict(zip(texts, vectors))
    result = {"rag": retrieval_demo(embedded), "drift": drift_demo(embedded),
              "sweep": million_sweep(args.rows, model_dir=args.model_dir), "embedding": telemetry}
    write(args.output / "results.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
