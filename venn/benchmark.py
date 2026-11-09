"""Venn space benchmark: nearest-reference, membership and population-shift tasks on public intent datasets."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from venn.reader import MODEL_ID, REVISION, embed_cached
from venn.index.decision import leave_one_out_distances, nearest_decision
from venn.index.density import knn_outlier_scores
from venn.index.shift import population_shift


def write(path, value):
    """Write ``value`` to ``path`` as indented JSON with a trailing newline; rejects NaN and infinity."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def text_hash(value):
    """Return the SHA-256 of ``value`` after case-folding and collapsing whitespace."""
    return hashlib.sha256(" ".join(value.casefold().split()).encode()).hexdigest()


def wilson(successes, count):
    """Return the 95% Wilson score interval ``[low, high]`` for ``successes`` out of ``count``.

    Raises ``ValueError`` if ``count`` is not positive.
    """
    if count <= 0:
        raise ValueError("Confidence interval needs observations")
    p, z = successes / count, 1.959963984540054
    scale = 1 + z * z / count
    center = (p + z * z / (2 * count)) / scale
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / scale
    return [max(0., center - radius), min(1., center + radius)]


def bootstrap_mean(values, rng, repeats=2000):
    """Return the 95% bootstrap interval of the mean of ``values`` as ``[low, high]``."""
    values = np.asarray(values, dtype=float)
    draws = rng.integers(len(values), size=(repeats, len(values)))
    return np.quantile(values[draws].mean(1), [.025, .975]).tolist()


def fetch_source(spec):
    """Load the dataset described by ``spec`` and return ``(rows, excluded)``.

    ``rows`` maps ``train`` and ``test`` to ``{"text", "label", "row", "sha256"}`` dicts with duplicate texts
    and out-of-scope labels removed; ``excluded`` counts what was dropped.
    """
    from datasets import load_dataset
    if spec["config"] == "default":
        dataset = load_dataset(spec["repository"], spec["config"], revision=spec["revision"])
    else:
        # These historical parquet commits predate dataset-card config declarations.
        root = f"hf://datasets/{spec['repository']}@{spec['revision']}/{spec['config']}"
        dataset = load_dataset("parquet", data_files={
            split: f"{root}/{split}/*.parquet" for split in ("train", "test")
        })
    names = dataset["train"].features[spec["label"]].names
    seen, rows, excluded = set(), {"train": [], "test": []}, {"duplicates": 0, "out_of_scope": 0}
    for split in rows:
        for index, row in enumerate(dataset[split]):
            label = int(row[spec["label"]])
            if names[label].lower() in {"oos", "out_of_scope"}:
                excluded["out_of_scope"] += 1
                continue
            value = row["text"]
            key = text_hash(value)
            if key in seen:
                excluded["duplicates"] += 1
                continue
            seen.add(key)
            rows[split].append({"text": value, "label": label, "row": index, "sha256": key})
    return rows, excluded


def geometric_tasks(train, train_y, test, test_y, protocol, seed):
    """Score the nearest-reference, 7+1 membership and 190+10 membership tasks on embedded splits.

    Returns ``(metrics, arrays)``: accuracy or recall with intervals per task, and the sampled rows and
    predictions. Raises ``ValueError`` when a label lacks enough distinct rows.
    """
    rng = np.random.default_rng(seed)
    labels = sorted(set(train_y) & set(test_y))
    references, queries = [], []
    for label in labels:
        pool, held = np.flatnonzero(train_y == label), np.flatnonzero(test_y == label)
        shots = protocol["nearest_reference"]["train_references_per_label"]
        count = protocol["nearest_reference"]["test_queries_per_label"]
        if len(pool) < shots or len(held) < count:
            raise ValueError(f"Insufficient distinct rows for label {label}; do not borrow or resample")
        references.extend(rng.choice(pool, shots, replace=False))
        queries.extend(rng.choice(held, count, replace=False))
    references, queries = np.array(references), np.array(queries)
    predicted, margins = nearest_decision(test[queries] @ train[references].T, train_y[references])
    correct = predicted == test_y[queries]
    label_means = [correct[test_y[queries] == label].mean() for label in labels]
    nearest = {"accuracy": float(correct.mean()),
               "ci95": bootstrap_mean(label_means, rng), "queries": len(queries),
               "references": len(references), "bootstrap_unit": "intent label"}
    grouped = {label: np.flatnonzero(train_y == label) for label in labels}
    small_targets, small_predictions, small_ids = [], [], []
    eligible = [label for label in labels if len(grouped[label]) >= 7]
    for _ in range(protocol["membership_small"]["trials"]):
        label = int(rng.choice(eligible))
        other = int(rng.choice([value for value in labels if value != label]))
        ids = np.r_[rng.choice(grouped[label], 7, replace=False), rng.choice(grouped[other])]
        order = rng.permutation(8)
        ids = ids[order]
        small_ids.append(ids)
        small_targets.append(int(np.flatnonzero(order == 7)[0]))
        small_predictions.append(int(leave_one_out_distances(train[ids]).argmax()))
    small_correct = np.array(small_targets) == np.array(small_predictions)
    small = {"accuracy": float(small_correct.mean()), "trials": len(small_correct),
             "ci95": wilson(int(small_correct.sum()), len(small_correct))}
    recalls, large_ids = [], []
    for _ in range(protocol["membership_large"]["trials"]):
        mix = rng.choice(labels, 5, replace=False)
        inside = np.concatenate([grouped[label] for label in mix])
        outside = np.concatenate([grouped[label] for label in labels if label not in mix])
        if len(inside) < 190:
            raise ValueError("190+10 requires 190 distinct inliers")
        ids = np.r_[rng.choice(inside, 190, replace=False), rng.choice(outside, 10, replace=False)]
        large_ids.append(ids)
        recalls.append(float(np.mean(np.argsort(knn_outlier_scores(train[ids]))[-10:] >= 190)))
    large = {"recall_at10": float(np.mean(recalls)), "trials": len(recalls),
             "ci95": bootstrap_mean(recalls, rng)}
    arrays = {"reference_rows": references, "query_rows": queries, "nearest_predictions": predicted,
              "nearest_targets": test_y[queries], "nearest_margins": margins,
              "small_rows": np.array(small_ids), "small_predictions": np.array(small_predictions),
              "small_targets": np.array(small_targets), "large_rows": np.array(large_ids),
              "large_recalls": np.array(recalls)}
    return {"nearest_reference": nearest, "membership_7plus1": small,
            "membership_190plus10": large}, arrays


def shift_tasks(vectors, targets, protocol, seed):
    """Measure population-shift rejection rates under null and shifted conditions.

    Returns ``(metrics, arrays)`` with per-condition rejection rate, interval and the sampled p-values and
    rows. The null fixes identical intent counts in both populations: this is
    a balanced, stratified finite-corpus null, not IID operational calibration.
    Raises ``ValueError`` without ten labels of at least 25 rows.
    """
    rng = np.random.default_rng(seed)
    eligible = [int(label) for label in np.unique(targets) if np.sum(targets == label) >= 25]
    if len(eligible) < 10:
        raise ValueError("Shift test needs ten labels with 25 distinct rows")
    p_values = {"null": [], "shifted": []}
    sampled = {"null": [], "shifted": []}
    spec = protocol["population_shift"]
    for kind in p_values:
        for trial in range(spec["trials_per_condition"]):
            labels = rng.choice(eligible, 10, replace=False)
            a, b = [], []
            counts = spec["null_counts"] if kind == "null" else spec["shift_counts"]
            for label, left, right in zip(labels, spec["null_counts"], counts):
                ids = rng.choice(np.flatnonzero(targets == label), left + right, replace=False)
                a.extend(ids[:left])
                b.extend(ids[left:])
            if set(a) & set(b):
                raise ValueError("Population overlap")
            answer = population_shift(vectors[a], vectors[b], bandwidth=spec["bandwidth"],
                                      permutations=spec["permutations"], seed=seed + trial,
                                      alpha=spec["alpha"])
            p_values[kind].append(answer["p_value"])
            sampled[kind].append([a, b])
    result = {}
    for kind, values in p_values.items():
        count = int(np.sum(np.array(values) <= spec["alpha"]))
        result[kind] = {"rejection_rate": count / len(values), "ci95": wilson(count, len(values)),
                        "trials": len(values)}
    arrays = {f"shift_{kind}_pvalues": np.array(values) for kind, values in p_values.items()}
    arrays.update({f"shift_{kind}_rows": np.array(values) for kind, values in sampled.items()})
    return result, arrays


def run(args):
    """Run the benchmark in ``args.protocol``, writing splits, predictions and ``summary.json`` under ``args.output``.

    Raises ``ValueError`` if the protocol's reader pin or an existing frozen protocol disagrees.
    """
    from venn.reproduction import preflight
    environment = preflight(args.gpus, args.output, args.cache)
    protocol = json.loads(args.protocol.read_text())
    if protocol["reader"] != MODEL_ID or protocol["reader_revision"] != REVISION:
        raise ValueError("Frozen profile mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = args.output / "protocol.json"
    if frozen.exists() and json.loads(frozen.read_text()) != protocol:
        raise ValueError("Protocol changed; use a new version and output")
    write(frozen, protocol)
    results = {"benchmark": protocol["name"], "version": protocol["version"],
               "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
               "implementation": environment["implementation"],
               "datasets": {}, "scope": protocol["scope"], "embedding": {}}
    sources = {}
    for name, spec in protocol["sources"].items():
        rows, excluded = fetch_source(spec)
        sources[name] = rows
        write(args.output / f"{name}-splits.json", {
            "source": spec, "excluded": excluded,
            "rows": {split: [{k: v for k, v in row.items() if k != "text"} for row in values]
                     for split, values in rows.items()}})
    texts = [row["text"] for rows in sources.values() for values in rows.values() for row in values]
    vectors, telemetry = embed_cached(texts, args.cache, args.output, args.gpus)
    results["embedding"] = telemetry
    offset = 0
    for i, (name, rows) in enumerate(sources.items()):
        partitions, targets = {}, {}
        for split, values in rows.items():
            partitions[split] = vectors[offset:offset + len(values)]
            targets[split] = np.array([row["label"] for row in values])
            offset += len(values)
        start = time.monotonic()
        score, raw = geometric_tasks(partitions["train"], targets["train"], partitions["test"],
                                    targets["test"], protocol, protocol["seed"] + i)
        shift, shift_raw = shift_tasks(partitions["test"], targets["test"], protocol, protocol["seed"] + 100 + i)
        np.savez_compressed(args.output / f"{name}-predictions.npz", **raw, **shift_raw)
        results["datasets"][name] = {**score, "population_shift": shift,
                                    "evaluation_seconds": time.monotonic() - start}
    write(args.output / "summary.json", results)
    print(json.dumps(results, indent=2))


PROTOCOL_NAME = Path("benchmarks") / "vennspace-v1.json"


def default_protocol():
    """The protocol beside an editable source checkout, else in the current directory (a checkout with a wheel install)."""
    for root in (Path(__file__).resolve().parents[1], Path.cwd()):
        if (root / PROTOCOL_NAME).is_file():
            return root / PROTOCOL_NAME
    return None


def main():
    """Command-line entry point: parse ``--protocol``, ``--output``, ``--cache`` and ``--gpus`` and run the benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path,
                        help="Protocol JSON (default: benchmarks/vennspace-v1.json in the source checkout)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    args.protocol = args.protocol or default_protocol()
    if args.protocol is None or not args.protocol.is_file():
        parser.error(f"No protocol at {args.protocol or PROTOCOL_NAME}. The VennSpace-1 protocol ships with the source "
                     "checkout, not the wheel: run from a clone of the repository or pass --protocol")
    run(args)


if __name__ == "__main__":
    main()
