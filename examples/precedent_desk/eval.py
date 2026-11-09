"""Precedent Desk: reproduce every number the demo shows from published Banking77 vectors.

The store is the Banking77 training split (10,003 closed tickets, 77 queues); the stream is the test split
(3,080 tickets). Everything runs on CPU from ``banking77-vectors.npz`` (EmbeddingGemma 2 vectors under the
pinned ``venn-1`` profile, float16) through the public ``DecisionService`` or the same arithmetic it uses.

Items, each written to ``<output>/<item>.json`` and merged into ``<output>/results.json``:

  a  history curve: nearest reference with 5 / 10 / 20 / 50 / all references per queue, 25 store draws
  b  zero-shot: Choice over the 77 queue names, with and without the abstention option
  c  label noise: 10 / 20 / 30 percent of store labels reassigned, 25 draws
  d  registration: a family head at 20 and 35 examples per queue, 25 draws, with the p >= 0.95 tail; each head
     trains on a 60/20/20 train/selection/report split within each queue, passed explicitly
  e  matched Jev run through OpenRouter (needs an API key, or a complete <output>/jev-responses.jsonl from an
     earlier run); otherwise the published 81.0 placeholder
  f  one new experience: 7 queues held out, 0 / 1 / 2 / 3 / 5 resolved tickets written back
  g  backlog sweep: IndexApply over all 13,083 tickets with the FX question, scored and timed
  h  label-string gate: zero-shot with fresh descriptions and with opaque codes; names vs codes on held-out queues
  i  k-vote nearest reference under label noise (report only)
  j  CLINC150 matched pair: Jev 1.13 and Venn zero-shot over the 150 intent names, same protocol
  k  CLINC150 history curve: nearest reference at 5 / 10 / 20 / 50 / all training rows per intent (added after j)

Provenance: VennSpace-1's Banking77 nearest-reference row (92.8%) is a 20-references-per-queue, 20-queries-per-queue
cell (1,540 queries). The full-split numbers this harness measures are different cells: zero-shot 92.1%, all closed
tickets 93.6%, Jev 79.7% (matched run).

    python examples/precedent_desk/eval.py --vectors banking77-vectors.npz --output /tmp/precedent-desk \\
        --model-dir "$(venn download-model)" --items abcfg

Numbers appear in the demo only if this script wrote them. Uncertainty has two sources and both are
reported: test-row variation (row bootstrap) and which tickets the store kept (store draws).
"""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from venn.benchmark import bootstrap_mean, text_hash, wilson
from venn.index.decision import nearest_decision
from venn.reproduction import file_sha256, implementation_provenance

ITEMS = "abcdefghijk"
CLINC_SOURCE = {"repository": "clinc/clinc_oos", "revision": "8e951f061d8a5bedeaec359c02de3185e8c3359a",
                "config": "plus", "label": "intent"}
BANKING_SOURCE = {"repository": "PolyAI/banking77", "revision": "5724874eb34f7571b611868e14cba91202b9b2a9",
                  "config": "default", "label": "label"}
SEED = 20261105
STOPS = (5, 10, 20, 50, "all")
NOISE = (0.1, 0.2, 0.3)
SIZES = (20, 35)
COUNTS = (0, 1, 2, 3, 5)
HELD_OUT, CHOICES = 7, 10
THRESHOLDS = (0.8, 0.9, 0.95)
FX_TRUTH = ("card_payment_wrong_exchange_rate", "exchange_rate")
FX_DIAGNOSTIC = ("exchange_charge", "exchange_via_app", "wrong_exchange_rate_for_cash_withdrawal")
QUEUE_INSTRUCTION = "Which banking support queue handles this customer message?"
JEV_MODEL = "typesafe/jev-1.13-20260917"
JEV_INSTRUCTION = "Which banking support queue handles this customer message? Choose the single best match."
PLACEHOLDER_JEV = {"accuracy": 0.81, "source": "OpenRouter blog, Sep 22, 2026 (third party, prompt unknown)"}
GENERIC_TEMPERATURE_FALLBACK = 1.048267123796744  # venn-1 choice_temperature; the service's own value is used when loaded


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def log(*values):
    print(time.strftime("%H:%M:%S"), *values, flush=True)


def unit(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def load(path):
    data = np.load(path, allow_pickle=False)
    names = data["intent_names"].tolist()
    return {
        "train": unit(data["train_vectors"]), "train_y": data["train_labels"].astype(np.int64), "train_sha": data["train_sha256"].tolist(),
        "test": unit(data["test_vectors"]), "test_y": data["test_labels"].astype(np.int64), "test_sha": data["test_sha256"].tolist(),
        "names": names, "options": unit(data["option_vectors"]), "option_names": data["option_names"].tolist(),
        "queue_instruction": unit(data["queue_instruction"]), "family_instruction": unit(data["family_instruction"]),
        "fx_instruction": unit(data["fx_instruction"]), "fx_options": unit(data["fx_option_vectors"]),
        "fx_labels": data["fx_option_labels"].tolist(),
    }


def overlap_diagnostic(V):
    """Describe official-split overlap without changing the historical evaluation."""
    train = {}
    for digest, label in zip(V["train_sha"], V["train_y"]):
        train.setdefault(digest, set()).add(int(label))
    matched = [(digest, int(label)) for digest, label in zip(V["test_sha"], V["test_y"])
               if digest in train]
    return {"rows": len(V["train_sha"]) + len(V["test_sha"]),
            "normalized_unique_texts": len(set(V["train_sha"]) | set(V["test_sha"])),
            "test_rows_matching_train": len(matched),
            "same_label_test_matches": sum(label in train[digest] for digest, label in matched),
            "evaluation": "official splits retained; diagnostic only, no deduplication or rescoring"}


def validate_clinc_rows(data, rows, split):
    """Check ordered text hashes, labels and original row IDs against the pinned source."""
    expected = rows[split]
    if (data[f"{split}_sha256"].tolist() != [row["sha256"] for row in expected]
            or data[f"{split}_labels"].tolist() != [row["label"] for row in expected]
            or data[f"{split}_rows"].tolist() != [row["row"] for row in expected]
            or len(data[f"{split}_vectors"]) != len(expected)):
        raise ValueError(f"CLINC vectors do not match the pinned {split} rows")


def input_provenance(item, args):
    """Identify vector inputs and existing local artifacts used by an item."""
    paths = [Path(args.vectors)]
    if item == "h" and args.model_dir:
        paths.append(Path(args.gate_vectors))
    if (item == "j" and args.model_dir) or item == "k":
        paths.append(Path(args.clinc_vectors))
    if item == "k":
        paths.append(Path(args.clinc_train_vectors))
    for name in (["e.json"] if item in "ach" else
                 ["j.json"] if item == "k" else
                 ["jev-responses.jsonl"] if item == "e" else
                 ["jev-clinc-responses.jsonl"] if item == "j" and args.model_dir else []):
        path = Path(args.output) / name
        if path.is_file():
            paths.append(path)
    if args.model_dir:
        directory = Path(args.model_dir)
        manifest = directory / "manifest.json"
        paths.append(manifest)
        paths.extend(directory / name for name in json.loads(manifest.read_text())["sha256"])
        registry = Path(args.output) / "venn-home" / "registry"
        paths.extend(path for path in registry.glob("*") if path.suffix in {".json", ".npz"})
    return {"files_sha256": {str(path): file_sha256(path) for path in paths},
            "sources": {"banking77": BANKING_SOURCE,
                        **({"clinc150": CLINC_SOURCE} if item in "jk" else {})}}


def check_texts(V, spec):
    """Re-read the pinned Banking77 rows and confirm the vectors were made from exactly those texts."""
    from datasets import load_dataset

    dataset = load_dataset(spec["repository"], spec["config"], revision=spec["revision"])
    for split in ("train", "test"):
        hashes = [text_hash(t) for t in dataset[split]["text"]]
        if hashes != V[f"{split}_sha"] or dataset[split]["label"] != V[f"{split}_y"].tolist():
            raise ValueError(f"The {split} vectors do not match the pinned Banking77 rows")
    return {split: [{"text": t, "label": int(l)} for t, l in zip(dataset[split]["text"], dataset[split]["label"])]
            for split in ("train", "test")}


def row_bootstrap(correct, rng, repeats=1000):
    correct = np.asarray(correct, dtype=np.float32)
    rows = rng.integers(len(correct), size=(repeats, len(correct)))
    return correct[rows].mean(1)


def interval(samples):
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def draw_store(rng, labels, per_intent):
    """Indices of up to ``per_intent`` rows per intent (all rows when ``per_intent`` is "all")."""
    if per_intent == "all":
        return np.arange(len(labels))
    picks = []
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        picks.append(rng.choice(rows, min(per_intent, len(rows)), replace=False))
    return np.sort(np.concatenate(picks))


def predict(queries, references, reference_labels):
    """Top-1 by label with the public nearest-reference arithmetic; returns (predicted, margin)."""
    return nearest_decision(queries @ references.T, reference_labels)


def jev_interval(output):
    """Jev's accuracy interval: the matched run if it exists, otherwise Wilson on the placeholder."""
    path = Path(output) / "e.json"
    if path.exists():
        record = json.loads(path.read_text())
        if record.get("status") == "measured":
            return record["accuracy"], record["ci95"], "9f-e matched run"
    rows = 3080
    return PLACEHOLDER_JEV["accuracy"], wilson(int(round(PLACEHOLDER_JEV["accuracy"] * rows)), rows), PLACEHOLDER_JEV["source"]


def service_for(args):
    from venn.server import DecisionService

    home = Path(args.output) / "venn-home"
    os.environ["VENN_HOME"] = str(home)
    return DecisionService(model_dir=args.model_dir, cache_path=home / "cache.sqlite", registry_path=home / "registry" / "calibrations.json")


# ----------------------------------------------------------------------------- a. history curve
def item_a(V, args):
    rng = np.random.default_rng(SEED)
    jev_point, jev_ci, jev_source = jev_interval(args.output)
    stops, crossovers = {}, []
    for stop in STOPS:
        draws = 1 if stop == "all" else args.draws
        per_draw, pooled, rows = [], [], []
        for draw in range(draws):
            keep = draw_store(np.random.default_rng(SEED + 1000 * draw + 1), V["train_y"], stop)
            predicted, _ = predict(V["test"], V["train"][keep], V["train_y"][keep])
            correct = predicted == V["test_y"]
            per_draw.append(float(correct.mean()))
            pooled.append(row_bootstrap(correct, rng))
            rows.append(len(keep))
        pooled = np.concatenate(pooled)
        stops[str(stop)] = {"accuracy": float(np.mean(per_draw)), "per_draw": per_draw, "draws": draws,
                            "ci95_pooled": interval(pooled), "draw_spread": [min(per_draw), max(per_draw)],
                            "store_rows_mean": float(np.mean(rows)), "store_rows": rows if stop != "all" else rows[0],
                            "clears_jev": interval(pooled)[0] > jev_ci[1]}
        per_draw_cross = [acc > jev_ci[1] for acc in per_draw]
        crossovers.append({"stop": stop, "share_of_draws_above_jev_upper": float(np.mean(per_draw_cross))})
    crossover = next((s for s in STOPS if s != "all" and stops[str(s)]["clears_jev"]), None)
    # API cross-check at "all": the public NearestReference answer must equal the arithmetic above.
    api = None
    if args.model_dir:
        service = service_for(args)
        started = time.perf_counter()
        handle = service.register_index(V["train"], [f"train:{i}" for i in range(len(V["train"]))],
                                        outcomes=[V["names"][y] for y in V["train_y"]])
        registration = time.perf_counter() - started
        predicted, _ = predict(V["test"], V["train"], V["train_y"])
        agree, latencies = 0, []
        for i in range(len(V["test"])):
            t0 = time.perf_counter()
            answer = service.decide({"state": {"vector": V["test"][i].tolist()},
                                     "questions": {"precedent": {"type": "NearestReference", "index_ref": handle, "k": 3}}})["answers"]["precedent"]
            latencies.append((time.perf_counter() - t0) * 1000)
            agree += int(answer["answer"] == V["names"][predicted[i]])
        api = {"agreement_with_arithmetic": agree / len(V["test"]), "registration_seconds": registration,
               "decide_ms_p50": float(np.percentile(latencies, 50)), "decide_ms_p95": float(np.percentile(latencies, 95)),
               "note": "one vector state per call through DecisionService.decide, CPU, store of 10,003 rows"}
    return {"item": "9f-a", "stops": stops, "crossover_stop": crossover, "crossover_rule": "smallest stop whose pooled 95% interval lower bound exceeds Jev's interval upper bound",
            "crossover_per_draw": crossovers, "jev": {"accuracy": jev_point, "ci95": jev_ci, "source": jev_source},
            "queries": int(len(V["test"])), "row_bootstrap_resamples": 1000, "api_cross_check": api,
            "note": "intents with fewer training rows than a stop contribute all their rows (the smallest intent has 35)"}


# ----------------------------------------------------------------------------- b. zero-shot
def zero_shot_distributions(service, V):
    options = [{"label": name, "vector": vector.tolist()} for name, vector in zip(V["names"], V["options"])]
    question = {"type": "Choice", "instructions_vec": V["queue_instruction"].tolist(), "options": options}
    distributions = np.zeros((len(V["test"]), len(V["names"]) + 1), dtype=np.float64)
    for i in range(len(V["test"])):
        answer = service.decide({"state": {"vector": V["test"][i].tolist()}, "questions": {"queue": question}})["answers"]["queue"]
        distributions[i] = [answer["probabilities"][name] for name in V["names"]] + [answer["probabilities"]["none"]]
    return distributions


def item_b(V, args):
    if not args.model_dir:
        return {"item": "9f-b", "status": "skipped: the zero-shot Choice path needs the venn-1 head (--model-dir)"}
    service = service_for(args)
    distributions = zero_shot_distributions(service, V)
    truth = V["test_y"]
    without = distributions[:, :-1].argmax(1)
    with_head = distributions.argmax(1)
    abstained = with_head == len(V["names"])
    correct_without = without == truth
    correct_with = (with_head == truth) & ~abstained
    rng = np.random.default_rng(SEED)
    return {"item": "9f-b", "queries": int(len(truth)),
            "without_abstention_option": {"accuracy": float(correct_without.mean()), "ci95": wilson(int(correct_without.sum()), len(truth)),
                                          "what": "Choice over the 77 queue names, no null criterion: the centered-cosine argmax"},
            "with_abstention_option": {"accuracy": float(correct_with.mean()), "ci95": wilson(int(correct_with.sum()), len(truth)),
                                       "abstention_rate": float(abstained.mean()), "accuracy_when_answered": float((with_head[~abstained] == truth[~abstained]).mean()) if (~abstained).any() else None,
                                       "what": "the same question with an explicit no-match option; an abstention counts as wrong"},
            "ci95_row_bootstrap_without": interval(row_bootstrap(correct_without, rng))}


# ----------------------------------------------------------------------------- c. label noise
def item_c(V, args):
    rng = np.random.default_rng(SEED + 7)
    jev_point, jev_ci, _ = jev_interval(args.output)
    levels = {}
    for fraction in NOISE:
        per_draw, pooled = [], []
        for draw in range(args.draws):
            local = np.random.default_rng(SEED + 2000 + 100 * draw + int(fraction * 100))
            noisy = V["train_y"].copy()
            flip = local.random(len(noisy)) < fraction
            offsets = local.integers(1, len(V["names"]), size=int(flip.sum()))
            noisy[flip] = (noisy[flip] + offsets) % len(V["names"])
            predicted, _ = predict(V["test"], V["train"], noisy)
            correct = predicted == V["test_y"]
            per_draw.append(float(correct.mean()))
            pooled.append(row_bootstrap(correct, rng))
        pooled = np.concatenate(pooled)
        levels[str(fraction)] = {"accuracy": float(np.mean(per_draw)), "per_draw": per_draw, "ci95_pooled": interval(pooled),
                                 "draw_spread": [min(per_draw), max(per_draw)], "below_jev": interval(pooled)[1] < jev_ci[0]}
    return {"item": "9f-c", "levels": levels, "draws": args.draws, "jev": {"accuracy": jev_point, "ci95": jev_ci},
            "note": "noised labels are reassigned uniformly among the other 76 queues; the query labels stay clean"}


# ----------------------------------------------------------------------------- d. registration
def protocol_split(labels, seed):
    """The registered 60/20/20 train/selection/report split within each queue, passed to register_family explicitly."""
    from venn.heads.family import split_labels

    train, selection, report = split_labels(np.asarray(labels), seed)
    return {"train": train.tolist(), "selection": selection.tolist(), "report": report.tolist()}


def item_d(V, args):
    if not args.model_dir:
        return {"item": "9f-d", "status": "skipped: family registration needs the venn-1 head (--model-dir)"}
    service = service_for(args)
    options = [{"label": name, "vector": vector.tolist()} for name, vector in zip(V["names"], V["options"])]
    question = {"type": "Choice", "instructions_vec": V["family_instruction"].tolist(), "options": options}
    cells = {}
    for size in SIZES:
        draws = []
        for draw in range(args.draws):
            local = np.random.default_rng(SEED + 3000 + 100 * draw + size)
            keep = draw_store(local, V["train_y"], size)
            examples = [{"state": V["train"][i].tolist(), "label": V["names"][V["train_y"][i]]} for i in keep]
            name = f"desk-{size}-{draw}"
            started = time.perf_counter()
            report = service.register_family(name, examples, options, V["family_instruction"].tolist(), seed=draw,
                                             split=protocol_split(V["train_y"][keep], draw))
            seconds = time.perf_counter() - started
            entry = service.families[name]
            # Held-out report slice: the third partition, identified by the state hashes the service recorded.
            report_hashes = set(entry["split_hashes"][2])
            states = unit(V["train"][keep])
            digests = [{hashlib.sha256(states[j].tobytes()).hexdigest(), hashlib.sha256(unit(states[j]).tobytes()).hexdigest()} for j in range(len(keep))]
            report_rows = [keep[j] for j in range(len(keep)) if digests[j] & report_hashes]
            if len(report_rows) < len(report_hashes) // 2:
                raise ValueError("Could not identify the family's report slice from its state hashes")
            family_question = {**question, "family": name}
            def answers(rows):
                out = np.zeros((len(rows), len(V["names"]) + 1))
                for k, i in enumerate(rows):
                    answer = service.decide({"state": {"vector": states_for(V, i).tolist()}, "questions": {"queue": family_question}})["answers"]["queue"]
                    out[k] = [answer["probabilities"][n] for n in V["names"]] + [answer["probabilities"]["none"]]
                return out
            test_p = answers([("test", i) for i in range(len(V["test"]))])
            test_pred = test_p[:, :-1].argmax(1)
            test_conf = test_p[:, :-1].max(1)
            correct = test_pred == V["test_y"]
            report_p = answers([("train", i) for i in report_rows])
            report_truth = V["train_y"][report_rows]
            report_conf, report_pred = report_p[:, :-1].max(1), report_p[:, :-1].argmax(1)
            covered = report_conf >= 0.95
            nearest_pred, _ = predict(V["test"], V["train"][keep], V["train_y"][keep])
            draws.append({
                "draw": draw, "registration_seconds": seconds, "examples": len(keep), "report_items": len(report_rows),
                "calibrated": bool(report["calibrated"]), "report_accuracy": report["report_accuracy"], "report_ece": report["report_ece"],
                "report_ece_ci95": report.get("report_ece_ci95"), "additional_report_items_needed": report.get("additional_report_items_needed"),
                "test_accuracy": float(correct.mean()), "test_ece": ece(test_conf, correct),
                "nearest_reference_same_store_accuracy": float((nearest_pred == V["test_y"]).mean()),
                "tail_p95": {"coverage": float(covered.mean()), "precision": float((report_pred[covered] == report_truth[covered]).mean()) if covered.any() else None,
                             "covered_items": int(covered.sum())},
            })
            log(f"9f-d size {size} draw {draw}: test {draws[-1]['test_accuracy']:.4f} vs nearest {draws[-1]['nearest_reference_same_store_accuracy']:.4f}, ece {draws[-1]['test_ece']:.4f}, {seconds:.0f} s")
        rng = np.random.default_rng(SEED)
        accuracy = [d["test_accuracy"] for d in draws]
        nearest = [d["nearest_reference_same_store_accuracy"] for d in draws]
        precision = [d["tail_p95"]["precision"] for d in draws if d["tail_p95"]["precision"] is not None]
        coverage = [d["tail_p95"]["coverage"] for d in draws]
        cells[str(size)] = {
            "draws": draws, "test_accuracy": float(np.mean(accuracy)), "test_accuracy_ci95_over_draws": bootstrap_mean(accuracy, rng),
            "nearest_reference_same_store": float(np.mean(nearest)), "head_minus_nearest": float(np.mean(accuracy) - np.mean(nearest)),
            "head_minus_nearest_ci95": bootstrap_mean([a - b for a, b in zip(accuracy, nearest)], rng),
            "test_ece": float(np.mean([d["test_ece"] for d in draws])), "report_ece": float(np.mean([d["report_ece"] for d in draws])),
            "calibrated_share": float(np.mean([d["calibrated"] for d in draws])), "report_items_mean": float(np.mean([d["report_items"] for d in draws])),
            "tail_p95": {"precision": float(np.mean(precision)) if precision else None, "precision_ci95": bootstrap_mean(precision, rng) if precision else None,
                         "coverage": float(np.mean(coverage)), "coverage_ci95": bootstrap_mean(coverage, rng)},
            "registration_seconds": {"mean": float(np.mean([d["registration_seconds"] for d in draws])), "max": float(max(d["registration_seconds"] for d in draws))},
            "claim_allowed": bootstrap_mean([a - b for a, b in zip(accuracy, nearest)], rng)[0] > 0,
        }
    return {"item": "9f-d", "cells": cells, "draws": args.draws, "queries": int(len(V["test"])),
            "note": "test ECE uses 15 equal-width bins on the top option probability; the report slice is the service's own held-out partition"}


def states_for(V, key):
    split, i = key
    return V[split][i]


def ece(confidence, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


# ----------------------------------------------------------------------------- e. matched Jev run
def item_e(V, args, texts):
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    key = os.environ.get(args.jev_key_env, "")
    cache_path = Path(args.output) / "jev-responses.jsonl"
    cached = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            record = json.loads(line)
            cached[record["request_sha256"]] = record

    def placeholder(note):
        acc = PLACEHOLDER_JEV["accuracy"]
        return {"item": "9f-e", "status": "placeholder", "accuracy": acc, "ci95": wilson(int(round(acc * len(V["test"]))), len(V["test"])),
                **PLACEHOLDER_JEV, "note": note}

    if not key and not cached:
        return placeholder(f"set {args.jev_key_env} to run the matched comparison (about 3,080 requests)")
    criteria = {name: name for name in V["option_names"]}
    name_of = dict(zip(V["option_names"], V["names"]))

    def ask(i):
        body = {"model": JEV_MODEL, "state": texts["test"][i]["text"],
                "questions": {"queue": {"type": "choice", "instructions": JEV_INSTRUCTION, "criteria": criteria}}}
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if digest in cached:
            return cached[digest]
        if not key:  # cache-only run: never make a request without a key
            return None
        request = urllib.request.Request(args.jev_endpoint, data=json.dumps(body).encode(), method="POST",
                                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read())
        answer = payload["answers"]["queue"]
        return {"request_sha256": digest, "row": i, "choice": answer.get("choice"), "confidence": answer.get("confidence"),
                "model": payload.get("model"), "usage": payload.get("usage"), "latency_ms": (time.perf_counter() - started) * 1000}

    records = []
    with ThreadPoolExecutor(max_workers=4) as pool, cache_path.open("a") as stream:
        for record in pool.map(ask, range(len(V["test"]))):
            if record is not None and record["request_sha256"] not in cached:
                stream.write(json.dumps(record) + "\n")
                cached[record["request_sha256"]] = record
            records.append(record)
    missing = sum(record is None for record in records)
    if missing:
        return placeholder(f"{cache_path} lacks {missing} of {len(records)} responses; "
                           f"set {args.jev_key_env} to request them (about 3,080 requests in all)")
    predicted = np.array([V["names"].index(name_of[r["choice"]]) if r["choice"] in name_of else -1 for r in records])
    correct = predicted == V["test_y"]
    confidence = np.array([r["confidence"] if r["confidence"] is not None else np.nan for r in records], dtype=float)
    bins = np.linspace(0, 1, 11)
    histogram = []
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidence > low) & (confidence <= high)
        histogram.append({"bin": [float(low), float(high)], "count": int(mask.sum()), "accuracy": float(correct[mask].mean()) if mask.any() else None})
    cost = sum((r.get("usage") or {}).get("cost", 0) or 0 for r in records)
    return {"item": "9f-e", "status": "measured", "accuracy": float(correct.mean()), "ci95": wilson(int(correct.sum()), len(correct)),
            "model": records[0].get("model"), "requests": len(records), "cost_usd": cost, "unparsed_choices": int((predicted < 0).sum()),
            "latency_ms_p50": float(np.nanmedian([r["latency_ms"] for r in records if "latency_ms" in r])) if any("latency_ms" in r for r in records) else None,
            "confidence_histogram": histogram, "prompt": {"instructions": JEV_INSTRUCTION, "criteria": "the 77 queue names, each described by itself"}}


# ----------------------------------------------------------------------------- f. one new experience
def membership_flag(candidate, members):
    """The public SetMembership rule: flagged when the candidate's k-th member distance exceeds the members' own 95th percentile."""
    k = min(10, len(members) - 1)
    distances = np.maximum(0, 1 - members @ members.T)
    np.fill_diagonal(distances, np.inf)
    reference = np.partition(distances, k - 1, axis=1)[:, k - 1]
    distance = float(np.partition(1 - members @ candidate, k - 1)[k - 1])
    return distance > float(np.quantile(reference, .95))


def item_f(V, args, generic_temperature):
    names = V["names"]
    cells = []
    for choice in range(CHOICES):
        rng = np.random.default_rng(SEED + 4000 + choice)
        held = np.sort(rng.choice(len(names), HELD_OUT, replace=False))
        base_rows = np.flatnonzero(~np.isin(V["train_y"], held))
        in_store_queries = np.flatnonzero(~np.isin(V["test_y"], held))
        _, margins = predict(V["test"][in_store_queries], V["train"][base_rows], V["train_y"][base_rows])
        tau = float(np.percentile(margins, 10))  # coverage 0.90 on the queues the store knows
        for intent in held:
            rows = np.flatnonzero(V["test_y"] == intent)
            order = rows[np.random.default_rng(SEED + 5000 + 100 * choice + int(intent)).permutation(len(rows))]
            record = {"choice": choice, "intent": int(intent), "name": names[intent], "tau_margin": tau, "counts": {}}
            for count in COUNTS:
                written, remaining = order[:count], order[count:]
                references = np.concatenate([V["train"][base_rows], V["test"][written]])
                reference_labels = np.concatenate([V["train_y"][base_rows], V["test_y"][written]])
                predicted, margin = predict(V["test"][remaining], references, reference_labels)
                correct = predicted == intent
                cell = {"accuracy": float(correct.mean()), "remaining": int(len(remaining)), "nearest_label_mode": names[int(np.bincount(predicted).argmax())]}
                if count == 0:
                    first = order[0]
                    p_first, m_first = predict(V["test"][first:first + 1], V["train"][base_rows], V["train_y"][base_rows])
                    probability = 1 / (1 + np.exp(-m_first[0] / generic_temperature))
                    members = V["train"][base_rows][V["train_y"][base_rows] == p_first[0]]
                    cell.update({"first_arrival": {"nearest_label": names[int(p_first[0])], "margin": float(m_first[0]), "probability": float(probability),
                                                   "none_by_tau": bool(m_first[0] < tau), "none_at_p0.5": bool(probability < .5), "none_at_p0.7": bool(probability < .7),
                                                   "none_at_p0.9": bool(probability < .9), "none_by_membership": bool(membership_flag(V["test"][first], members))},
                                 "all_rows_none_by_tau_rate": float((margin < tau).mean())})
                record["counts"][str(count)] = cell
            cells.append(record)
        log(f"9f-f choice {choice}: held out {[names[i] for i in held]}")
    rng = np.random.default_rng(SEED)
    summary = {}
    for count in COUNTS:
        values = [c["counts"][str(count)]["accuracy"] for c in cells]
        summary[str(count)] = {"accuracy": float(np.mean(values)), "ci95": bootstrap_mean(values, rng, repeats=1000), "cells": len(values)}
    first = [c["counts"]["0"]["first_arrival"] for c in cells]
    none_rates = {key: float(np.mean([f[key] for f in first])) for key in ("none_by_tau", "none_at_p0.5", "none_at_p0.7", "none_at_p0.9", "none_by_membership")}
    none_rates["none_by_tau_ci95"] = bootstrap_mean([float(f["none_by_tau"]) for f in first], rng, repeats=1000)
    in_store_margins = [c["tau_margin"] for c in cells]
    gate = {"count_1_lower_bound": summary["1"]["ci95"][0], "count_1_passes": summary["1"]["ci95"][0] > .5,
            "count_0_none_rate": none_rates["none_by_tau"], "count_0_passes": none_rates["none_by_tau"] >= .8}
    gate["passed"] = gate["count_1_passes"] and gate["count_0_passes"]
    return {"item": "9f-f", "summary": summary, "first_arrival_none_rates": none_rates, "gate": gate, "cells": cells,
            "uncertainty_scope": "historical CI resamples 70 intent-within-choice cells, not ten held-out-choice clusters; shared stores, thresholds and repeated intents may correlate cells. Gate result retained, not a cluster-aware generalization guarantee",
            "abstention_rule": "tau is the 10th percentile of nearest-reference margins on the test rows of the queues the store knows (coverage 0.90), per choice",
            "generic_temperature": generic_temperature, "tau_margin_mean": float(np.mean(in_store_margins)),
            "note": "accuracy is over the held-out queue's remaining test rows against a store of the other 70 queues plus the written rows"}


# ----------------------------------------------------------------------------- g. backlog sweep
def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def item_g(V, args):
    if not args.model_dir:
        return {"item": "9f-g", "status": "skipped: IndexApply needs the venn-1 head (--model-dir)"}
    service = service_for(args)
    vectors = np.concatenate([V["train"], V["test"]])
    labels = np.concatenate([V["train_y"], V["test_y"]])
    ids = [f"train:{i}" for i in range(len(V["train"]))] + [f"test:{i}" for i in range(len(V["test"]))]
    started = time.perf_counter()
    handle = service.register_index(vectors, ids)
    registration = time.perf_counter() - started
    truth = np.isin(labels, [V["names"].index(n) for n in FX_TRUTH])
    diagnostic = np.isin(labels, [V["names"].index(n) for n in FX_DIAGNOSTIC])
    options = [{"label": label, "vector": vector.tolist()} for label, vector in zip(V["fx_labels"], V["fx_options"])]
    question = {"type": "IndexApply", "instructions_vec": V["fx_instruction"].tolist(), "options": options, "batch_size": 65536}
    results, timings = {}, {}
    for threshold in THRESHOLDS:
        durations, rows = [], None
        for _ in range(5):
            t0 = time.perf_counter()
            lines = list(service.stream_index(handle, {**question, "threshold": threshold}))
            rows = [row for line in lines for row in json.loads(line)["rows"]]
            durations.append(time.perf_counter() - t0)
        answer = np.array([row["answer"] for row in rows])
        hit, review = answer == "affected", answer == "none"
        tp = int((hit & truth).sum())
        results[str(threshold)] = {"hits": int(hit.sum()), "review": int(review.sum()), "not_affected": int((answer == "not_affected").sum()),
                                   "precision": tp / max(1, int(hit.sum())), "precision_ci95": wilson(tp, int(hit.sum())) if hit.any() else None,
                                   "recall": tp / int(truth.sum()), "recall_ci95": wilson(tp, int(truth.sum())),
                                   "hits_in_diagnostic_fx_intents": int((hit & diagnostic).sum()), "hits_elsewhere": int((hit & ~truth & ~diagnostic).sum()),
                                   "truth_in_review": int((review & truth).sum())}
        timings[str(threshold)] = {"in_process_seconds_median": float(np.median(durations)), "repeats": 5,
                                   "boundary": "scoring, NDJSON serialization and consumption of every line inside one process; excludes index registration"}
    loopback = None
    if not args.skip_http:
        loopback = loopback_sweep(args, vectors, ids, question)
    # Exploratory, added after the pre-registered two-option row was measured: the same one-pass sweep with the
    # 77 queue names as options, counting rows the sweep files under either FX queue. Reported beside, never instead.
    queue_options = [{"label": name, "vector": vector.tolist()} for name, vector in zip(V["names"], V["options"])]
    exploratory = {}
    for threshold in (None, 0.9):
        q = {"type": "IndexApply", "instructions_vec": V["queue_instruction"].tolist(), "options": queue_options, "batch_size": 65536}
        if threshold is not None:
            q["threshold"] = threshold
        t0 = time.perf_counter()
        rows = [row for line in service.stream_index(handle, q) for row in json.loads(line)["rows"]]
        seconds = time.perf_counter() - t0
        answer = np.array([row["answer"] for row in rows])
        hit = np.isin(answer, list(FX_TRUTH))
        tp = int((hit & truth).sum())
        exploratory[str(threshold)] = {"hits": int(hit.sum()), "review": int((answer == "none").sum()), "precision": tp / max(1, int(hit.sum())),
                                       "precision_ci95": wilson(tp, int(hit.sum())) if hit.any() else None, "recall": tp / int(truth.sum()),
                                       "recall_ci95": wilson(tp, int(truth.sum())), "hits_in_diagnostic_fx_intents": int((hit & diagnostic).sum()),
                                       "in_process_seconds": seconds, "queue_accuracy_over_all_rows": float((answer == np.array(V["names"])[labels]).mean())}
    return {"item": "9f-g", "rows": int(len(vectors)), "ground_truth_rows": int(truth.sum()), "diagnostic_fx_rows": int(diagnostic.sum()),
            "thresholds": results, "headline_threshold": "0.9", "timing": {"index_registration_seconds": registration, **timings, "loopback_http": loopback},
            "question": {"instructions": "fx_instruction vector", "options": V["fx_labels"]},
            "exploratory_77_queue_sweep": {"status": "exploratory; added after the pre-registered two-option row was measured",
                                           "question": "IndexApply with the 77 queue names as options; a hit is a row filed under either FX queue", "thresholds": exploratory}}


def loopback_sweep(args, vectors, ids, question):
    """Time the same sweep over loopback HTTP: registration, then five streamed sweeps consumed by a client."""
    import urllib.request

    port = free_port()
    code = ("import uvicorn; from venn.server import DecisionService, create_app; "
            f"uvicorn.run(create_app(DecisionService(model_dir={args.model_dir!r})), host='127.0.0.1', port={port}, log_level='error')")
    env = {**os.environ, "VENN_HOME": str(Path(args.output) / "venn-home-http")}
    process = subprocess.Popen([sys.executable, "-c", code], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(base + "/health", timeout=2).read()
                break
            except Exception:
                time.sleep(.5)
        else:
            raise RuntimeError("server did not start")
        body = json.dumps({"vectors": {"vectors": vectors.tolist()}, "ids": ids}).encode()
        t0 = time.perf_counter()
        with urllib.request.urlopen(urllib.request.Request(base + "/indexes", data=body, headers={"Content-Type": "application/json"}), timeout=600) as r:
            handle = json.loads(r.read())["index_ref"]
        registration = time.perf_counter() - t0
        durations, rows = [], 0
        payload = json.dumps({**question, "type": "index_apply", "threshold": .9}).encode()
        for _ in range(5):
            t0 = time.perf_counter()
            rows = 0
            with urllib.request.urlopen(urllib.request.Request(base + f"/index-apply/{handle}", data=payload, headers={"Content-Type": "application/json"}), timeout=600) as r:
                for line in r:
                    rows += len(json.loads(line)["rows"])
            durations.append(time.perf_counter() - t0)
        return {"registration_seconds": registration, "sweep_seconds_median": float(np.median(durations)), "rows_consumed": rows, "repeats": 5,
                "boundary": "loopback HTTP: request, scoring, NDJSON serialization, transfer and client-side parsing of every line; the JSON index upload is timed separately"}
    finally:
        process.kill()


# ----------------------------------------------------------------------------- h. label-string gate
def choice_predictions(service, V, option_vectors, instruction, queries):
    """Zero-shot Choice argmax (no abstention option) over ``option_vectors`` for each query vector."""
    options = [{"label": str(i), "vector": vector.tolist()} for i, vector in enumerate(option_vectors)]
    question = {"type": "Choice", "instructions_vec": instruction.tolist(), "options": options}
    out = np.zeros(len(queries), dtype=np.int64)
    for i in range(len(queries)):
        answer = service.decide({"state": {"vector": queries[i].tolist()}, "questions": {"q": question}})["answers"]["q"]
        probabilities = answer["probabilities"]
        out[i] = int(max((k for k in probabilities if k != "none"), key=lambda k: probabilities[k]))
    return out


def item_h(V, args):
    if not args.model_dir:
        return {"item": "9f-h", "status": "skipped: needs the venn-1 head (--model-dir)"}
    gate = np.load(args.gate_vectors, allow_pickle=False)
    if gate["intent_names"].tolist() != V["names"]:
        raise ValueError("Gate vectors do not match the queue order")
    sets = {"names": V["options"], "descriptions": unit(gate["description_vectors"]), "codes": unit(gate["code_vectors"])}
    service = service_for(args)
    jev_point, jev_ci, jev_source = jev_interval(args.output)
    rng = np.random.default_rng(SEED)
    zero_shot, predictions = {}, {}
    for key, vectors in sets.items():
        predicted = choice_predictions(service, V, vectors, V["queue_instruction"], V["test"])
        predictions[key] = predicted
        correct = predicted == V["test_y"]
        zero_shot[key] = {"accuracy": float(correct.mean()), "ci95": wilson(int(correct.sum()), len(correct)), "ci95_row_bootstrap": interval(row_bootstrap(correct, rng))}
    # Held-out queues (the 9f-f protocol): what the queue's label vector adds beyond written examples.
    names = V["names"]
    cells = []
    for choice in range(CHOICES):
        local = np.random.default_rng(SEED + 4000 + choice)
        held = np.sort(local.choice(len(names), HELD_OUT, replace=False))
        base_rows = np.flatnonzero(~np.isin(V["train_y"], held))
        for intent in held:
            rows = np.flatnonzero(V["test_y"] == intent)
            order = rows[np.random.default_rng(SEED + 5000 + 100 * choice + int(intent)).permutation(len(rows))]
            record = {"choice": choice, "intent": int(intent), "counts": {}}
            for count in COUNTS:
                written, remaining = order[:count], order[count:]
                references = np.concatenate([V["train"][base_rows], V["test"][written]])
                reference_labels = np.concatenate([V["train_y"][base_rows], V["test_y"][written]])
                cell = {}
                predicted, _ = predict(V["test"][remaining], references, reference_labels)
                cell["examples_only"] = float((predicted == intent).mean())
                for key, vectors in sets.items():
                    extra_labels = np.arange(len(names))
                    predicted, _ = predict(V["test"][remaining], np.concatenate([references, vectors]), np.concatenate([reference_labels, extra_labels]))
                    cell[f"{key}_plus_examples"] = float((predicted == intent).mean())
                    cell[f"{key}_alone"] = float((predictions[key][remaining] == intent).mean())
                record["counts"][str(count)] = cell
            cells.append(record)
    summary = {}
    keys = ["examples_only"] + [f"{k}_plus_examples" for k in sets] + [f"{k}_alone" for k in sets]
    for count in COUNTS:
        summary[str(count)] = {key: {"accuracy": float(np.mean([c["counts"][str(count)][key] for c in cells])),
                                     "ci95": bootstrap_mean([c["counts"][str(count)][key] for c in cells], rng, repeats=1000)} for key in keys}
    passed = zero_shot["descriptions"]["ci95"][0] > jev_ci[1] and zero_shot["codes"]["accuracy"] < .05
    return {"item": "9f-h", "zero_shot": zero_shot, "held_out": summary, "cells": cells,
            "jev": {"accuracy": jev_point, "ci95": jev_ci, "source": jev_source},
            "gate": {"description_lower_bound": zero_shot["descriptions"]["ci95"][0], "jev_upper_bound": jev_ci[1],
                     "code_accuracy": zero_shot["codes"]["accuracy"], "passed": bool(passed)},
            "note": "plus-examples adds one reference row per queue holding that queue's option vector, so a held-out queue is reachable by its label even with no written example"}


# ----------------------------------------------------------------------------- i. k-vote under noise
def vote_predict(queries, references, reference_labels, k=11):
    scores = queries @ references.T
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    labels = reference_labels[top]
    best, _ = nearest_decision(scores, reference_labels)
    out = np.empty(len(queries), dtype=np.int64)
    for i in range(len(queries)):
        counts = np.bincount(labels[i])
        winners = np.flatnonzero(counts == counts.max())
        out[i] = winners[0] if len(winners) == 1 else (best[i] if best[i] in winners else winners[0])
    return out


def item_i(V, args):
    rng = np.random.default_rng(SEED + 9)
    levels = {}
    for fraction in (0.0, *NOISE):
        per_draw, pooled, single = [], [], []
        for draw in range(args.draws if fraction else 1):
            local = np.random.default_rng(SEED + 2000 + 100 * draw + int(fraction * 100))
            noisy = V["train_y"].copy()
            if fraction:
                flip = local.random(len(noisy)) < fraction
                noisy[flip] = (noisy[flip] + local.integers(1, len(V["names"]), size=int(flip.sum()))) % len(V["names"])
            predicted = vote_predict(V["test"], V["train"], noisy)
            correct = predicted == V["test_y"]
            per_draw.append(float(correct.mean()))
            pooled.append(row_bootstrap(correct, rng))
            best, _ = predict(V["test"], V["train"], noisy)
            single.append(float((best == V["test_y"]).mean()))
        levels[str(fraction)] = {"k_vote_accuracy": float(np.mean(per_draw)), "ci95_pooled": interval(np.concatenate(pooled)),
                                 "best_single_reference_accuracy": float(np.mean(single)), "draws": len(per_draw)}
    return {"item": "9f-i", "k": 11, "levels": levels, "note": "majority label among the 11 nearest stored rows; ties go to the best single reference"}


# ----------------------------------------------------------------------------- j. CLINC150 matched pair
def item_j(V, args):
    if not args.model_dir:
        return {"item": "9f-j", "status": "skipped: needs the venn-1 head (--model-dir)"}
    data = np.load(args.clinc_vectors, allow_pickle=False)
    from venn.benchmark import fetch_source

    rows, _ = fetch_source(CLINC_SOURCE)
    test = rows["test"]
    validate_clinc_rows(data, rows, "test")
    queries, truth = unit(data["test_vectors"]), data["test_labels"].astype(np.int64)
    intent_names, option_names = data["intent_names"].tolist(), data["option_names"].tolist()
    in_scope = [i for i, name in enumerate(intent_names) if name != "oos"]
    options = unit(data["option_vectors"])[in_scope]
    service = service_for(args)
    predicted_local = choice_predictions(service, V, options, unit(data["instruction"]), queries)
    predicted = np.array(in_scope)[predicted_local]
    correct = predicted == truth
    rng = np.random.default_rng(SEED)
    venn = {"accuracy": float(correct.mean()), "ci95": wilson(int(correct.sum()), len(correct)), "ci95_row_bootstrap": interval(row_bootstrap(correct, rng)),
            "what": "Choice over the 150 in-scope intent names, no abstention option"}
    key = os.environ.get(args.jev_key_env, "")
    jev = {"status": "not run: no API key"}
    if key:
        import urllib.request
        from concurrent.futures import ThreadPoolExecutor

        cache_path = Path(args.output) / "jev-clinc-responses.jsonl"
        cached = {}
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                record = json.loads(line)
                cached[record["request_sha256"]] = record
        criteria = {option_names[i]: option_names[i] for i in in_scope}
        name_of = {option_names[i]: i for i in in_scope}

        def ask(i):
            body = {"model": JEV_MODEL, "state": test[i]["text"],
                    "questions": {"topic": {"type": "choice", "instructions": "Which support topic handles this request? Choose the single best match.", "criteria": criteria}}}
            digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
            if digest in cached:
                return cached[digest]
            request = urllib.request.Request(args.jev_endpoint, data=json.dumps(body).encode(), method="POST",
                                             headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read())
            answer = payload["answers"]["topic"]
            return {"request_sha256": digest, "row": i, "choice": answer.get("choice"), "confidence": answer.get("confidence"),
                    "model": payload.get("model"), "usage": payload.get("usage"), "latency_ms": (time.perf_counter() - started) * 1000}

        records = []
        with ThreadPoolExecutor(max_workers=4) as pool, cache_path.open("a") as stream:
            for record in pool.map(ask, range(len(test))):
                if record["request_sha256"] not in cached:
                    stream.write(json.dumps(record) + "\n")
                    cached[record["request_sha256"]] = record
                records.append(record)
        jev_predicted = np.array([name_of.get(r["choice"], -1) for r in records])
        jev_correct = jev_predicted == truth
        jev = {"status": "measured", "accuracy": float(jev_correct.mean()), "ci95": wilson(int(jev_correct.sum()), len(jev_correct)),
               "model": records[0].get("model"), "requests": len(records), "cost_usd": sum((r.get("usage") or {}).get("cost", 0) or 0 for r in records),
               "unparsed_choices": int((jev_predicted < 0).sum()), "latency_ms_p50": float(np.median([r["latency_ms"] for r in records if "latency_ms" in r]))}
    passed = jev.get("status") == "measured" and venn["ci95"][0] > jev["ci95"][1]
    return {"item": "9f-j", "dataset": CLINC_SOURCE, "rows": int(len(truth)), "intents": len(in_scope), "venn_zero_shot": venn, "jev": jev,
            "gate": {"passed": bool(passed), "rule": "Venn zero-shot interval lower bound above Jev's interval upper bound"}}


# ----------------------------------------------------------------------------- k. CLINC150 history curve
def item_k(V, args):
    test = np.load(args.clinc_vectors, allow_pickle=False)
    train = np.load(args.clinc_train_vectors, allow_pickle=False)
    row_identity = "not checked against source; use --check-texts for pinned row verification"
    if args.check_texts:
        from venn.benchmark import fetch_source

        rows, _ = fetch_source(CLINC_SOURCE)
        validate_clinc_rows(test, rows, "test")
        validate_clinc_rows(train, rows, "train")
        row_identity = "ordered text hashes, labels and original row IDs verified for both splits"
    queries, truth = unit(test["test_vectors"]), test["test_labels"].astype(np.int64)
    references, labels = unit(train["train_vectors"]), train["train_labels"].astype(np.int64)
    jev_path = Path(args.output) / "j.json"
    jev = json.loads(jev_path.read_text())["jev"] if jev_path.exists() else {}
    jev_ci = jev.get("ci95")
    rng = np.random.default_rng(SEED + 11)
    stops = {}
    for stop in STOPS:
        draws = 1 if stop == "all" else args.draws
        per_draw, pooled = [], []
        for draw in range(draws):
            keep = draw_store(np.random.default_rng(SEED + 7000 + 1000 * draw + 1), labels, stop)
            predicted, _ = predict(queries, references[keep], labels[keep])
            correct = predicted == truth
            per_draw.append(float(correct.mean()))
            pooled.append(row_bootstrap(correct, rng))
        pooled = np.concatenate(pooled)
        stops[str(stop)] = {"accuracy": float(np.mean(per_draw)), "per_draw": per_draw, "ci95_pooled": interval(pooled), "draw_spread": [min(per_draw), max(per_draw)],
                            "clears_jev": bool(jev_ci and interval(pooled)[0] > jev_ci[1])}
    vote = vote_predict(queries, references, labels)
    vote_correct = vote == truth
    crossover = next((s for s in STOPS if stops[str(s)]["clears_jev"]), None)
    return {"item": "9f-k", "status": "exploratory, added after seeing 9f-j", "dataset": "clinc150 plus, in-scope", "queries": int(len(truth)), "store_rows": int(len(labels)),
            "row_identity_verification": row_identity,
            "stops": stops, "crossover_stop": crossover, "jev": jev,
            "k_vote_all": {"accuracy": float(vote_correct.mean()), "ci95": wilson(int(vote_correct.sum()), len(vote_correct))},
            "note": "CLINC has 100 training rows per intent, so every stop is fully sampled"}


# ----------------------------------------------------------------------------- driver
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vectors", required=True, help="banking77-vectors.npz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--items", default="abcdefg", help="Item letters to run, for example abcfg")
    parser.add_argument("--draws", type=int, default=25)
    parser.add_argument("--model-dir", default=os.environ.get("VENN_MODEL_DIR"), help="venn-1 head bundle for b, d, g and the API cross-check")
    parser.add_argument("--check-texts", action="store_true", help="Re-read pinned Banking77 rows and, for k, both CLINC splits; verify vector row identities")
    parser.add_argument("--jev-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--jev-endpoint", default="https://openrouter.ai/api/alpha/decisions")
    parser.add_argument("--skip-http", action="store_true", help="Skip the loopback HTTP timing in g")
    parser.add_argument("--gate-vectors", default=None, help="banking77-gate-vectors.npz (descriptions and codes) for item h")
    parser.add_argument("--clinc-vectors", default=None, help="clinc-vectors.npz for items j and k")
    parser.add_argument("--clinc-train-vectors", default=None, help="clinc-train-vectors.npz for item k")
    args = parser.parse_args()
    unknown = sorted(set(args.items) - set(ITEMS))
    if unknown or not args.items:
        parser.error(f"--items takes letters from {ITEMS} with no separators, for example abcfg; got {args.items!r}")
    V = load(args.vectors)
    texts = None
    if args.check_texts or "e" in args.items:
        texts = check_texts(V, BANKING_SOURCE)
        write(Path(args.output) / "texts-verified.json", {"status": "verified", "rows": {k: len(v) for k, v in texts.items()}})
        log("texts verified against the pinned Banking77 revision")
    generic = GENERIC_TEMPERATURE_FALLBACK
    if args.model_dir:
        generic = service_for(args).generic
    args.gate_vectors = args.gate_vectors or str(Path(args.vectors).with_name("banking77-gate-vectors.npz"))
    args.clinc_vectors = args.clinc_vectors or str(Path(args.vectors).with_name("clinc-vectors.npz"))
    args.clinc_train_vectors = args.clinc_train_vectors or str(Path(args.vectors).with_name("clinc-train-vectors.npz"))
    implementation = implementation_provenance()
    overlap = overlap_diagnostic(V)
    runners = {"a": lambda: item_a(V, args), "b": lambda: item_b(V, args), "c": lambda: item_c(V, args), "d": lambda: item_d(V, args),
               "e": lambda: item_e(V, args, texts), "f": lambda: item_f(V, args, generic), "g": lambda: item_g(V, args),
               "h": lambda: item_h(V, args), "i": lambda: item_i(V, args), "j": lambda: item_j(V, args), "k": lambda: item_k(V, args)}
    # e first: a, c and h read Jev's matched run from e.json when it exists; j before k as listed.
    for item in sorted(set(args.items), key=("e" + ITEMS.replace("e", "")).index):
        provenance = input_provenance(item, args)
        started = time.perf_counter()
        log(f"running 9f-{item}")
        result = runners[item]()
        result["seconds"] = time.perf_counter() - started
        result["vectors_sha256"] = provenance["files_sha256"][str(Path(args.vectors))]
        result["input_provenance"] = provenance
        result["implementation"] = implementation
        result["banking77_overlap"] = overlap
        write(Path(args.output) / f"{item}.json", result)
        log(f"9f-{item} done in {result['seconds']:.0f} s")
    merged = {item: json.loads((Path(args.output) / f"{item}.json").read_text()) for item in ITEMS if (Path(args.output) / f"{item}.json").exists()}
    write(Path(args.output) / "results.json", merged)


if __name__ == "__main__":
    main()
