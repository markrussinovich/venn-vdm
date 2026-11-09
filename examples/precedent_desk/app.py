"""Precedent Desk: a support desk whose closed tickets are the model.

A small local web app over the public ``DecisionService``. The store is the Banking77 training split
(closed tickets with the queue each was resolved in); the stream is the test split. Every accuracy
number on screen is read from ``results.json`` written by ``eval.py``; the app never computes accuracy
itself. Routing, precedents, queue fit, write-back and the backlog sweep are live calls.

    python examples/precedent_desk/app.py --vectors banking77-vectors.npz --results /tmp/precedent-desk/results.json \\
        --model-dir "$(venn download-model)" --prepare   # first run: fetch the ticket texts
    python examples/precedent_desk/app.py --vectors banking77-vectors.npz --results /tmp/precedent-desk/results.json

Then open http://127.0.0.1:8765. Typing a ticket of your own needs the pinned reader in the local
Hugging Face cache (``venn download-embedder``); everything else runs from the published vectors.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from venn.http_security import LocalHTTPGuard

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
harness = __import__("eval")  # the sibling eval.py: shared seeds, store draws and the Jev request shape

SEED = harness.SEED
STOPS = harness.STOPS
FINE_TUNED = 0.938
QUEUE_INSTRUCTION = harness.QUEUE_INSTRUCTION


class Desk:
    def __init__(self, args):
        from venn.server import DecisionService

        self.args = args
        self.V = harness.load(args.vectors)
        self.names = self.V["names"]
        self.results = json.loads(Path(args.results).read_text()) if Path(args.results).exists() else {}
        self.notes = json.loads((HERE / "resolution_notes.json").read_text())
        tickets_path = Path(args.tickets)
        if args.prepare or not tickets_path.exists():
            texts = harness.check_texts(self.V, {"repository": "PolyAI/banking77", "revision": "5724874eb34f7571b611868e14cba91202b9b2a9", "config": "default"})
            tickets_path.parent.mkdir(parents=True, exist_ok=True)
            tickets_path.write_text(json.dumps(texts))
        self.texts = json.loads(tickets_path.read_text())
        home = Path(args.state)
        os.environ["VENN_HOME"] = str(home)
        self.service = DecisionService(model_dir=args.model_dir, cache_path=home / "cache.sqlite", registry_path=home / "registry" / "calibrations.json")
        self.generic = self.service.generic
        # Held-out queues: the first held-out choice of eval item f, so the live beat matches the measured one.
        self.held_out = sorted(np.random.default_rng(SEED + 4000).choice(len(self.names), harness.HELD_OUT, replace=False).tolist())
        self.stop = "all"
        self.history = True  # False: route from the queue labels alone (zero-shot), the store is not consulted
        gate_path = Path(args.gate_vectors or Path(args.vectors).with_name("banking77-gate-vectors.npz"))
        self.description_vectors = harness.unit(np.load(gate_path, allow_pickle=False)["description_vectors"]) if gate_path.exists() else None
        # Which zero-shot option set: fresh "descriptions" (the fair baseline, needs the gate vectors) or the original "names".
        self.label_set = "descriptions" if self.description_vectors is not None else "names"
        self.descriptions = json.loads((HERE / "queue_descriptions.json").read_text())
        self.written = []  # (split, row) tuples resolved during the session, in order
        self.indexes = {}
        self.jev = self.load_jev(args.jev_cache)
        self.jev_key = os.environ.get(args.jev_key_env, "")
        self.backlog = None
        self.embeddings = 0
        self.rebuild()

    # ----------------------------------------------------------------- store
    def base_rows(self):
        keep = harness.draw_store(np.random.default_rng(SEED + 1), self.V["train_y"], self.stop)
        return keep[~np.isin(self.V["train_y"][keep], self.held_out)]

    def rebuild(self):
        rows = self.base_rows()
        vectors = [self.V["train"][rows]] + [self.V[split][[i]] for split, i in self.written]
        labels = [self.names[y] for y in self.V["train_y"][rows]] + [self.names[self.V[f"{split}_y"][i]] for split, i in self.written]
        ids = [f"train:{i}" for i in rows] + [f"{split}:{i}" for split, i in self.written]
        self.store_vectors, self.store_labels, self.store_ids = np.concatenate(vectors), labels, ids
        self.handle = self.service.register_index(self.store_vectors, ids, outcomes=labels)
        counts = {name: 0 for name in self.names}
        for label in labels:
            counts[label] += 1
        self.counts = counts
        _, margins = harness.predict(self.V["test"][~np.isin(self.V["test_y"], self.held_out)], self.store_vectors, np.array([self.names.index(l) for l in labels]))
        self.tau = float(np.percentile(margins, 10))

    def state(self):
        a = self.results.get("a", {})
        return {"stop": self.stop, "history": self.history, "label_set": self.label_set, "has_descriptions": self.description_vectors is not None, "store_rows": len(self.store_ids), "written": len(self.written), "counts": self.counts,
                "held_out": [self.names[i] for i in self.held_out], "tau_margin": self.tau,
                "results": self.summary(), "jev_available": bool(self.jev), "embeddings": self.embeddings,
                "stops": [str(s) for s in STOPS], "fine_tuned": FINE_TUNED,
                "history_rows_at_stop": {str(s): v.get("store_rows_mean") for s, v in a.get("stops", {}).items()}}

    def summary(self):
        r = self.results
        out = {}
        if "a" in r:
            out["history"] = {s: {"accuracy": v["accuracy"], "ci95": v["ci95_pooled"]} for s, v in r["a"]["stops"].items()}
            out["crossover_stop"] = r["a"]["crossover_stop"]
            out["jev"] = r["a"]["jev"]
        if "b" in r and "without_abstention_option" in r["b"]:
            out["zero_shot"] = r["b"]["without_abstention_option"]
        if "c" in r:
            out["noise"] = {k: {"accuracy": v["accuracy"], "ci95": v["ci95_pooled"]} for k, v in r["c"]["levels"].items()}
        if "f" in r:
            out["experience"] = {"counts": r["f"]["summary"], "none_rate": r["f"]["first_arrival_none_rates"], "gate": r["f"]["gate"]}
        if "g" in r and "thresholds" in r["g"]:
            out["sweep"] = {"thresholds": r["g"]["thresholds"], "timing": r["g"]["timing"], "rows": r["g"]["rows"]}
        if "d" in r and "cells" in r["d"]:
            out["registration"] = {k: {key: v[key] for key in ("test_accuracy", "test_accuracy_ci95_over_draws", "nearest_reference_same_store", "test_ece", "calibrated_share", "tail_p95", "registration_seconds", "claim_allowed")}
                                   for k, v in r["d"]["cells"].items()}
        if "e" in r:
            out["jev_run"] = {k: r["e"].get(k) for k in ("status", "accuracy", "ci95", "model", "requests", "cost_usd")}
        if "h" in r and "zero_shot" in r["h"]:
            out["label_strings"] = {"zero_shot": r["h"]["zero_shot"], "gate": r["h"]["gate"],
                                    "held_out_count_0": r["h"]["held_out"]["0"], "held_out_count_1": r["h"]["held_out"]["1"]}
        if "i" in r:
            out["k_vote_noise"] = {k: {"k_vote": v["k_vote_accuracy"], "ci95": v["ci95_pooled"], "best_single": v["best_single_reference_accuracy"]} for k, v in r["i"]["levels"].items()}
        if "j" in r and "venn_zero_shot" in r["j"]:
            out["clinc"] = {"venn_zero_shot": r["j"]["venn_zero_shot"], "jev": r["j"]["jev"], "gate": r["j"]["gate"], "rows": r["j"]["rows"]}
        if "k" in r and "stops" in r["k"]:
            out["clinc_history"] = {"stops": {k: {"accuracy": v["accuracy"], "ci95": v["ci95_pooled"]} for k, v in r["k"]["stops"].items()}, "crossover_stop": r["k"].get("crossover_stop"), "k_vote_all": r["k"].get("k_vote_all")}
        return out

    def load_jev(self, path):
        if not path or not Path(path).exists():
            return {}
        cached = {}
        for line in Path(path).read_text().splitlines():
            record = json.loads(line)
            cached[record["row"]] = record
        return cached

    # ----------------------------------------------------------------- tickets
    def ticket(self, split, row):
        item = self.texts[split][row]
        return {"split": split, "row": row, "id": f"{split}:{row}", "text": item["text"], "label": self.names[item["label"]],
                "held_out": item["label"] in self.held_out}

    def pick(self, kind, rng):
        y = self.V["test_y"]
        if kind == "held_out":
            rows = np.flatnonzero(np.isin(y, self.held_out))
        elif kind == "duplicate":
            rows = np.flatnonzero(y == self.names.index("transaction_charged_twice"))
        else:
            rows = np.flatnonzero(~np.isin(y, self.held_out))
        return self.ticket("test", int(rng.choice(rows)))

    # ----------------------------------------------------------------- decisions
    def queue_fit(self, vector, queue):
        """The public SetMembership question against one queue's stored tickets."""
        members = self.store_vectors[np.array(self.store_labels) == queue]
        if len(members) < 2:
            return None
        fit = self.service.decide({"state": {"vector": vector.tolist()},
                                   "questions": {"belongs": {"type": "SetMembership", "members": members.tolist()}}})["answers"]["belongs"]
        return {"queue": queue, "answer": bool(fit["answer"]), "distance": fit["distance"], "reference_cutoff": fit["reference_cutoff"], "members": int(len(members))}

    def route_zero_shot(self, vector):
        """Queue labels only: Choice over the 77 option vectors (fresh descriptions or the original names), no store, no precedents."""
        vectors = self.description_vectors if (self.label_set == "descriptions" and self.description_vectors is not None) else self.V["options"]
        options = [{"label": name, "vector": v.tolist()} for name, v in zip(self.names, vectors)]
        started = time.perf_counter()
        answer = self.service.decide({"state": {"vector": vector.tolist()},
                                      "questions": {"queue": {"type": "Choice", "instructions_vec": self.V["queue_instruction"].tolist(), "options": options}}})["answers"]["queue"]
        latency = (time.perf_counter() - started) * 1000
        probabilities = {k: v for k, v in answer["probabilities"].items() if k != "none"}
        queue = max(probabilities, key=probabilities.get)
        label_set = "fresh descriptions" if (self.label_set == "descriptions" and self.description_vectors is not None) else "original names"
        return {"queue": queue, "confidence": probabilities[queue], "margin": None, "confident": True, "abstain": False, "precedents": [], "belongs": None,
                "latency_ms": latency, "store_rows": 0, "embeddings_this_call": 0, "mode": "queue names only", "label_set": label_set,
                "option_text": self.descriptions.get(queue, "") if label_set == "fresh descriptions" else queue.replace("_", " "),
                "request": {"state": {"vector": "<768 floats>"}, "questions": {"queue": {"type": "choice", "instructions": QUEUE_INSTRUCTION, "criteria": f"the 77 queue {label_set}"}}}}

    def route(self, vector, row=None, k=3):
        if not self.history:
            return self.route_zero_shot(vector)
        started = time.perf_counter()
        answer = self.service.decide({"state": {"vector": vector.tolist()},
                                      "questions": {"precedent": {"type": "NearestReference", "index_ref": self.handle, "k": k}}})["answers"]["precedent"]
        latency = (time.perf_counter() - started) * 1000
        label = answer["answer"]
        precedents = []
        for ref in answer["evidence_ids"]:
            split, i = ref.split(":")
            item = self.texts[split][int(i)]
            precedents.append({"id": ref, "text": item["text"], "queue": self.names[item["label"]], "note": self.notes.get(self.names[item["label"]], "")})
        belongs = self.queue_fit(vector, label)
        confident = answer["margin"] >= self.tau
        return {"queue": label, "confidence": answer["confidence"], "margin": answer["margin"], "confident": confident,
                "abstain": not confident, "precedents": precedents, "belongs": belongs, "latency_ms": latency,
                "store_rows": len(self.store_ids), "embeddings_this_call": 0, "mode": "closed tickets",
                "request": {"state": {"vector": "<768 floats>"}, "questions": {"precedent": {"type": "nearest_reference", "store": self.handle[:12] + "…", "k": k},
                                                                             "belongs": {"type": "set_membership", "set": f"queue:{label}"}}}}

    def jev_answer(self, row, text=None):
        """Jev's routing of one ticket: the cached matched-run response, or a live call when a key is configured."""
        record = self.jev.get(row) if row is not None else None
        if record is None and text and self.jev_key:
            record = self.jev_live(text)
        if record is None:
            return {"available": False, "note": "No cached Jev response for this ticket and no API key configured; no answer is invented."}
        return {"available": True, "choice": record.get("choice"), "confidence": record.get("confidence"), "latency_ms": record.get("latency_ms"),
                "model": record.get("model"), "live": record.get("live", False)}

    def jev_live(self, text):
        import urllib.request

        criteria = {name: name for name in self.V["option_names"]}
        body = {"model": harness.JEV_MODEL, "state": text, "questions": {"queue": {"type": "choice", "instructions": harness.JEV_INSTRUCTION, "criteria": criteria}}}
        request = urllib.request.Request(self.args.jev_endpoint, data=json.dumps(body).encode(), method="POST",
                                         headers={"Authorization": f"Bearer {self.jev_key}", "Content-Type": "application/json"})
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read())
        except Exception as error:  # a failed call is shown as unavailable, never guessed
            return {"choice": None, "confidence": None, "model": harness.JEV_MODEL, "live": True, "error": str(error)[:200]}
        answer = payload["answers"]["queue"]
        return {"choice": answer.get("choice"), "confidence": answer.get("confidence"), "model": payload.get("model"), "live": True,
                "latency_ms": (time.perf_counter() - started) * 1000}

    def resolve(self, split, row):
        key = (split, int(row))
        if key not in self.written:
            self.written.append(key)
            self.rebuild()
        queue = self.names[self.V[f"{split}_y"][int(row)]]
        written_for_queue = sum(1 for s, i in self.written if self.names[self.V[f"{s}_y"][i]] == queue)
        measured = self.results.get("f", {}).get("summary", {})
        return {"store_rows": len(self.store_ids), "queue": queue, "written_for_queue": written_for_queue,
                "measured_curve": measured, "note": "the curve is eval.py's held-out measurement over 70 queue draws; the counter never scores the ticket that was written"}

    def sweep(self, threshold, mode="registered"):
        """One IndexApply pass over all 13,083 tickets: the pre-registered two-option FX question, or the 77-queue re-tag."""
        if self.backlog is None:
            vectors = np.concatenate([self.V["train"], self.V["test"]])
            ids = [f"train:{i}" for i in range(len(self.V["train"]))] + [f"test:{i}" for i in range(len(self.V["test"]))]
            self.backlog = (self.service.register_index(vectors, ids), np.concatenate([self.V["train_y"], self.V["test_y"]]))
        handle, labels = self.backlog
        if mode == "registered":
            options = [{"label": l, "vector": v.tolist()} for l, v in zip(self.V["fx_labels"], self.V["fx_options"])]
            question = {"type": "IndexApply", "instructions_vec": self.V["fx_instruction"].tolist(), "options": options, "threshold": threshold, "batch_size": 65536}
            is_hit = lambda row: row["answer"] == "affected"  # noqa: E731
        else:
            options = [{"label": name, "vector": vector.tolist()} for name, vector in zip(self.names, self.V["options"])]
            question = {"type": "IndexApply", "instructions_vec": self.V["queue_instruction"].tolist(), "options": options, "batch_size": 65536}
            is_hit = lambda row: row["answer"] in harness.FX_TRUTH  # noqa: E731
        started = time.perf_counter()
        rows = [row for line in self.service.stream_index(handle, question) for row in json.loads(line)["rows"]]
        seconds = time.perf_counter() - started
        truth = {self.names.index(n) for n in harness.FX_TRUTH}
        hits = [row for row in rows if is_hit(row)]
        review = [row for row in rows if row["answer"] == "none"]
        sample = []
        for row in sorted(hits, key=lambda r: -r["confidence"])[:40]:
            split, i = row["id"].split(":")
            sample.append({"id": row["id"], "text": self.texts[split][int(i)]["text"], "queue": self.names[self.texts[split][int(i)]["label"]], "confidence": row["confidence"]})
        hit_labels = [labels[self.index_of(row["id"])] for row in hits]
        true_hits = sum(1 for y in hit_labels if y in truth)
        g = self.results.get("g", {})
        measured = g.get("thresholds", {}).get(str(threshold)) if mode == "registered" else g.get("exploratory_77_queue_sweep", {}).get("thresholds", {}).get("None")
        return {"rows": len(rows), "hits": len(hits), "review": len(review), "true_hits": true_hits, "truth_rows": int(np.isin(labels, list(truth)).sum()),
                "seconds": seconds, "boundary": "in-process scoring, NDJSON serialization and consumption; the index was registered beforehand",
                "mode": mode, "label": "pre-registered two-option question" if mode == "registered" else "exploratory: re-tag with the 77 queue names, file the two FX queues",
                "sample": sample, "measured": measured}

    def index_of(self, ref):
        split, i = ref.split(":")
        return int(i) + (0 if split == "train" else len(self.V["train"]))

    def embed(self, text):
        self.embeddings += 1
        return self.service.embed([text])[0]

    def register(self, per_queue=35):
        rng = np.random.default_rng(SEED + 3000 + per_queue)
        rows = self.base_rows()
        labels = self.V["train_y"][rows]
        keep = np.concatenate([rng.choice(rows[labels == y], min(per_queue, int((labels == y).sum())), replace=False) for y in np.unique(labels)])
        examples = [{"state": self.V["train"][i].tolist(), "label": self.names[self.V["train_y"][i]]} for i in keep]
        options = [{"label": name, "vector": vector.tolist()} for name, vector in zip(self.names, self.V["options"]) if self.names.index(name) not in self.held_out]
        started = time.perf_counter()
        report = self.service.register_family("desk", examples, options, self.V["family_instruction"].tolist(), seed=0,
                                             split=harness.protocol_split(self.V["train_y"][keep], 0))
        return {"seconds": time.perf_counter() - started, "examples": len(keep), "report": {k: report[k] for k in report if k not in ("split_hashes",)},
                "measured": self.results.get("d", {}).get("cells", {}).get(str(per_queue))}


def create_app(desk):
    app = FastAPI()
    app.add_middleware(LocalHTTPGuard)
    rng = np.random.default_rng(SEED + 99)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return (HERE / "static" / "index.html").read_text()

    @app.get("/api/state")
    async def state():
        return desk.state()

    @app.get("/api/ticket")
    async def ticket(kind: str = "random", split: str = "test", row: int = -1):
        if row >= 0:
            return desk.ticket(split, row)
        return desk.pick(kind, rng)

    @app.post("/api/route")
    async def route(value: dict):
        if "text" in value and value["text"]:
            vector = desk.embed(value["text"])
            venn = desk.route(vector)
            venn["embeddings_this_call"] = 1
            return {"venn": venn, "jev": desk.jev_answer(None, value["text"]), "ticket": {"text": value["text"], "label": None, "typed": True}}
        split, row = value.get("split", "test"), int(value["row"])
        vector = desk.V[split][row]
        ticket = desk.ticket(split, row)
        return {"venn": desk.route(vector), "jev": desk.jev_answer(row if split == "test" else None, ticket["text"]), "ticket": ticket}

    @app.post("/api/resolve")
    async def resolve(value: dict):
        return desk.resolve(value.get("split", "test"), int(value["row"]))

    @app.post("/api/history")
    async def history(value: dict):
        if value.get("label_set") == "descriptions" and desk.description_vectors is None:
            return JSONResponse({"error": "Fresh queue descriptions need banking77-gate-vectors.npz (--gate-vectors)"}, status_code=409)
        desk.history = bool(value.get("on", True))
        if value.get("label_set") in ("descriptions", "names"):
            desk.label_set = value["label_set"]
        return desk.state()

    @app.post("/api/fit")
    async def fit(value: dict):
        vector = desk.V[value.get("split", "test")][int(value["row"])]
        return {"fit": desk.queue_fit(vector, value["queue"])}

    @app.post("/api/stop")
    async def stop(value: dict):
        desk.stop = value["stop"] if value["stop"] == "all" else int(value["stop"])
        desk.rebuild()
        return desk.state()

    @app.post("/api/sweep")
    async def sweep(value: dict):
        return desk.sweep(float(value.get("threshold", .9)), value.get("mode", "registered"))

    @app.post("/api/register")
    async def register(value: dict):
        return desk.register(int(value.get("per_queue", 35)))

    @app.get("/api/results")
    async def results():
        return JSONResponse(desk.results)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--results", required=True, help="results.json written by eval.py")
    parser.add_argument("--model-dir", default=os.environ.get("VENN_MODEL_DIR"))
    parser.add_argument("--tickets", default=None, help="Ticket texts fetched from the pinned Banking77 revision (default: tickets.json in --state)")
    parser.add_argument("--jev-cache", default=None, help="jev-responses.jsonl from eval.py item e, if the matched run was made")
    parser.add_argument("--jev-key-env", default="OPENROUTER_API_KEY", help="Environment variable with an OpenRouter key for live Jev calls on typed tickets")
    parser.add_argument("--jev-endpoint", default="https://openrouter.ai/api/alpha/decisions")
    parser.add_argument("--state", default=str(Path(os.environ.get("VENN_HOME", Path.home() / ".cache" / "venn")) / "precedent-desk"),
                        help="Embedding cache, registry and ticket texts (default: $VENN_HOME/precedent-desk)")
    parser.add_argument("--prepare", action="store_true", help="Fetch the ticket texts even if a copy exists")
    parser.add_argument("--gate-vectors", default=None, help="banking77-gate-vectors.npz with the fresh queue descriptions (default: beside --vectors)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    args.tickets = args.tickets or str(Path(args.state) / "tickets.json")
    desk = Desk(args)
    print(json.dumps({"store_rows": len(desk.store_ids), "held_out": [desk.names[i] for i in desk.held_out], "url": f"http://127.0.0.1:{args.port}"}))
    uvicorn.run(create_app(desk), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
