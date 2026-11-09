"""Local food-photo demo: train once, evaluate separately, then judge photos."""
import argparse
import base64
import hashlib
import json
import threading
import time
from pathlib import Path

from venn import DecisionService

LABELS = ("hotdog", "not_hotdog")
INSTRUCTIONS = "Is this a hot dog: a sausage served in a split bun?"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def photos(folder):
    return sorted(path for path in folder.rglob("*")
                  if path.is_file() and path.suffix.lower() in EXTENSIONS)


class Demo:
    def __init__(self, data, state, model_dir, device):
        self.data = data.resolve()
        self.state = state
        state.mkdir(parents=True, exist_ok=True)
        self.service = DecisionService(model_dir=model_dir, cache_path=state / "cache.sqlite",
                                       registry_path=state / "registry.json")
        self.device = device
        self.service.media_model = None
        self.lock = threading.Lock()
        self.job = {"busy": False, "phase": "Ready", "result": None, "error": None}
        self.report_path = state / "training.json"
        self.training = json.loads(self.report_path.read_text()) if self.report_path.exists() else None
        paths = photos(self.data)
        self.images = {str(i): path for i, path in enumerate(paths)}

    def inventory(self):
        return [{"id": key, "name": path.name,
                 "group": path.relative_to(self.data).parent.as_posix()}
                for key, path in self.images.items()]

    def labeled(self, split):
        return [(path, label) for label in LABELS
                for path in photos(self.data / split / label)]

    def reader(self):
        if self.service.media_model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            from venn.reader import MODEL_ID, REVISION

            started = time.perf_counter()
            self.service.media_model = SentenceTransformer(
                MODEL_ID, revision=REVISION, device=self.device,
                model_kwargs={"dtype": torch.float32}, local_files_only=True)
            return time.perf_counter() - started
        return 0.0

    def vector(self, path):
        started = time.perf_counter()
        before = self.service.embed_count
        vector = self.service.vector({"image": base64.b64encode(path.read_bytes()).decode("ascii")})
        return vector, time.perf_counter() - started, self.service.embed_count == before

    def train(self):
        rows = self.labeled("train")
        if any(sum(label == item for _, label in rows) < 5 for item in LABELS):
            raise ValueError("Add at least five distinct training photos to each train label folder.")
        hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path, _ in rows]
        if len(set(hashes)) != len(hashes):
            raise ValueError("Training contains duplicate files; use distinct photos.")
        load_seconds = self.reader()
        examples, embedding_seconds, hits = [], 0.0, 0
        for i, (path, label) in enumerate(rows):
            self.job["phase"] = f"Reading training photos: {i + 1}/{len(rows)}"
            vector, seconds, cached = self.vector(path)
            embedding_seconds += seconds
            hits += cached
            examples.append({"state": vector.tolist(), "label": label})
        self.job["phase"] = "Fitting Venn family (cross-validation on training photos only)"
        started = time.perf_counter()
        report = self.service.register_family(
            "hotdog", examples, form="linear", instructions=INSTRUCTIONS,
            options=[{"label": label, "vector": label} for label in LABELS])
        result = {"photos": len(rows), "reader_load_seconds": load_seconds,
                  "embedding_seconds": embedding_seconds, "cache_hits": hits,
                  "fit_seconds": time.perf_counter() - started,
                  "cv_accuracy": report["cv_accuracy"], "device": self.device,
                  "training_hashes": hashes}
        self.report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        self.training = result
        return {key: value for key, value in result.items() if key != "training_hashes"}

    def judge(self, path):
        if "hotdog" not in self.service.families or self.training is None:
            raise ValueError("Train a family first.")
        load_seconds = self.reader()
        vector, embedding_seconds, cached = self.vector(path)
        entry = self.service.families["hotdog"]
        question = {"hotdog": {
            "type": "Choice", "family": "hotdog", "instructions": INSTRUCTIONS,
            "options": [{"label": label, "vector": value}
                        for label, value in zip(entry["labels"], entry["options"])]}}
        started = time.perf_counter()
        answer = self.service.decide({"state": {"vector": vector.tolist()},
                                     "questions": question})["answers"]["hotdog"]
        return {"name": path.name, "answer": answer["answer"],
                "probabilities": answer["probabilities"],
                "calibrated": answer["calibrated"], "reader_load_seconds": load_seconds,
                "embedding_ms": embedding_seconds * 1000,
                "decision_ms": (time.perf_counter() - started) * 1000, "cached": cached}

    def compare_descriptions(self, path):
        """Send a training-free image Choice request through Venn's Decisions API."""
        import base64
        from venn.decisions import decide

        prompt = {
            "model": "venn-1",
            "state": {"image": base64.b64encode(path.read_bytes()).decode("ascii")},
            "questions": {"food": {
                "type": "choice",
                "instructions": "Which food is pictured?",
                "criteria": {
                    "hotdog": "A hot dog: a sausage in a long split bun.",
                    "hamburger": "A hamburger: a ground beef patty in a round bun.",
                    "none": None,
                },
            }},
        }
        load_seconds = self.reader()  # load on --device here, before the service would load it on CPU
        started = time.perf_counter()
        response = decide(self.service, prompt, experimental_media=True)
        elapsed = (time.perf_counter() - started) * 1000
        answer = response["answers"]["food"]
        response.pop("venn", None)  # Do not dump the 768-value state vector into the UI.
        return {
            "method": "venn_decisions_api",
            "name": path.name,
            "answer": answer["choice"],
            "probabilities": answer["probabilities"],
            "calibrated": answer["venn"]["calibrated"],
            "prompt": {**prompt, "state": {"image": "<base64 image omitted>"}},
            "response": response,
            "elapsed_ms": elapsed,
            "reader_load_seconds": load_seconds,
            "cached": False,
        }

    def evaluate(self):
        if self.training is None:
            raise ValueError("Train a family first.")
        rows = self.labeled("test")
        if not rows or any(not photos(self.data / "test" / label) for label in LABELS):
            raise ValueError("Add unseen photos to both test label folders.")
        train_hashes = set(self.training["training_hashes"])
        hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path, _ in rows]
        if len(set(hashes)) != len(hashes) or train_hashes.intersection(hashes):
            raise ValueError("Test photos must be distinct and not used in training.")
        results = []
        for i, (path, label) in enumerate(rows):
            self.job["phase"] = f"Evaluating unseen photos: {i + 1}/{len(rows)}"
            results.append({**self.judge(path), "expected": label})
        correct = sum(row["answer"] == row["expected"] for row in results)
        return {"photos": len(rows), "correct": correct, "accuracy": correct / len(rows),
                "embedding_ms_total": sum(row["embedding_ms"] for row in results),
                "decision_ms_total": sum(row["decision_ms"] for row in results),
                "cache_hits": sum(row["cached"] for row in results), "results": results}

    def start(self, operation):
        if not self.lock.acquire(blocking=False):
            return False
        self.job = {"busy": True, "phase": "Starting", "result": None, "error": None}

        def run():
            try:
                self.job["result"] = operation()
                self.job["phase"] = "Done"
            except Exception as error:
                import logging
                logging.exception("Hot dog demo operation failed")
                self.job["error"] = str(error)
                self.job["phase"] = "Failed"
            finally:
                self.job["busy"] = False
                self.lock.release()

        threading.Thread(target=run, daemon=True).start()
        return True


def create_app(demo):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from venn.http_security import LocalHTTPGuard

    app = FastAPI()
    app.add_middleware(LocalHTTPGuard)

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).with_name("index.html"))

    @app.get("/state")
    def state():
        training = {key: value for key, value in (demo.training or {}).items()
                    if key != "training_hashes"}
        return {"images": demo.inventory(), "trained": "hotdog" in demo.service.families,
                "training": training, "device": demo.device, "job": demo.job}

    @app.get("/image/{image_id}")
    def image(image_id: str):
        if image_id not in demo.images:
            raise HTTPException(404, "Unknown photo")
        return FileResponse(demo.images[image_id])

    @app.post("/action/{action}")
    def action(action: str, value: dict):
        if action == "train":
            operation = demo.train
        elif action == "evaluate":
            operation = demo.evaluate
        elif action in {"judge", "compare"} and value.get("id") in demo.images:
            path = demo.images[value["id"]]
            operation = lambda: demo.judge(path) if action == "judge" else demo.compare_descriptions(path)
        else:
            raise HTTPException(400, "Unknown action or photo")
        if not demo.start(operation):
            raise HTTPException(409, "Wait for the current operation to finish.")
        return {"started": True}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "photos")
    parser.add_argument("--state", type=Path, default=Path(__file__).parent / "local-state")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0; reader uses float32")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--action", choices=["train", "evaluate", "judge"], default="judge")
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    if args.model_dir is None:
        from venn.cli import download_head
        args.model_dir = download_head()
    demo = Demo(args.data, args.state, args.model_dir, args.device)
    if args.serve:
        import uvicorn
        uvicorn.run(create_app(demo), host="127.0.0.1", port=args.port)
    else:
        if args.action == "judge" and args.image is None:
            parser.error("--action judge requires --image")
        result = {"train": demo.train, "evaluate": demo.evaluate,
                  "judge": lambda: demo.judge(args.image)}[args.action]()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
