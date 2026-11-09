"""Predict a wine critic's score (80-100) from the tasting note with a Venn regression family.

Downloads the pinned winemag CSV from Hugging Face at run time (CC BY-SA 4.0; nothing is stored in this
repository), takes the same 2,000 notes and 1,200 / 800 split as TypeSafe's "Autoresearch feature
discovery" cookbook, embeds the notes with the venn-1 reader, registers a regression family on the 1,200
fitting notes and prints the held-out RMSE on the other 800.

    python examples/wine/wine.py                    # the API's default reader on CPU
    python examples/wine/wine.py --device cuda:0    # the same reader on a GPU, bf16
    python examples/wine/wine.py --form kernel_ridge  # SentenceSimilarity + length inputs
    python examples/wine/wine.py --form kernel_ridge --device cuda:0
"""
import argparse
import csv
import io
import os
import random
import time
import urllib.request
from pathlib import Path

import numpy as np

from venn import DecisionService

CSV = "https://huggingface.co/datasets/GroNLP/ik-nlp-22_winemag/resolve/90eb39f35fc64e556fc17f06d4137a4a69ec3297/train.csv"


def load(n_fit=1200, n_held_out=800, seed=0):
    """The cookbook's selection: drop empty rows and repeated notes, shuffle with seed 0, take the first 2,000."""
    request = urllib.request.Request(CSV, headers={"User-Agent": "venn-example/1.0"})
    text = urllib.request.urlopen(request, timeout=300).read().decode()
    rows, seen = [], set()
    for row in csv.DictReader(io.StringIO(text)):
        if row["description"] and row["points"] and row["description"] not in seen:
            seen.add(row["description"])
            rows.append((row["description"], float(row["points"])))
    random.Random(seed).shuffle(rows)
    rows = rows[: n_fit + n_held_out]
    return [r[0] for r in rows], np.array([r[1] for r in rows]), n_fit


def gpu_reader(device):
    """The pinned venn-1 reader on a GPU in bf16, for DecisionService(embed=...)."""
    import torch
    from sentence_transformers import SentenceTransformer

    from venn.reader import MODEL_ID, REVISION

    model = SentenceTransformer(MODEL_ID, revision=REVISION, device=device, model_kwargs={"dtype": torch.bfloat16},
                                config_kwargs={"vision_config": None, "audio_config": None})
    return lambda texts, prompt_name="Classification": model.encode(
        texts, prompt_name=prompt_name, batch_size=64, normalize_embeddings=False)


def fit_and_predict(service, notes, points, n_fit, form="ridge"):
    """Fit only the first n_fit notes and predict the rest, keeping the embedding prompt consistent."""
    timings = {}
    kernel = form == "kernel_ridge"
    prompt = "SentenceSimilarity" if kernel else "Classification"
    started = time.perf_counter()
    vectors = service.embed(notes, prompt=prompt)
    timings["embed"] = time.perf_counter() - started

    started = time.perf_counter()
    examples = [{"state": note if kernel else {"vector": vector.tolist()}, "points": float(score)}
                for note, vector, score in zip(notes[:n_fit], vectors[:n_fit], points[:n_fit])]
    settings = {"prompt": prompt, "inputs": ["log_words", "log_chars"]} if kernel else {}
    report = service.register_family("wine_points", examples, objective="regress", target="points", form=form,
                                    split={"train": list(range(n_fit))}, **settings)
    timings["register"] = time.perf_counter() - started

    started = time.perf_counter()
    question = {"points": {"type": "Score", "family": "wine_points"}}
    predicted = []
    for start in range(n_fit, len(notes), 512):
        state = {"texts": notes[start:start + 512]} if kernel else {"vectors": vectors[start:start + 512].tolist()}
        answer = service.decide({"state": state, "questions": question})
        predicted += [row["score"] for row in answer["answers"]["points"]["rows"]]
    timings["predict"] = time.perf_counter() - started
    return report, np.asarray(predicted), timings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", help="cpu (the API's default reader) or cuda:N")
    parser.add_argument("--form", choices=("ridge", "kernel_ridge"), default="ridge",
                        help="ridge baseline or kernel_ridge with SentenceSimilarity embeddings and note length")
    parser.add_argument("--model-dir", type=Path,
                        help="Head assets; defaults to VENN_MODEL_DIR, else the published venn-1 head is downloaded")
    args = parser.parse_args()
    if args.model_dir is None:
        from venn.cli import download_head

        args.model_dir = os.environ.get("VENN_MODEL_DIR") or download_head()
    run_started = time.perf_counter()
    timings = {}
    started = time.perf_counter()
    notes, points, n_fit = load()
    timings["download"] = time.perf_counter() - started

    venn = DecisionService(embed=None if args.device == "cpu" else gpu_reader(args.device), model_dir=args.model_dir)
    report, predicted, stages = fit_and_predict(venn, notes, points, n_fit, args.form)
    timings.update(stages)

    held_out = points[n_fit:]
    rmse = float(np.sqrt(np.mean((held_out - np.array(predicted)) ** 2)))
    print(f"{args.form}, alpha {report['alpha']}, held-out RMSE {rmse:.3f} on {len(held_out)} notes")
    print("seconds: " + ", ".join(f"{k} {v:.1f}" for k, v in timings.items())
          + f", total including model startup {time.perf_counter() - run_started:.1f}")
    if args.device == "cpu":
        print(f"CPU embedding threads: {venn.last_batch_threads}")


if __name__ == "__main__":
    main()
