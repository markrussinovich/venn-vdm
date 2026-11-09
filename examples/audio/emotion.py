"""Fit a linear emotion family from labeled WAV clips, then classify a new clip.

Run from the checkout: python examples/audio/emotion.py --labels clips.csv --clip new.wav
The CSV has path,label,speaker columns; relative paths resolve beside the CSV.
"""
import argparse
import base64
import csv
import json
import os
from pathlib import Path

from venn import DecisionService


def audio_state(path):
    """Read one local WAV as the raw-audio state accepted by DecisionService."""
    return {"audio": base64.b64encode(path.read_bytes()).decode("ascii")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path,
                        help="Head assets; defaults to VENN_MODEL_DIR, else the published venn-1 head is downloaded")
    args = parser.parse_args()
    with args.labels.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows or any(not all(row.get(key) for key in ("path", "label", "speaker")) for row in rows):
        parser.error("The CSV needs nonempty path,label,speaker columns")
    paths = [(args.labels.parent / row["path"]).resolve() for row in rows]
    if args.clip.resolve() in paths:
        parser.error("Choose a new clip that is not in the fitting CSV")
    if args.model_dir is None:
        from venn.cli import download_head

        args.model_dir = os.environ.get("VENN_MODEL_DIR") or download_head()
    service = DecisionService(model_dir=args.model_dir)
    examples = [{"state": audio_state(path), "label": row["label"], "group_id": row["speaker"]}
                for path, row in zip(paths, rows)]
    labels = sorted({row["label"] for row in rows})
    instructions = "Which emotion is the speaker expressing?"
    report = service.register_family(
        "audio_emotion", examples, form="linear", instructions=instructions,
        options=[{"label": label, "vector": label} for label in labels],
    )
    options = service.families["audio_emotion"]["options"]
    result = service.decide({"state": audio_state(args.clip), "questions": {"emotion": {
        "type": "Choice", "family": "audio_emotion", "instructions": instructions,
        "options": [{"label": label, "vector": vector} for label, vector in zip(labels, options)],
    }}})
    print(json.dumps({"training_clips": len(rows), "cv_accuracy": report["cv_accuracy"],
                      "answer": result["answers"]["emotion"]}, indent=2))


if __name__ == "__main__":
    main()
