"""Compare a VennSpace run with the rounded published reference intervals."""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compare(run, reference):
    if run.get("protocol_sha256") != reference["protocol_sha256"]:
        raise ValueError("Protocol checksum differs from the published reference")
    rows = []
    for metric in reference["metrics"]:
        value = run
        try:
            for key in metric["path"]:
                value = value[key]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Missing metric: {'.'.join(metric['path'])}") from error
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Metric must be a finite number: {metric['path']}")
        low, high = metric["interval"]
        rows.append({"metric": ".".join(metric["path"]), "observed": value,
                     "reference": metric["value"], "difference": value - metric["value"],
                     "reference_interval": [low, high], "within_interval": low <= value <= high})
    return {"criterion": reference["criterion"], "passed": all(row["within_interval"] for row in rows),
            "metrics": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="summary.json written by venn benchmark")
    parser.add_argument("--reference", type=Path, default=ROOT / "benchmarks/vennspace-v1-reference.json")
    args = parser.parse_args()
    try:
        result = compare(json.loads(args.run.read_text()), json.loads(args.reference.read_text()))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Cannot compare: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
