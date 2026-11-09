"""Require successful push CI for the exact release revision."""
import argparse
import json


def check(runs, revision):
    latest = runs.get("workflow_runs", [])
    if not latest:
        raise ValueError(f"No push CI run found for release revision {revision}")
    run = latest[0]
    if run.get("head_sha") != revision or run.get("event") != "push":
        raise ValueError("CI run does not identify the release revision and push event")
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        raise ValueError(
            f"Release CI has not passed: {run.get('status')}/{run.get('conclusion')}")
    return run["id"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_json")
    parser.add_argument("revision")
    args = parser.parse_args()
    with open(args.runs_json, encoding="utf-8") as source:
        try:
            run_id = check(json.load(source), args.revision)
        except ValueError as error:
            parser.exit(1, f"{error}\n")
    print(f"Release revision {args.revision} passed CI run {run_id}")
