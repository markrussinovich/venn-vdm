"""Route a support ticket by text through a running local server (reader and head downloaded)."""
import argparse
import json
from urllib.error import URLError

from venn.client import Client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--text", default="I cannot sign in")
    args = parser.parse_args()
    try:
        result = Client(args.endpoint).decide(args.text, {
        "team": {"type": "choice", "instructions": "Who owns this ticket?",
                 "criteria": {"account": "Login or access", "billing": "Payments",
                              "no_match": None}}})
    except URLError as error:
        raise SystemExit(f"Cannot reach {args.endpoint}: {error.reason}. Start the server first with `venn serve`.")
    print(json.dumps({"model": result["model"], "answers": result["answers"]}, indent=2))


if __name__ == "__main__":
    main()
