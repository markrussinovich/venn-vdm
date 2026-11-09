"""Synthetic HTTP index lifecycle; requires a local server, but no reader."""
import argparse
import json
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np


def run(endpoint):
    def request(method, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        with urlopen(Request(endpoint.rstrip("/") + path, data=data, method=method,
                             headers={"Content-Type": "application/json"}), timeout=60) as response:
            return json.load(response)

    # Synthetic coordinates test the transport. Never register them as real text embeddings.
    vectors = np.eye(3, 768, dtype=np.float32).tolist()
    handle = request("POST", "/indexes", {"vectors": {"vectors": vectors},
                     "ids": ["a", "b", "c"], "outcomes": ["account", "billing", "shipping"]})["index_ref"]
    try:
        nearest = request("POST", "/api/alpha/decisions", {
            "model": "venn-1", "state": {"vector": vectors[0]},
            "questions": {"team": {"type": "nearest_reference", "store": handle}}})
        question = {"type": "index_apply", "instructions_vec": vectors[0], "batch_size": 2,
                    "options": [{"label": label, "vector": vector}
                                for label, vector in zip(["account", "billing", "shipping"], vectors)]}
        consumed = []
        with urlopen(Request(endpoint.rstrip("/") + "/index-apply/" + handle,
                             data=json.dumps(question).encode(),
                             headers={"Content-Type": "application/json"}), timeout=60) as response:
            for line in response:
                consumed.extend(row["id"] for row in json.loads(line)["rows"])
        if consumed != ["a", "b", "c"]:
            raise RuntimeError("Streaming lost or reordered rows")
        return {"scope": "synthetic API lifecycle; no semantic claim", "consumed_ids": consumed,
                "nearest": nearest["answers"]["team"]}
    finally:
        request("DELETE", "/indexes/" + handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    endpoint = parser.parse_args().endpoint
    try:
        print(json.dumps(run(endpoint), indent=2))
    except URLError as error:
        raise SystemExit(f"Cannot reach {endpoint}: {error.reason}. Start a server first, for example "
                         "`python scripts/contract_server.py --port 8000`.")
