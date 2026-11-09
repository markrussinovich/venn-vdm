"""A loopback test server with fake embeddings and an untrained head, for contract tests only."""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import uvicorn
from safetensors.torch import save_file

from venn import DecisionService, create_app
from venn.heads.abstention import CosineAbstention
from venn.profiles import VENN_1


def embed(texts):
    return np.array([
        np.random.default_rng(int.from_bytes(hashlib.sha256(text.encode()).digest()[:8]))
        .normal(size=768) for text in texts
    ], dtype=np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="venn-contract-") as directory:
        root = Path(directory)
        assets = root / "assets"
        assets.mkdir()
        (assets / "config.json").write_text(json.dumps({
            "reader": {"id": VENN_1.reader, "revision": VENN_1.revision, "dimensions": 768},
            "baseline": {"temperature": .1, "none_logit": 0}, "choice_temperature": 1,
        }))
        save_file(CosineAbstention(.1, 0).state_dict(), str(assets / "universal.safetensors"))
        (assets / "manifest.json").write_text(json.dumps({"sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in assets.iterdir()
        }}))
        service = DecisionService(embed, model_dir=assets, cache_path=root / "cache.sqlite",
                                  registry_path=root / "registry.json")
        uvicorn.run(create_app(service), host="127.0.0.1", port=args.port)
