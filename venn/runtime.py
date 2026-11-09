"""Locate and verify the model bundle and choose the writable state directory."""

import hashlib
import json
import os
from pathlib import Path

from safetensors.torch import load_file


def model_directory(path=None):
    """Return the head directory from ``path``, ``VENN_MODEL_DIR`` or the packaged assets, after checking its manifest.

    Every file named in ``manifest.json`` must match its SHA-256. Raises ``FileNotFoundError`` when no
    manifest or listed file exists and ``ValueError`` when a file is misplaced or its digest differs.
    """
    located = path or os.environ.get("VENN_MODEL_DIR")
    if not located:
        raise FileNotFoundError("No head directory given. Run 'venn download-model' and pass --model-dir or set VENN_MODEL_DIR; "
                                "for a model-free CPU example, run 'venn quickstart'.")
    directory = Path(located)
    if not (directory / "manifest.json").is_file():
        raise FileNotFoundError(
            f"No head found at {directory}. Run 'venn download-model', or set VENN_MODEL_DIR or --model-dir "
            "to supplied assets. For a model-free CPU example, run 'venn quickstart'."
        )
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        candidate = directory / name
        if candidate.parent != directory or hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Invalid Venn model artifact: {name}")
    return directory


def load_weights(path):
    """Load a safetensors file into a CPU state dict."""
    return load_file(str(path), device="cpu")


def state_directory():
    """Return the state directory: ``VENN_HOME`` if set, else ``$XDG_CACHE_HOME/venn`` (default ``~/.cache/venn``)."""
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return Path(os.environ.get("VENN_HOME", cache / "venn"))

