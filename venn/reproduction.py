"""Preflight and provenance for fresh semantic evaluations."""
import importlib.metadata
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path


def file_sha256(path):
    """Hash a local artifact without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_provenance(root=None):
    """Record available source identity; protocol identity is not code identity."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    paths = sorted((root / "venn").rglob("*.py"))
    paths += [root / "examples/precedent_desk/eval.py", root / "scripts/compare_reference.py"]
    result = {"source_revision": None, "source_dirty": None,
              "code_sha256": {path.relative_to(root).as_posix(): file_sha256(path)
                              for path in paths if path.is_file()}}
    try:
        git_root = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
        if Path(git_root).resolve() == root.resolve():
            result["source_revision"] = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
            result["source_dirty"] = bool(subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
                text=True, stderr=subprocess.DEVNULL, timeout=5).strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def preflight(gpus, output, cache):
    """Check CUDA, required packages and writable ``output``/``cache`` directories, then write ``environment.json``.

    Returns the provenance dict; raises ``ValueError`` when ``gpus`` is below 1 or exceeds the visible
    devices, or a required package is missing.
    """
    import torch
    if gpus < 1:
        raise ValueError("Fresh semantic runs require CUDA: --gpus must be at least 1. Use venn quickstart or venn sweep for CPU demos.")
    available = torch.cuda.device_count()
    if gpus > available:
        raise ValueError(f"Requested {gpus} CUDA devices, but {available} are visible. Set --gpus to a visible device count.")
    versions = {}
    for name in ("numpy", "torch", "transformers", "sentence-transformers", "datasets", "peft"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise ValueError(f"Missing {name}; install '.[embedding,benchmark]' in the active environment") from error
    for directory in (Path(output), Path(cache).parent):
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=directory) as probe:
            probe.write(b"write check")
    provenance = {"python": sys.version, "platform": platform.platform(), "packages": versions,
                  "cuda_runtime": torch.version.cuda,
                  "devices": [torch.cuda.get_device_name(i) for i in range(gpus)],
                  "command": sys.argv, "implementation": implementation_provenance()}
    (Path(output) / "environment.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance
