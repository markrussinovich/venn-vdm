"""Stage and publish the venn-1 decision head to a Hugging Face model repository.

The staged bundle holds only what the loader needs (``assets/config.json``,
``assets/manifest.json``, ``assets/universal.safetensors``) plus a model card built
from ``docs/model-card.md``. The weights and the loader configuration are checked
against the pinned checksums before anything is copied, so a different head cannot
be published by accident. Repositories are created private unless ``--public`` is given.

    python scripts/publish_head.py --model-dir /path/to/assets --output /tmp/venn-head --dry-run
    python scripts/publish_head.py --model-dir /path/to/assets --output /tmp/venn-head --repo-id owner/venn-1
"""
import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

from venn.profiles import HEAD_CONFIG_SHA256, HEAD_REPO_ID, HEAD_SHA256, VENN_1

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/markrussinovich/venn-vdm"
PUBLIC_CONFIG_KEYS = ("format_version", "baseline", "choice_temperature", "reader", "default_head")
CARD_FRONT_MATTER = """---
base_model: google/embeddinggemma-2
language:
  - en
tags:
  - venn
  - vector-decision-model
  - decision-head
  - abstention
  - embeddings
library_name: venn-vdm
license: other
license_name: venn-1-redistribution-permission
license_link: https://huggingface.co/markrussinovich/venn-1/blob/main/README.md#redistribution
---

"""


def sha256(path):
    """Return the hex SHA-256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_card(source=ROOT / "docs/model-card.md"):
    """Build the Hub model card: front matter plus the model card with links made absolute."""
    body = Path(source).read_text(encoding="utf-8")
    # Repository navigation, contents and pager blocks are for the docs site, not the Hub card.
    body = re.sub(r"<!-- docs-(navigation|contents|pager):start -->.*?<!-- docs-\1:end -->\n*", "", body, flags=re.S)
    body = re.sub(r"; pin revision\s+`[0-9a-f]{40}`", "; use the [immutable revision pinned in the runtime]("
                  + REPOSITORY + "/blob/main/venn/profiles.py)", body)
    body = re.sub(r"\]\(\.\./", f"]({REPOSITORY}/blob/main/", body)
    body = re.sub(r"\]\((?!https?://|#)([^)]+\.md(?:#[^)]*)?)\)", rf"]({REPOSITORY}/blob/main/docs/\1)", body)
    return CARD_FRONT_MATTER + body


def stage(model_dir, output, card_source=ROOT / "docs/model-card.md"):
    """Validate a head directory and write the loader bundle plus model card to ``output``.

    Raises ``ValueError`` if the weights or loader configuration differ from the pinned
    venn-1 head, or if ``output`` already exists.
    """
    model_dir, output = Path(model_dir), Path(output)
    if output.exists():
        raise ValueError("Use a new output directory")
    weights = model_dir / "universal.safetensors"
    if not weights.is_file() or sha256(weights) != HEAD_SHA256:
        raise ValueError("universal.safetensors is not the pinned venn-1 head")
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("reader") != {"id": VENN_1.reader, "revision": VENN_1.revision, "dimensions": VENN_1.dimensions}:
        raise ValueError("config.json does not pin the venn-1 reader")
    try:
        public = {key: config[key] for key in PUBLIC_CONFIG_KEYS}
    except KeyError as error:
        raise ValueError(f"config.json is missing {error.args[0]}") from error
    digest = hashlib.sha256(json.dumps(public, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != HEAD_CONFIG_SHA256:
        raise ValueError("Loader configuration differs from the pinned venn-1 head")
    assets = output / "assets"
    assets.mkdir(parents=True)
    (assets / "config.json").write_text(json.dumps(public, indent=2) + "\n")
    shutil.copy2(weights, assets / "universal.safetensors")
    hashes = {path.name: sha256(path) for path in sorted(assets.iterdir())}
    (assets / "manifest.json").write_text(json.dumps({"sha256": hashes}, indent=2) + "\n")
    (output / "README.md").write_text(model_card(card_source), encoding="utf-8")
    return output


def publish(output, repo_id=HEAD_REPO_ID, public=False, dry_run=False, token=None):
    """Create the Hub repository if needed and upload a staged bundle. Returns the commit SHA.

    ``dry_run`` validates the bundle and returns ``None`` without any network call.
    """
    output = Path(output)
    expected = {"README.md", "assets/config.json", "assets/manifest.json", "assets/universal.safetensors"}
    present = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    if present != expected:
        raise ValueError(f"Bundle must contain exactly {sorted(expected)}")
    manifest = json.loads((output / "assets/manifest.json").read_text(encoding="utf-8"))["sha256"]
    for name, digest in manifest.items():
        if sha256(output / "assets" / name) != digest:
            raise ValueError(f"Checksum mismatch: assets/{name}")
    if manifest["universal.safetensors"] != HEAD_SHA256:
        raise ValueError("Bundle does not hold the pinned venn-1 head")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo_id):
        raise ValueError("Provide a Hugging Face namespace/repository")
    if dry_run:
        print(f"Validated {len(present)} files for {repo_id}; no network calls")
        return None
    from huggingface_hub import HfApi

    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=repo_id, repo_type="model", private=not public, exist_ok=True)
    commit = api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(output),
                               commit_message="Publish venn-1 decision head")
    print(f"Published {repo_id}@{commit.oid} ({'public' if public else 'private'})")
    return commit.oid


def main():
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory holding config.json and universal.safetensors")
    parser.add_argument("--output", type=Path, required=True, help="New directory for the staged bundle")
    parser.add_argument("--repo-id", default=HEAD_REPO_ID)
    parser.add_argument("--public", action="store_true", help="Create a public repository (default: private)")
    parser.add_argument("--dry-run", action="store_true", help="Stage and validate without contacting the Hub")
    args = parser.parse_args()
    stage(args.model_dir, args.output)
    publish(args.output, args.repo_id, public=args.public, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
