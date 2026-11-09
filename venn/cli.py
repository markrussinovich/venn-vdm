"""The ``venn`` command line.

Subcommands: ``quickstart`` and ``sweep`` run on CPU with nothing downloaded;
``download-embedder`` and ``download-model`` fetch the pinned reader and head;
``serve`` starts the HTTP decision API; ``demos``, ``benchmark`` and ``redteam``
run the authored demos and evaluation suites; ``mcp`` runs the optional MCP bridge.
"""
import argparse
import importlib
import os
import sys
from pathlib import Path

from venn.reader import MODEL_ID, REVISION
from venn.profiles import HEAD_REPO_ID, HEAD_REVISION

# Subcommands implemented by modules with their own argparse ``main()``.
PASSTHROUGH = {
    "demos": ("venn.demos", "Run the authored retrieval, drift and million-row sweep demos"),
    "benchmark": ("venn.benchmark", "Run the VennSpace-1 geometric benchmark"),
    "redteam": ("venn.redteam", "Run the fixed semantic boundary suite"),
}


HELP_URL = "https://github.com/markrussinovich/venn-vdm/blob/main/docs/getting-started.md#troubleshooting"


def fetch(repo_id, revision, what, **kwargs):
    """Download a Hub snapshot, turning Hub and network errors into a short message."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    try:
        return snapshot_download(repo_id, revision=revision, **kwargs)
    except HfHubHTTPError as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        reason = ("not found or not accessible; check the repository name, your Hub login and that the repository is public"
                  if status in (401, 403, 404) else f"Hub error {status}")
        raise SystemExit(f"Could not download the {what} from {repo_id}@{revision[:12]}: {reason}.\nSee {HELP_URL}") from error
    except (LocalEntryNotFoundError, OSError) as error:
        raise SystemExit(f"Could not download the {what} from {repo_id}: no network access to huggingface.co "
                         f"and no cached copy ({type(error).__name__}).\nSee {HELP_URL}") from error


def download_head(repo_id=HEAD_REPO_ID, revision=HEAD_REVISION):
    """Download the published head from the Hub and return its ``assets`` directory."""
    return Path(fetch(repo_id, revision, "venn-1 head", allow_patterns=["assets/*"])) / "assets"


def main(argv=None):
    """Parse arguments and run the selected subcommand."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in PASSTHROUGH:
        module = importlib.import_module(PASSTHROUGH[argv[0]][0])
        sys.argv = [f"venn {argv[0]}", *argv[1:]]
        return module.main()
    parser = argparse.ArgumentParser(prog="venn", description="Typed decisions over embeddings.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("quickstart", help="Run a CPU example with authored vectors; nothing is downloaded",
                        description="Route a synthetic ticket from a small outcome store. Runs on CPU and downloads nothing.")
    sweep = commands.add_parser("sweep", help="Time a synthetic CPU vector sweep; nothing is downloaded")
    sweep.add_argument("--rows", type=int, default=1000, help="Synthetic vectors to score, 1..1000000 (default 1000)")
    sweep.add_argument("--batch-size", type=int, default=8192, help="Rows scored per batch, 1..65536 (default 8192)")
    serve = commands.add_parser("serve", help="Serve the HTTP decision API on loopback")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--allowed-host", action="append",
                       help="Accept this exact HTTP hostname (repeatable, no wildcards); replaces the loopback defaults "
                            "127.0.0.1, localhost and ::1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--no-preload", action="store_true",
                       help="Do not load the text reader in the background at start; it then loads on the first text "
                            "request (for low-memory or vector-only deployments)")
    serve.add_argument("--model-dir", type=Path,
                       help="Head assets directory; defaults to VENN_MODEL_DIR, else the published head is downloaded")
    serve.add_argument("--site-dir", type=Path, help="Mount a built documentation site at /guide")
    commands.add_parser("download-embedder", help="Download the pinned reader (EmbeddingGemma 2)")
    download = commands.add_parser("download-model", help="Download the venn-1 head and print its assets path")
    download.add_argument("repo_id", nargs="?", default=HEAD_REPO_ID)
    download.add_argument("--revision", default=HEAD_REVISION, help="Hub commit or tag")
    for name, (_, help_text) in PASSTHROUGH.items():
        commands.add_parser(name, help=help_text, add_help=False)
    mcp = commands.add_parser("mcp", help="Run the optional MCP stdio bridge against a running server")
    mcp.add_argument("--endpoint", default="http://127.0.0.1:8000")
    args = parser.parse_args(argv)
    if args.command == "quickstart":
        from venn.quickstart import main as quickstart

        quickstart()
    elif args.command == "sweep":
        import json

        from venn.demos import million_sweep

        if not 1 <= args.rows <= 1000000 or not 1 <= args.batch_size <= 65536:
            parser.error("Use 1..1000000 rows and a batch size in 1..65536")
        print(json.dumps(million_sweep(args.rows, args.batch_size), indent=2))
    elif args.command == "serve":
        import uvicorn

        from venn import DecisionService, create_app

        from venn.http_security import allowed_host_set

        try:
            allowed_host_set(args.allowed_host)
        except ValueError as error:
            parser.error(f"--allowed-host: {error}")
        model_dir = args.model_dir or os.environ.get("VENN_MODEL_DIR") or download_head()
        service = DecisionService(model_dir=model_dir)
        app = create_app(service, site_dir=args.site_dir, allowed_hosts=args.allowed_host)
        if not args.no_preload:
            service.preload_reader()
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.command == "mcp":
        try:
            from venn.mcp_server import create_server

            create_server(args.endpoint).run(transport="stdio")
        except ModuleNotFoundError as error:
            if error.name == "mcp" or error.name.startswith("mcp."):
                parser.error("Install the mcp extra: pip install 'venn-vdm[mcp]'")
            raise
    elif args.command == "download-embedder":
        print(fetch(MODEL_ID, REVISION, "EmbeddingGemma 2 reader"))
    else:
        print(download_head(args.repo_id, args.revision))
