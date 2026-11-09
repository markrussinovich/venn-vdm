"""Check local documentation links and declared installation extras."""
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def check(root=ROOT):
    extras = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    errors = []
    for path in [root / "README.md", root / "CONTRIBUTING.md",
                 *sorted((root / "docs").rglob("*.md")), *sorted((root / "examples").rglob("*.md"))]:
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"\]\(([^\s)]+)\)", text):
            url = urlsplit(match[1])
            if not url.scheme and not url.netloc and url.path:
                if not (path.parent / unquote(url.path)).is_file():
                    errors.append(f"{path.relative_to(root)}: missing link {match[1]}")
        for match in re.finditer(r"\.\[([a-zA-Z0-9_, -]+)\]", text):
            for extra in match[1].split(","):
                if extra.strip() not in extras:
                    errors.append(f"{path.relative_to(root)}: unknown extra {extra.strip()}")
    return errors


if __name__ == "__main__":
    errors = check()
    print("\n".join(errors) if errors else "Documentation paths and installation extras OK")
    raise SystemExit(bool(errors))
