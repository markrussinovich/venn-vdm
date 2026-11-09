"""Render the documentation into a static site for GitHub Pages or local preview."""
import argparse
import html
import json
import posixpath
import re
import shutil
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import markdown


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/markrussinovich/venn-vdm"
MEDIA_RELEASE = REPOSITORY + "/releases/download/media-v1"
VIDEO_URL = MEDIA_RELEASE + "/venn-arena.mp4"
# Reading order also drives the sidebar and breadcrumbs. Maintainer pages have
# their own reading trail so user documentation does not lead into release work.
NAVIGATION = (
    ("Start here", ("overview", "getting-started", "installation", "concepts", "venn-and-jev")),
    ("Task guides", ("cookbooks", "outcome-api")),
    ("Examples", ("examples/index", "examples/hotdog/README", "examples/precedent_desk/index", "examples/wine/index",
                  "examples/audio/index", "arena-example", "demos")),
    ("Reference", ("api", "design", "model-card")),
    ("Results and limits", ("performance", "benchmark", "evidence", "reproduction", "redteam",
                            "limitations", "same-memory-video", "examples/same_memory_video/index")),
    ("Project", ("policies", "licenses", "CONTRIBUTING")),
    ("Maintainers", ("maintainers/maintaining", "maintainers/website", "maintainers/release-checklist",
                     "maintainers/pypi", "maintainers/publishing-the-head", "maintainers/development-evidence",
                     "maintainers/evaluation-plan", "maintainers/license-audit")),
)
PAGE_LABELS = {
    "overview": "Documentation overview",
    "venn-and-jev": "Venn vs Jev",
    "evidence": "Results and evidence",
    "cookbooks": "Task recipes",
    "outcome-api": "Outcome stores",
    "examples/index": "Choose an example",
    "examples/hotdog/README": "Hot dog image classifier",
    "examples/precedent_desk/index": "Precedent Desk",
    "examples/wine/index": "Wine predictor",
    "examples/audio/index": "Audio classifier",
    "arena-example": "Screen-only arena",
    "demos": "Demos and videos",
    "api": "API reference",
    "same-memory-video": "Comparison video methods",
    "examples/same_memory_video/index": "Comparison notebooks and renderer",
}


def plain_text(value):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value)).split())


def navigation_groups(targets):
    available = {target.removesuffix(".html") for target in targets.values()} - {"index"}
    groups = [(label, [name for name in names if name in available]) for label, names in NAVIGATION]
    covered = {name for _, names in groups for name in names}
    groups.append(("More documentation", sorted(available - covered)))
    return [(label, names) for label, names in groups if names]


def source_pages(root):
    """Return documentation sources and their stable HTML destinations."""
    docs = root / "docs"
    example_pages = [page for page in sorted((root / "examples").rglob("*.md"))
                     if not {"photos", "local-state", "data", "__pycache__"}.intersection(page.relative_to(root / "examples").parts)]
    canonical = docs / "README.md"
    compatibility = docs / "overview.md"
    pages = [root / "README.md", *[page for page in sorted(docs.rglob("*.md"))
                                  if not (canonical.is_file() and page == compatibility)], *example_pages]
    if (root / "CONTRIBUTING.md").is_file():
        pages.append(root / "CONTRIBUTING.md")
    targets = {}
    for page in pages:
        if page == root / "README.md":
            target = "index.html"
        elif page == canonical:
            target = "overview.html"
        elif page.is_relative_to(docs):
            target = page.relative_to(docs).with_suffix(".html").as_posix()
        elif page.name == "README.md" and not page.with_name("index.html").is_file():
            target = (page.parent.relative_to(root) / "index.html").as_posix()
        else:
            target = page.relative_to(root).with_suffix(".html").as_posix()
        targets[page.resolve()] = target
    if canonical.is_file() and compatibility.is_file():
        targets[compatibility.resolve()] = "overview.html"
    return pages, targets


def without_source_navigation(text):
    """The HTML shell replaces navigation embedded for GitHub Markdown readers."""
    return re.sub(r'<!-- docs-(navigation|contents|pager):start -->.*?<!-- docs-\1:end -->\s*',
                  "", text, flags=re.DOTALL)


HERO = """<section class="hero" aria-labelledby="hero-title">
<div><div class="eyebrow">Embedding-native decision systems</div>
<h1 id="hero-title">Venn: <span class="model-type">Vector Decision Models</span></h1>
<p>Typed decisions over embeddings you already have: route, score, filter and
compare populations with a small local decision layer instead of a generative model.</p>
<p>When a lookup is not enough, train a small custom head on your own data through
the same API, as a linear classifier, a residual Choice head or a numeric predictor, and serve it under the
question you already ask.</p>
<div class="actions"><a class="button primary" href="getting-started.html">Get started</a>
<a class="button" href="https://github.com/markrussinovich/venn-vdm">View repository</a></div></div>
<svg class="diagram" viewBox="0 0 400 330" role="img" aria-labelledby="diagram-title">
<title id="diagram-title">Cached vectors feed a small Venn decision layer and multiple decisions</title>
<g fill="none" stroke="#2a404b" stroke-width="1.5">
<circle cx="150" cy="135" r="95"/><circle cx="245" cy="135" r="95"/>
<path d="M75 260H325M100 260V230M200 260V215M300 260V230"/></g>
<g fill="#8ce8ce">
<circle cx="106" cy="100" r="4"/><circle cx="130" cy="157" r="4"/>
<circle cx="105" cy="180" r="4"/><circle cx="145" cy="75" r="4"/>
<circle cx="195" cy="112" r="6"/><circle cx="210" cy="145" r="6"/>
<circle cx="186" cy="165" r="6"/></g>
<g fill="#7899ce">
<circle cx="260" cy="85" r="4"/><circle cx="295" cy="140" r="4"/>
<circle cx="266" cy="180" r="4"/><circle cx="245" cy="120" r="4"/></g>
<text x="60" y="25">Your vectors</text><text x="245" y="25">Your questions</text>
<text x="158" y="205">Venn</text>
<text x="64" y="290">Route</text><text x="177" y="290">Score</text>
<text x="278" y="290">Filter</text>
<text class="caption" x="60" y="320">Reusable state. Small heads you can train.</text></svg>
</section>"""


def build(output):
    """Write the rendered site to ``output``, which must not exist yet."""
    if output.exists():
        raise ValueError("Use a new empty site output")
    docs = ROOT / "docs"
    video_urls = {
        stem: (f"media/{stem}.mp4" if (docs / "media" / f"{stem}.mp4").is_file()
               else f"{MEDIA_RELEASE}/{stem}.mp4")
        for stem in ("venn-arena", "venn-same-memory-review", "venn-same-memory-vertical-review")
    }
    output.mkdir(parents=True)
    pages, page_targets = source_pages(ROOT)
    example_pages = [page for page in pages if page.is_relative_to(ROOT / "examples")]
    groups = navigation_groups(page_targets)
    order = ["index", *(name for _, names in groups for name in names if not name.startswith("maintainers/"))]
    maintainer_order = [name for _, names in groups for name in names if name.startswith("maintainers/")]
    labels = {}
    for page in pages:
        heading = re.search(r"^# (.+)$", page.read_text(encoding="utf-8"), re.MULTILINE)
        name = page_targets[page.resolve()].removesuffix(".html")
        labels[name] = PAGE_LABELS.get(name) or (plain_text(markdown.markdown(heading[1])) if heading else page.stem.replace("-", " ").title())
    labels["index"] = "Home"
    search_entries = []

    for path in pages:
        current = page_targets[path.resolve()].removesuffix(".html")
        prefix = "../" * current.count("/")
        page_video_urls = {stem: url if urlsplit(url).scheme else prefix + url
                           for stem, url in video_urls.items()}

        def page_link(match):
            url = urlsplit(match[1])
            if url.scheme or url.netloc or not url.path:
                return match[0]
            source = (path.parent / url.path).resolve()
            target = page_targets.get(source)
            if target is None:
                base = docs if source.is_relative_to(docs) else ROOT
                if not source.is_relative_to(base):
                    return match[0]
                target = source.relative_to(base).as_posix()
            relative = posixpath.relpath(target, posixpath.dirname(current) or ".")
            return "](" + urlunsplit(("", "", relative, url.query, url.fragment)) + ")"

        body = re.sub(r'\]\(([^\s)]+)\)', page_link,
                      without_source_navigation(path.read_text(encoding="utf-8")))
        # GitHub renders bare attachment URLs as players. Restore the poster
        # markup here so the site uses its own player, captions and media URL.
        body = re.sub(
            r'<!-- video: (venn-arena|venn-same-memory-(?:vertical-)?review) -->\s*'
            r'https://github\.com/user-attachments/assets/[\w-]+',
            lambda match: f'[![Watch the video](media/{match[1]}-poster.jpg)]'
                          f'({MEDIA_RELEASE}/{match[1]}.mp4)',
            body,
        )
        for stem, url in page_video_urls.items():
            body = body.replace(f"{MEDIA_RELEASE}/{stem}.mp4", url)
        landing = current == "index"
        # The hero supplies the title and introduction on the landing page, so the
        # README body starts at its first section heading.
        if landing and "\n## " in body:
            body = body[body.index("\n## ") + 1:]
        renderer = markdown.Markdown(
            extensions=["fenced_code", "tables", "toc"],
            extension_configs={"toc": {"permalink": "#", "permalink_title": "Link to this section", "toc_depth": "2-3"}},
        )
        content = renderer.convert(body)
        toc = ('<details class="page-contents" open><summary>On this page</summary>'
               '<nav aria-label="On this page">' + renderer.toc + '</nav></details>') if renderer.toc_tokens else ""
        # Index sections separately so a result can jump straight to the answer.
        sections = re.split(r'<h[23]\b[^>]*id="([^"]+)"[^>]*>(.*?)</h[23]>', content, flags=re.DOTALL)
        search_entries.append({"title": labels[current], "section": "", "url": current + ".html",
                               "text": plain_text(sections[0]),
                               "audience": "maintainer" if current.startswith("maintainers/") else "user"})
        for i in range(1, len(sections), 3):
            search_entries.append({"title": labels[current], "section": plain_text(sections[i + 1]).removesuffix("#").strip(),
                                   "url": current + ".html#" + sections[i], "text": plain_text(sections[i + 2]),
                                   "audience": "maintainer" if current.startswith("maintainers/") else "user"})
        content = re.sub(r'<img\b[^>]*src="media/million-sweep.gif"[^>]*>',
                         r'<details class="recording"><summary>Show synthetic sweep recording (animated)</summary>\g<0></details>',
                         content)
        content = re.sub(
            r'<p>\s*<a href="[^"]*venn-arena\.mp4"><img\b[^>]*></a>\s*</p>',
            '<figure class="arena-video"><video controls playsinline preload="none" '
            f'width="1920" height="1080" poster="{prefix}media/venn-arena-poster.jpg" '
            'aria-label="Venn screen-only arena application">'
            f'<source src="{page_video_urls["venn-arena"]}" type="video/mp4">'
            f'<track kind="captions" src="{prefix}media/venn-arena-captions.vtt" '
            'srclang="en" label="English">'
            f'<a href="{page_video_urls["venn-arena"]}">Download the video</a></video>'
            '<figcaption>Watch Venn use past experience to choose each move.</figcaption></figure>',
            content,
        )
        def companion_video(match):
            stem = match[1]
            url = page_video_urls[stem]
            vertical = "vertical" in stem
            width, height = (1080, 1920) if vertical else (1920, 1080)
            label = "Decisions over stored embeddings: " + ("vertical" if vertical else "landscape") + " review draft"
            return (
                '<figure class="arena-video"><video controls playsinline preload="none" '
                f'width="{width}" height="{height}" poster="{prefix}media/{stem}-poster.jpg" '
                f'aria-label="{label}"><source src="{url}" type="video/mp4">'
                f'<track kind="captions" src="{prefix}media/{stem}-captions.vtt" srclang="en" label="English">'
                f'<a href="{url}">Download the review draft</a></video>'
                '<figcaption>Local review draft. Public release is on hold.</figcaption></figure>'
            )
        content = re.sub(
            r'<p>\s*<a href="[^"]*/(venn-same-memory-(?:vertical-)?review)\.mp4"><img\b[^>]*></a>\s*</p>',
            companion_video, content,
        )
        if landing:
            content = re.sub(r"^<h1\b[^>]*>.*?</h1>\s*", "", content, count=1)
        target = output / page_targets[path.resolve()]
        target.parent.mkdir(parents=True, exist_ok=True)
        title = "Venn - Vector Decision Models" if landing else f"{labels[current]} | Venn"

        def link(name, label=None, **attrs):
            attributes = "".join(f' {key}="{html.escape(value, quote=True)}"' for key, value in attrs.items())
            active = ' aria-current="page"' if name == current else ""
            return f'<a href="{prefix}{name}.html"{active}{attributes}>{html.escape(label or labels[name])}</a>'

        nav = " ".join(link(name, label) for name, label in
                       (("overview", "Docs"), ("getting-started", "Get started"),
                        ("examples/index", "Examples"), ("api", "API")) if name in labels)
        sidebar = link("index")
        for label, names in groups:
            items = "".join(f'<li>{link(name)}</li>' for name in names)
            if label == "Maintainers":
                opened = " open" if current.startswith("maintainers/") else ""
                sidebar += (f'<details class="maintainer-navigation"{opened}><summary>Maintainers</summary>'
                            f'<ul>{items}</ul></details>')
            else:
                sidebar += f'<section><h2>{label}</h2><ul>{items}</ul></section>'
        group = next((label for label, names in groups if current in names), "")
        breadcrumbs = "" if landing else (
            '<nav class="breadcrumbs" aria-label="Breadcrumb">' + link("index") +
            f'<span aria-hidden="true">/</span><span>{html.escape(group)}</span>'
            f'<span aria-hidden="true">/</span><span aria-current="page">{html.escape(labels[current])}</span></nav>'
        )
        trail = maintainer_order if current.startswith("maintainers/") else order
        index = trail.index(current)
        previous = link(trail[index - 1], "← Previous: " + labels[trail[index - 1]], rel="prev") if index else "<span></span>"
        following = link(trail[index + 1], "Next: " + labels[trail[index + 1]] + " →", rel="next") if index + 1 < len(trail) else ""
        pager = f'<nav class="page-navigation" aria-label="Previous and next pages">{previous}{following}</nav>'
        search = ('<section class="docs-search" aria-label="Documentation search">'
                  '<label for="docs-query">Search documentation</label>'
                  '<input id="docs-query" type="search" placeholder="Search pages and sections…" '
                  'autocomplete="off" aria-controls="search-results" aria-describedby="search-status">'
                  '<p id="search-status" role="status" aria-live="polite"></p>'
                  '<ol id="search-results"></ol></section>')
        source_url = REPOSITORY + "/blob/main/" + path.relative_to(ROOT).as_posix()
        page_source = f'<p class="page-source"><a href="{html.escape(source_url, quote=True)}">View this page on GitHub</a></p>'
        target.write_text(
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{html.escape(title)}</title>'
            '<meta name="description" content="Venn: typed decisions over embeddings. '
            'Route, score, filter and compare populations with a small local decision layer.">'
            f'<meta property="og:title" content="{html.escape(title)}">'
            '<meta property="og:type" content="website">'
            f'<link rel="icon" type="image/svg+xml" href="{prefix}media/venn-logo.svg">'
            f'<link rel="stylesheet" href="{prefix}site.css">'
            f'<script defer src="{prefix}search-index.js"></script>'
            f'<script defer src="{prefix}site.js"></script></head><body>'
            '<a class="skip" href="#main">Skip to content</a>'
            f'<header class="bar"><a class="brand" href="{prefix}index.html">'
            f'<img src="{prefix}media/venn-logo.svg" alt="" width="34" height="34">Venn</a>'
            f'<nav aria-label="Primary">{nav}<a href="{REPOSITORY}">GitHub</a></nav></header>'
            '<div class="docs-layout"><aside class="docs-sidebar">'
            f'{search}<details class="docs-menu" open><summary>Browse documentation</summary>'
            f'<nav aria-label="Documentation">{sidebar}</nav></details></aside>'
            # Restore the sidebar's scroll position before first paint so it stays put across pages.
            '<script>try{const s=document.querySelector(".docs-sidebar"),y=sessionStorage.getItem("venn-sidebar");'
            'if(s&&y)s.scrollTop=+y}catch(e){}</script>'
            f'<main id="main" tabindex="-1" class="{"landing" if landing else "documentation"}">'
            f'{breadcrumbs}{HERO if landing else ""}<div class="reading-layout">'
            f'<article class="content">{content}{pager}{page_source}</article>{toc}</div></main></div>'
            f'<footer><nav aria-label="Project links">{link("index")}<a href="{REPOSITORY}">GitHub</a></nav>'
            'Venn · Vector Decision Model · Research preview · Original code: MIT License'
            '</footer></body></html>',
            encoding="utf-8",
        )
    shutil.copy2(docs / "site.css", output / "site.css")
    shutil.copy2(Path(__file__).with_name("site.js"), output / "site.js")
    (output / "search-index.js").write_text("window.VENN_SEARCH = " + json.dumps(search_entries, ensure_ascii=True) + ";\n", encoding="utf-8")
    for relative in ("clients/venn.mjs", "skills/venn/SKILL.md"):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    # Keep Markdown downloads alongside the rendered maintainer/example pages.
    for source in [*sorted((docs / "maintainers").glob("*.md")),
                   *example_pages]:
        relative = source.relative_to(docs) if source.is_relative_to(docs) else source.relative_to(ROOT)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (output / "media").mkdir()
    shutil.copy2(docs / "media/million-sweep.gif", output / "media/million-sweep.gif")
    shutil.copy2(docs / "media/venn-logo.svg", output / "media/venn-logo.svg")
    for name in ("hotdog-burger-screenshot.png", "hotdog-burger-result.json", "venn-arena.mp4", "venn-arena-poster.jpg", "venn-arena-storyboard.png",
                 "venn-arena-captions.srt", "venn-arena-captions.vtt", "venn-arena-manifest.json",
                 "precedent-desk-poster.jpg", "precedent-desk-storyboard.png",
                 "precedent-desk-captions.srt", "precedent-desk-captions.vtt", "precedent-desk-manifest.json",
                 "freedoom-copying.txt", "freedoom-credits.txt"):
        if (docs / "media" / name).is_file():
            shutil.copy2(docs / "media" / name, output / "media" / name)
    desk = ROOT / "examples" / "precedent_desk"
    for source in sorted((docs / "media").glob("venn-same-memory-*")):
        if source.is_file():
            shutil.copy2(source, output / "media" / source.name)
    evidence = docs / "media" / "same-memory"
    if evidence.is_dir():
        shutil.copytree(evidence, output / "media" / "same-memory")
    for name in ("same_memory_video", "wine", "audio"):
        example = ROOT / "examples" / name
        if example.is_dir():
            shutil.copytree(example, output / "examples" / name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), dirs_exist_ok=True)
    # The local image demo owns an index.html application. Keep it distinct from
    # its rendered README, and never copy user photos or saved model state.
    for name in ("hotdog.py", "index.html"):
        source = ROOT / "examples/hotdog" / name
        if source.is_file():
            destination = output / "examples/hotdog" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    if desk.is_dir():
        for relative in ("README.md", "eval.py", "app.py", "resolution_notes.json", "queue_descriptions.json", "static/index.html"):
            if (desk / relative).is_file():
                target = output / "examples" / "precedent_desk" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(desk / relative, target)
    for name in ("LICENSE", "NOTICE", "CITATION.cff", "SECURITY.md", "CODE_OF_CONDUCT.md"):
        if (ROOT / name).is_file():
            shutil.copy2(ROOT / name, output / name)
    for directory, pattern in (("benchmarks", "*.json"), ("examples", "*.py")):
        for source in (ROOT / directory).glob(pattern):
            target = output / directory / source.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(source, target)
    if (ROOT / "examples/README.md").is_file():
        shutil.copy2(ROOT / "examples/README.md", output / "examples/README.md")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
