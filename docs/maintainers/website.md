# Documentation site

<!-- docs-navigation:start -->
[Documentation](../README.md) · [Get started](../getting-started.md) · [Concepts](../concepts.md) · [Venn vs Jev](../venn-and-jev.md) · [Examples](../../examples/README.md) · [Task recipes](../cookbooks.md) · [API](../api.md) · [Results](../evidence.md)

<details>
<summary>Browse all documentation</summary>

**Project home:** [Home](../../README.md)

**Start here:** [Documentation overview](../README.md) · [Getting started](../getting-started.md) · [Installation](../installation.md) · [Concepts](../concepts.md) · [Venn vs Jev](../venn-and-jev.md)

**Task guides:** [Task recipes](../cookbooks.md) · [Outcome stores](../outcome-api.md)

**Examples:** [Choose an example](../../examples/README.md) · [Hot dog image classifier](../../examples/hotdog/README.md) · [Precedent Desk](../../examples/precedent_desk/README.md) · [Wine predictor](../../examples/wine/README.md) · [Audio classifier](../../examples/audio/README.md) · [Screen-only arena](../arena-example.md) · [Demos and videos](../demos.md)

**Reference:** [API reference](../api.md) · [Design rationale](../design.md) · [Venn 1 model card](../model-card.md)

**Results and limits:** [Performance and cost](../performance.md) · [VennSpace-1](../benchmark.md) · [Results and evidence](../evidence.md) · [Reproduction](../reproduction.md) · [Semantic boundary findings](../redteam.md) · [Limitations](../limitations.md) · [Comparison video methods](../same-memory-video.md) · [Comparison notebooks and renderer](../../examples/same_memory_video/README.md)

**Project:** [Policies](../policies.md) · [Licenses and attribution](../licenses.md) · [Contributing to Venn](../../CONTRIBUTING.md)

**Maintainers:** [Maintainer guide](maintaining.md) · **Documentation site** · [Release checklist](release-checklist.md) · [Publishing the Python package](pypi.md) · [Publishing the decision head](publishing-the-head.md) · [Development evidence](development-evidence.md) · [Comparative evaluation protocol](evaluation-plan.md) · [License audit notes](license-audit.md)

</details>

**Maintainers / Documentation site**
<!-- docs-navigation:end -->

<!-- docs-contents:start -->
<details>
<summary>On this page</summary>

- [Navigation](#navigation)
- [Editing Markdown navigation](#editing-markdown-navigation)
- [Preview locally](#preview-locally)
- [Deploy with GitHub Pages](#deploy-with-github-pages)
- [Private release preparation](#private-release-preparation)

</details>
<!-- docs-contents:end -->

`scripts/build_site.py` renders `README.md`, all Markdown under `docs/` and
`examples/`, and `CONTRIBUTING.md` into a static site. It copies runnable scripts,
media, clients and benchmark files alongside. The site has no backend; it never loads a model.

## Navigation

Every page has grouped documentation navigation, a current-page indicator,
previous/next links, a GitHub source link, and a section table of contents with
heading permalinks. User documentation and maintainer procedures have separate
previous/next trails. Maintainer navigation is collapsed on user pages and open
when reading a maintainer page.
Interior pages also have breadcrumbs. On smaller screens, the documentation
menu and section contents collapse above the article; both remain available
through native disclosure controls. They remain usable without JavaScript.

Search runs locally in the browser over page titles, section headings and text.
Results link directly to matching pages or sections. Search defaults to the
current audience: user documentation or maintainer documentation. Readers can
select either audience or all pages. Ctrl/Cmd+K focuses search; Enter focuses the
first result and Escape clears the query. Search requires JavaScript;
navigation and section links also work with JavaScript disabled.

The reading order and groups live in `NAVIGATION` in `scripts/build_site.py`.
New Markdown pages appear automatically under **More documentation** until added
to a group. Concise sidebar labels live in `PAGE_LABELS`; source headings and filenames do
not have to change to improve navigation. Example READMEs render as
`examples/index.html` and `examples/<name>/index.html`, with rewritten relative
links and the same shell as the rest of the documentation. Where an example
already owns `index.html` as its runnable app (the hot dog demo), its guide is
`README.html` instead. User photo, model-state and data folders are excluded. Original Markdown
downloads remain alongside them. The builder generates `search-index.js` on each build and copies
`scripts/site.js`; neither search nor navigation needs an external service.

## Editing Markdown navigation

`docs/README.md` is the documentation home shown when someone opens the folder
on GitHub. The site keeps its existing `overview.html` address; `overview.md`
is a compatibility pointer to the README.

Markdown pages include a shared navigation bar, a collapsible directory of all
documentation pages, section links and previous/next links so readers can browse
individual files on GitHub or in an editor. The
static site replaces these blocks with its own sidebar and page controls, so
navigation is not duplicated or indexed as article content.

After adding a page, changing a heading or changing `NAVIGATION`, regenerate the
marked blocks and check them:

```bash
python scripts/update_doc_navigation.py
python scripts/update_doc_navigation.py --check
python scripts/check_docs.py
```

Edit article text outside the generated markers. The navigation consistency test
checks that the committed blocks match the shared reading order and headings.

## Preview locally

```bash
python -m pip install -e ".[dev]"
python scripts/build_site.py --output /tmp/venn-site
python scripts/serve_site.py /tmp/venn-site --port 8080
```

All links are relative, so the same build works under the GitHub project path
`/venn/`. The expected URL is `https://markrussinovich.github.io/venn-vdm/`.

## Deploy with GitHub Pages

The `Documentation site` workflow builds the site on every push and pull
request to `main` and uploads it as an artifact. To deploy it:

1. In the repository settings, open **Pages** and set the source to
   **GitHub Actions**.
2. Add the repository variable `DEPLOY_PAGES` with the value `true`.

Pushes to `main` then deploy automatically after the repository is public.
The workflow checks both repository visibility and `DEPLOY_PAGES`; private
release preparation keeps `DEPLOY_PAGES=false`. Pull requests only build.
Note that a Pages site is public even when the repository is private, so enable
deployment only when the repository is ready to be public.

## Private release preparation

The planned page address is `https://markrussinovich.github.io/venn-vdm/`.
GitHub's `has_pages` field reflects an actual Pages configuration; it cannot
be enabled as an independent repository flag. The current account plan rejects
Pages for this private repository. Build a local preview with the command above
and leave deployment disabled until launch.

<!-- docs-pager:start -->
---

[← Previous: Maintainer guide](maintaining.md) · [Next: Release checklist →](release-checklist.md) · [Documentation home](../README.md)
<!-- docs-pager:end -->
