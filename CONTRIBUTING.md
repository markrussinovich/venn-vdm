# Contributing to Venn

<!-- docs-navigation:start -->
[Documentation](docs/README.md) · [Get started](docs/getting-started.md) · [Concepts](docs/concepts.md) · [Venn vs Jev](docs/venn-and-jev.md) · [Examples](examples/README.md) · [Task recipes](docs/cookbooks.md) · [API](docs/api.md) · [Results](docs/evidence.md)

<details>
<summary>Browse all documentation</summary>

**Project home:** [Home](README.md)

**Start here:** [Documentation overview](docs/README.md) · [Getting started](docs/getting-started.md) · [Installation](docs/installation.md) · [Concepts](docs/concepts.md) · [Venn vs Jev](docs/venn-and-jev.md)

**Task guides:** [Task recipes](docs/cookbooks.md) · [Outcome stores](docs/outcome-api.md)

**Examples:** [Choose an example](examples/README.md) · [Hot dog image classifier](examples/hotdog/README.md) · [Precedent Desk](examples/precedent_desk/README.md) · [Wine predictor](examples/wine/README.md) · [Audio classifier](examples/audio/README.md) · [Screen-only arena](docs/arena-example.md) · [Demos and videos](docs/demos.md)

**Reference:** [API reference](docs/api.md) · [Design rationale](docs/design.md) · [Venn 1 model card](docs/model-card.md)

**Results and limits:** [Performance and cost](docs/performance.md) · [VennSpace-1](docs/benchmark.md) · [Results and evidence](docs/evidence.md) · [Reproduction](docs/reproduction.md) · [Semantic boundary findings](docs/redteam.md) · [Limitations](docs/limitations.md) · [Comparison video methods](docs/same-memory-video.md) · [Comparison notebooks and renderer](examples/same_memory_video/README.md)

**Project:** [Policies](docs/policies.md) · [Licenses and attribution](docs/licenses.md) · **Contributing to Venn**

**Maintainers:** [Maintainer guide](docs/maintainers/maintaining.md) · [Documentation site](docs/maintainers/website.md) · [Release checklist](docs/maintainers/release-checklist.md) · [Publishing the Python package](docs/maintainers/pypi.md) · [Publishing the decision head](docs/maintainers/publishing-the-head.md) · [Development evidence](docs/maintainers/development-evidence.md) · [Comparative evaluation protocol](docs/maintainers/evaluation-plan.md) · [License audit notes](docs/maintainers/license-audit.md)

</details>

**Project / Contributing to Venn**
<!-- docs-navigation:end -->

<!-- docs-contents:start -->
<details>
<summary>On this page</summary>

- [Set up](#set-up)
- [Check your change](#check-your-change)
- [Pull requests](#pull-requests)

</details>
<!-- docs-contents:end -->

Thanks for helping. Please read the [code of conduct](CODE_OF_CONDUCT.md) first.

## Set up

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional, CPU-only
python -m pip install -e '.[dev]'
```

For Linux ARM64 or native Windows ARM64, follow the
[platform setup](docs/getting-started.md#arm64-and-windows). On Windows, use
`.\.venv\Scripts\python.exe` in place of `python` in the checks below.

## Check your change

```bash
python -m pyflakes venn scripts examples tests
python scripts/update_doc_navigation.py --check
python scripts/check_docs.py
python -m pytest -q
```

Tests use deterministic fake embeddings and small untrained fixtures, so they
need no reader or head download and no GPU. Tests for optional adapters skip
unless their extras are installed; install `.[adapters]` to run them.

CI lints, runs the tests, exercises unmodified upstream Decisions clients
against the contract server and smoke-tests the built wheel.

## Pull requests

- Explain the problem, the user-visible behavior and how you checked it.
- Add or update tests for changed behavior, and keep examples runnable.
- Keep synthetic fixtures clearly distinct from measured results.
- Do not change the frozen benchmark protocol to improve a number; create a new
  benchmark version instead.
- Never include private text, embeddings, credentials or model files in an
  issue or pull request.

Security issues go through the private channel in [SECURITY.md](SECURITY.md).
The maintainer guide is in [docs/maintainers](docs/maintainers/maintaining.md).

<!-- docs-pager:start -->
---

[← Previous: Licenses and attribution](docs/licenses.md) · [Documentation home](docs/README.md)
<!-- docs-pager:end -->
