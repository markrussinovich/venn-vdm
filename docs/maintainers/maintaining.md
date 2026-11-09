# Maintainer guide

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

**Maintainers:** **Maintainer guide** · [Documentation site](website.md) · [Release checklist](release-checklist.md) · [Publishing the Python package](pypi.md) · [Publishing the decision head](publishing-the-head.md) · [Development evidence](development-evidence.md) · [Comparative evaluation protocol](evaluation-plan.md) · [License audit notes](license-audit.md)

</details>

**Maintainers / Maintainer guide**
<!-- docs-navigation:end -->

<!-- docs-contents:start -->
<details>
<summary>On this page</summary>

- [Everyday checks](#everyday-checks)
- [Release](#release)
- [History hygiene](#history-hygiene)

</details>
<!-- docs-contents:end -->

This folder holds material for maintainers. User documentation lives one level up.

## Everyday checks

The tracked tree is the public tree. A test checks that no private material
(design notes, run outputs, weights, caches) is tracked.

```bash
python -m pyflakes venn scripts examples tests
python scripts/check_docs.py
python -m pytest -q
python -m build
```

## Release

1. Update the version in `pyproject.toml` and `CITATION.cff`.
2. Build the site and the wheel locally and smoke-test the wheel from outside
   the checkout: `venn quickstart` and `venn sweep --rows 19 --batch-size 8`.
3. Tag `vX.Y.Z` on `main`. CI checks the tag against the project version, runs
   everything and uploads the distributions as a workflow artifact.
4. Publish the distributions through the manually dispatched Trusted Publishing
   workflow after the launch decision (see [PyPI setup](pypi.md)).

The decision head is published separately; see [publishing the head](publishing-the-head.md).
The documentation site is described in [website](website.md).

## History hygiene

Deleting a file from the tree does not remove it from Git history. Before
changing repository visibility, inspect the full history for anything that
must not be public and rewrite it if necessary.

<!-- docs-pager:start -->
---

[Next: Documentation site →](website.md) · [Documentation home](../README.md)
<!-- docs-pager:end -->
