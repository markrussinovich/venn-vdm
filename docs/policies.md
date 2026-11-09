# Policies

<!-- docs-navigation:start -->
[Documentation](README.md) · [Get started](getting-started.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md) · [Examples](../examples/README.md) · [Task recipes](cookbooks.md) · [API](api.md) · [Results](evidence.md)

<details>
<summary>Browse all documentation</summary>

**Project home:** [Home](../README.md)

**Start here:** [Documentation overview](README.md) · [Getting started](getting-started.md) · [Installation](installation.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md)

**Task guides:** [Task recipes](cookbooks.md) · [Outcome stores](outcome-api.md)

**Examples:** [Choose an example](../examples/README.md) · [Hot dog image classifier](../examples/hotdog/README.md) · [Precedent Desk](../examples/precedent_desk/README.md) · [Wine predictor](../examples/wine/README.md) · [Audio classifier](../examples/audio/README.md) · [Screen-only arena](arena-example.md) · [Demos and videos](demos.md)

**Reference:** [API reference](api.md) · [Design rationale](design.md) · [Venn 1 model card](model-card.md)

**Results and limits:** [Performance and cost](performance.md) · [VennSpace-1](benchmark.md) · [Results and evidence](evidence.md) · [Reproduction](reproduction.md) · [Semantic boundary findings](redteam.md) · [Limitations](limitations.md) · [Comparison video methods](same-memory-video.md) · [Comparison notebooks and renderer](../examples/same_memory_video/README.md)

**Project:** **Policies** · [Licenses and attribution](licenses.md) · [Contributing to Venn](../CONTRIBUTING.md)

**Maintainers:** [Maintainer guide](maintainers/maintaining.md) · [Documentation site](maintainers/website.md) · [Release checklist](maintainers/release-checklist.md) · [Publishing the Python package](maintainers/pypi.md) · [Publishing the decision head](maintainers/publishing-the-head.md) · [Development evidence](maintainers/development-evidence.md) · [Comparative evaluation protocol](maintainers/evaluation-plan.md) · [License audit notes](maintainers/license-audit.md)

</details>

**Project / Policies**
<!-- docs-navigation:end -->

<!-- docs-contents:start -->
<details>
<summary>On this page</summary>

- [Versions and deprecation](#versions-and-deprecation)
- [Data](#data)
- [Experimental media](#experimental-media)

</details>
<!-- docs-contents:end -->

## Versions and deprecation

Model IDs identify complete profiles; `venn-1` is the EmbeddingGemma 2 profile.
Published versioned assets and tags are immutable. Breaking model or API changes
get at least **60 days' notice** before deprecation. Interfaces marked
experimental are identified as such and are never silently substituted for
stable behavior.

## Data

Raw inputs, embeddings, reference stores and registries remain local unless
the caller deliberately configures a remote endpoint. They may contain
personal or confidential information. They are not anonymized by embedding.
SQLite caches and model registries need application-level access controls,
retention, deletion and backups. No telemetry or automatic upstream payload
upload is required by Venn.

Evaluation loaders fetch dataset texts from pinned original sources under their
own terms; raw evaluation texts are not bundled in the source release.
Selected Banking77 and CLINC embedding snapshots are distributed separately
for the [Precedent Desk reproduction](../examples/precedent_desk/README.md).
Those snapshots retain their source-dataset terms and are not anonymized data.
Other evaluation corpora, vectors and fitted evaluation heads are excluded.
The MIT License covers original code, not upstream datasets or weights.

## Experimental media

Media is disabled on the Decisions surface without
`X-Venn-Experimental: media`. Opt-in acknowledges measured limitations,
not calibration or production fitness. No ASR or second-model replacement
is silently substituted. Referenced-image assets and Box are unsupported.

<!-- docs-pager:start -->
---

[← Previous: Comparison notebooks and renderer](../examples/same_memory_video/README.md) · [Next: Licenses and attribution →](licenses.md) · [Documentation home](README.md)
<!-- docs-pager:end -->
