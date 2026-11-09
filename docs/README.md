# Documentation overview

<!-- docs-navigation:start -->
**Documentation** · [Get started](getting-started.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md) · [Examples](../examples/README.md) · [Task recipes](cookbooks.md) · [API](api.md) · [Results](evidence.md)

<details>
<summary>Browse all documentation</summary>

**Project home:** [Home](../README.md)

**Start here:** **Documentation overview** · [Getting started](getting-started.md) · [Installation](installation.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md)

**Task guides:** [Task recipes](cookbooks.md) · [Outcome stores](outcome-api.md)

**Examples:** [Choose an example](../examples/README.md) · [Hot dog image classifier](../examples/hotdog/README.md) · [Precedent Desk](../examples/precedent_desk/README.md) · [Wine predictor](../examples/wine/README.md) · [Audio classifier](../examples/audio/README.md) · [Screen-only arena](arena-example.md) · [Demos and videos](demos.md)

**Reference:** [API reference](api.md) · [Design rationale](design.md) · [Venn 1 model card](model-card.md)

**Results and limits:** [Performance and cost](performance.md) · [VennSpace-1](benchmark.md) · [Results and evidence](evidence.md) · [Reproduction](reproduction.md) · [Semantic boundary findings](redteam.md) · [Limitations](limitations.md) · [Comparison video methods](same-memory-video.md) · [Comparison notebooks and renderer](../examples/same_memory_video/README.md)

**Project:** [Policies](policies.md) · [Licenses and attribution](licenses.md) · [Contributing to Venn](../CONTRIBUTING.md)

**Maintainers:** [Maintainer guide](maintainers/maintaining.md) · [Documentation site](maintainers/website.md) · [Release checklist](maintainers/release-checklist.md) · [Publishing the Python package](maintainers/pypi.md) · [Publishing the decision head](maintainers/publishing-the-head.md) · [Development evidence](maintainers/development-evidence.md) · [Comparative evaluation protocol](maintainers/evaluation-plan.md) · [License audit notes](maintainers/license-audit.md)

</details>

**Start here / Documentation overview**
<!-- docs-navigation:end -->

<!-- docs-contents:start -->
<details>
<summary>On this page</summary>

- [Get a first result](#get-a-first-result)
- [Choose a workflow](#choose-a-workflow)
- [Find the right kind of documentation](#find-the-right-kind-of-documentation)
- [Contribute to Venn](#contribute-to-venn)

</details>
<!-- docs-contents:end -->

Venn makes typed decisions over reusable embeddings. Start with a small CPU
example, then choose whether a question should use described options, historical
cases, or a predictor fitted on labeled examples.

## Get a first result

Follow [getting started](getting-started.md) to install from the checkout and run
`venn quickstart`. It needs no reader or head download and uses authored vectors
to show the interface. For platform-specific Python and PyTorch setup, use
[installation](installation.md).

Read [concepts](concepts.md) for readers, compatible profiles, stores and families.
[Venn or Jev?](venn-and-jev.md) explains when reusable state helps and where a
text-reading decision model is a better fit.

## Choose a workflow

| Goal | Start here | Continue with |
| --- | --- | --- |
| Decide from past cases and their outcomes | [Outcome stores](outcome-api.md) | [Small runnable example](../examples/outcome_store.py), then [arena application](arena-example.md) |
| Ask Choice, Score or Noul questions about text | [Serve text questions](getting-started.md#serve-text-questions) | [Routing recipe](cookbooks.md#route-a-support-ticket), [API reference](api.md#question-types) |
| Fit a classifier or numeric predictor | [Families](concepts.md#families) | [Task recipes](cookbooks.md#fit-and-reload-a-local-family), [wine predictor](../examples/wine/README.md), [audio classifier](../examples/audio/README.md) |
| Train a local image classifier from personal photos | [Hot dog image demo](../examples/hotdog/README.md) | [Linear-family recipe](cookbooks.md#fit-a-linear-choice-family), [audio classifier](../examples/audio/README.md) |
| Route support tickets from resolved history | [Precedent Desk](../examples/precedent_desk/README.md) | [Index lifecycle](api.md#index-lifecycle) and [collection sweep](cookbooks.md#sweep-a-real-vector-store) |
| See the system before running it | [Demos and videos](demos.md) | [Complete example catalog](../examples/README.md) |

Each example lists its own prerequisites. Authored-vector examples demonstrate
contracts; real-reader workflows download model weights. Check the
[profile compatibility rules](concepts.md#profiles-and-compatibility) before
connecting an existing store or changing readers.

## Find the right kind of documentation

- **Start here:** setup, the main concepts, and a comparison of approaches.
- **Task guides:** recipes and implementation steps for a specific operation.
- **Examples:** complete applications, scripts and videos with prerequisites.
- **Reference:** request and response contracts, design rationale and model details.
- **Results and limits:** measurements, evidence, reproduction and failure boundaries.
- **Project:** contribution, policy and licensing information.

For exact fields and errors, go to the [API reference](api.md). For interpreting
reported results, use the [evidence ledger](evidence.md) and
[limitations](limitations.md); [reproduction](reproduction.md) separates contract
checks, geometric measurements and trained-head evaluation.

## Contribute to Venn

See [contributing](../CONTRIBUTING.md) for development changes, bug reports and
documentation improvements.

<!-- docs-pager:start -->
---

[← Previous: Home](../README.md) · [Next: Getting started →](getting-started.md) · [Documentation home](README.md)
<!-- docs-pager:end -->
