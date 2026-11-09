# Limitations

<!-- docs-navigation:start -->
[Documentation](README.md) · [Get started](getting-started.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md) · [Examples](../examples/README.md) · [Task recipes](cookbooks.md) · [API](api.md) · [Results](evidence.md)

<details>
<summary>Browse all documentation</summary>

**Project home:** [Home](../README.md)

**Start here:** [Documentation overview](README.md) · [Getting started](getting-started.md) · [Installation](installation.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md)

**Task guides:** [Task recipes](cookbooks.md) · [Outcome stores](outcome-api.md)

**Examples:** [Choose an example](../examples/README.md) · [Hot dog image classifier](../examples/hotdog/README.md) · [Precedent Desk](../examples/precedent_desk/README.md) · [Wine predictor](../examples/wine/README.md) · [Audio classifier](../examples/audio/README.md) · [Screen-only arena](arena-example.md) · [Demos and videos](demos.md)

**Reference:** [API reference](api.md) · [Design rationale](design.md) · [Venn 1 model card](model-card.md)

**Results and limits:** [Performance and cost](performance.md) · [VennSpace-1](benchmark.md) · [Results and evidence](evidence.md) · [Reproduction](reproduction.md) · [Semantic boundary findings](redteam.md) · **Limitations** · [Comparison video methods](same-memory-video.md) · [Comparison notebooks and renderer](../examples/same_memory_video/README.md)

**Project:** [Policies](policies.md) · [Licenses and attribution](licenses.md) · [Contributing to Venn](../CONTRIBUTING.md)

**Maintainers:** [Maintainer guide](maintainers/maintaining.md) · [Documentation site](maintainers/website.md) · [Release checklist](maintainers/release-checklist.md) · [Publishing the Python package](maintainers/pypi.md) · [Publishing the decision head](maintainers/publishing-the-head.md) · [Development evidence](maintainers/development-evidence.md) · [Comparative evaluation protocol](maintainers/evaluation-plan.md) · [License audit notes](maintainers/license-audit.md)

</details>

**Results and limits / Limitations**
<!-- docs-navigation:end -->

Venn is not universally calibrated, a privacy boundary, an authorization
system or a validated guardrail. Embeddings can be inverted. Negation,
numbers, dates, identity and adversarial paraphrases can defeat a frozen
embedding even when nearby states look geometrically similar.

Images/audio are experimental and do not have established parity with text.
Media training/use terms need separate review.

Compatibility tests establish wire format, not semantic correctness.
Conditional option probabilities can be overconfident, and standard SDKs
may drop Venn-specific abstention metadata. A permutation p-value is not a
posterior. Population-shift sensitivity depends on the sample size and shift.

The [fixed semantic boundary search](redteam.md) exposes paraphrase/format
flips and especially poor numeric robustness. The [space benchmark](benchmark.md)
reports only its headless frozen geometric tasks, with conditional
finite-corpus intervals. This repository contains no model weights, training
runs or raw evaluation archives. Download the published head for trained-head
inference.

Outcome utilities depend on the app's reward definition, observed feedback,
exploration and store freshness. They are not calibrated probabilities.
Approximate search, duplicate-state merging and exact-key overrides can change
the policy; equal vector dimensions do not establish compatible representations.
The experiments behind the outcome API used one seed each and do not establish
generalization to other applications. HTTP outcome-only serving still requires
the head; only the in-process path is headless.

<!-- docs-pager:start -->
---

[← Previous: Semantic boundary findings](redteam.md) · [Next: Comparison video methods →](same-memory-video.md) · [Documentation home](README.md)
<!-- docs-pager:end -->
