# Semantic boundary findings

<!-- docs-navigation:start -->
[Documentation](README.md) · [Get started](getting-started.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md) · [Examples](../examples/README.md) · [Task recipes](cookbooks.md) · [API](api.md) · [Results](evidence.md)

<details>
<summary>Browse all documentation</summary>

**Project home:** [Home](../README.md)

**Start here:** [Documentation overview](README.md) · [Getting started](getting-started.md) · [Installation](installation.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md)

**Task guides:** [Task recipes](cookbooks.md) · [Outcome stores](outcome-api.md)

**Examples:** [Choose an example](../examples/README.md) · [Hot dog image classifier](../examples/hotdog/README.md) · [Precedent Desk](../examples/precedent_desk/README.md) · [Wine predictor](../examples/wine/README.md) · [Audio classifier](../examples/audio/README.md) · [Screen-only arena](arena-example.md) · [Demos and videos](demos.md)

**Reference:** [API reference](api.md) · [Design rationale](design.md) · [Venn 1 model card](model-card.md)

**Results and limits:** [Performance and cost](performance.md) · [VennSpace-1](benchmark.md) · [Results and evidence](evidence.md) · [Reproduction](reproduction.md) · **Semantic boundary findings** · [Limitations](limitations.md) · [Comparison video methods](same-memory-video.md) · [Comparison notebooks and renderer](../examples/same_memory_video/README.md)

**Project:** [Policies](policies.md) · [Licenses and attribution](licenses.md) · [Contributing to Venn](../CONTRIBUTING.md)

**Maintainers:** [Maintainer guide](maintainers/maintaining.md) · [Documentation site](maintainers/website.md) · [Release checklist](maintainers/release-checklist.md) · [Publishing the Python package](maintainers/pypi.md) · [Publishing the decision head](maintainers/publishing-the-head.md) · [Development evidence](maintainers/development-evidence.md) · [Comparative evaluation protocol](maintainers/evaluation-plan.md) · [License audit notes](maintainers/license-audit.md)

</details>

**Results and limits / Semantic boundary findings**
<!-- docs-navigation:end -->

Embedding similarity is not reasoning. Venn is therefore not a guardrail,
authorization mechanism or numeric rule engine, and this suite shows where it
fails. API compatibility and probability normalization do not repair these
failures.

The frozen public suite (`venn redteam --output /tmp/venn-redteam
--cache /tmp/venn-cache.sqlite --gpus 1`) contains 32 authored cases across
support routing, sentiment, numeric comparison and explicit refund negation.
It evaluates 288 intent-preserving/paraphrase/visual-format candidates and
64 separate instruction-conflict candidates. All candidates are evaluated:
the finite boundary search chooses the worst outcome per original case.
Paraphrases are authored and labeled, not independently human-certified.
This is a limited fixed suite, not a robustness certification.

| Task | Cases | Baseline correct | Any intent-preserving candidate flips | Worst-case correct | Instruction-conflict flip |
| --- | --- | --- | --- | --- | --- |
| Support routing | 12 | 100% | 8.33% | 91.67% | 8.33% |
| Sentiment | 12 | 83.33% | 16.67% | 83.33% | 75% |
| Numeric comparison | 4 | 50% | 50% | 25% | 50% |
| Explicit negation | 4 | 100% | 25% | 75% | 25% |

These are local reference measurements for the `venn-1` profile, not estimates
of broad population accuracy. The original case, not each correlated candidate,
is the interval unit for Wilson 95% intervals. Small four-case intervals are
especially wide. The runnable suite writes detailed outputs to the directory
you supply, outside the checkout. A flip can correct an initially wrong answer;
it is not synonymous with a new error. Instruction-conflict variants are **not**
counted as meaning-preserving.

Scoring uses public centered-cosine **conditional Choice ordering**. Retain an
explicit abstention/fallback path and validate on your own labeled task and
threat model. Do not treat arbitrary instructions as a general reasoning
capability, untrusted suffixes as harmless, or similar embeddings as proof of
identical meaning. Unknown examples can be worse than this suite.

An independent rerun of the same suite in a fresh isolated environment on four
A100 GPUs with fresh embeddings matched all sixteen aggregate rates within
their reference intervals.

<!-- docs-pager:start -->
---

[← Previous: Reproduction](reproduction.md) · [Next: Limitations →](limitations.md) · [Documentation home](README.md)
<!-- docs-pager:end -->
