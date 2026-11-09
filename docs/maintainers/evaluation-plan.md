# Comparative evaluation protocol

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

**Maintainers:** [Maintainer guide](maintaining.md) · [Documentation site](website.md) · [Release checklist](release-checklist.md) · [Publishing the Python package](pypi.md) · [Publishing the decision head](publishing-the-head.md) · [Development evidence](development-evidence.md) · **Comparative evaluation protocol** · [License audit notes](license-audit.md)

</details>

**Maintainers / Comparative evaluation protocol**
<!-- docs-navigation:end -->

<!-- docs-contents:start -->
<details>
<summary>On this page</summary>

- [Question](#question)
- [Matched comparisons](#matched-comparisons)
- [Splits and uncertainty](#splits-and-uncertainty)
- [Required release artifacts](#required-release-artifacts)

</details>
<!-- docs-contents:end -->

Status: proposed experiments, not measured results or a preregistration of
completed work. Freeze a versioned protocol and all split identities before
running the comparisons below. VennSpace-1 remains unchanged.

## Question

What does a learned Venn head or registered family add to decisions made from
the same frozen embeddings, and when does reuse reduce total workload cost?
Keep the contribution of the reader distinct from that of the decision layer.

## Matched comparisons

| Workload | Required controls | Measurements |
| --- | --- | --- |
| Routing with no-match examples | Raw/centered cosine, nearest reference, centroid, linear classifier, learned abstention and registered family | Accuracy, macro-F1, risk versus coverage, no-match precision/recall, Brier score and calibration intervals |
| Ordered scoring | Cosine ordering, linear/ridge ordinal baseline, registered family | Spearman rho, absolute error, ordering violations and calibration where meaningful |
| Repeated decisions over cached rows | Same batches/options on headless and trained-head paths | Cold/warm latency, p50/p95, memory, rows/s, serialization and full consumption |
| End-to-end decision systems | Relevant semantic router/few-shot classifier and optional generative decision baseline | Quality and abstention at comparable coverage, encoder cost, fitting cost, repeated-query amortization and cost per accepted correct decision |

For Jev, evaluate multiple questions in one request where supported. For every
baseline, use its relevant caching/batching capabilities. Report fresh-content
and already-embedded conditions separately, with cold/warm latency, throughput,
peak memory and actual compute/provider cost. Include vector-store I/O and
serialization. Follow the [cost-accounting boundaries](../performance.md).

Use identical official test rows and training budgets where methods permit.
For methods that tune the encoder, report that as a separate condition rather
than attributing the gain to a head. Specify versions, prompts, label budgets,
seeds and parameter choices before evaluation. Compare within method scope;
MTEB and ColBERT are useful evaluation/architecture references, not interchangeable
routing baselines.

## Splits and uncertainty

Separate fitting, early stopping, probability/threshold selection and final
reporting. Account for overlap with the generic head's training datasets;
Banking77 and CLINC are not automatically unseen tasks. Include new domains
and genuine no-match examples. Group duplicates and related variants before
splitting. Choose the resampling unit at the independent example/task level,
not at the paraphrase level. Report all chosen tasks and failed runs.

Ablate centered options, learned abstention, family residuals and calibration.
For K=2 record the runtime's fallback decision. For ordered scoring separately
validate label order; a nominal classifier is not an ordinal model by default.

## Required release artifacts

Each run must identify source commit, protocol hash, reader and head hashes,
source revisions, split hashes, runtime versions, hardware, seeds, commands,
metrics and uncertainty, resource consumption and exclusions. Make an aggregate
reference and comparison command public. Document any unavailable training
recipe or checkpoint explicitly. No success threshold should be chosen after
examining final test outcomes.

<!-- docs-pager:start -->
---

[← Previous: Development evidence](development-evidence.md) · [Next: License audit notes →](license-audit.md) · [Documentation home](../README.md)
<!-- docs-pager:end -->
