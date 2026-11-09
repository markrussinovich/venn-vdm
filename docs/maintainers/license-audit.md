# License audit notes

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

**Maintainers:** [Maintainer guide](maintaining.md) · [Documentation site](website.md) · [Release checklist](release-checklist.md) · [Publishing the Python package](pypi.md) · [Publishing the decision head](publishing-the-head.md) · [Development evidence](development-evidence.md) · [Comparative evaluation protocol](evaluation-plan.md) · **License audit notes**

</details>

**Maintainers / License audit notes**
<!-- docs-navigation:end -->

<!-- docs-contents:start -->
<details>
<summary>On this page</summary>

- [Head training sources](#head-training-sources)
- [Reasoning](#reasoning)
- [Other corpora](#other-corpora)

</details>
<!-- docs-contents:end -->

These are maintainer working notes supporting the publication decision for the
venn-1 head. They are not legal advice. User-facing attribution is in
[licenses](../licenses.md).

## Head training sources

**Decision, November 7, 2026:** the project owner confirmed that the published
`venn-1` head can be redistributed. The public model card and license page now
record that permission. The source review below remains as historical context;
it is not an unresolved release blocker.

The venn-1 generic head was trained on CLINC, MASSIVE, Banking77, DBpedia,
Yahoo Answers and AG News. The current source cards declare:

| Dataset | Declared license | Notes |
| --- | --- | --- |
| CLINC | CC BY 3.0 | |
| MASSIVE | CC BY 4.0 | |
| Banking77 | CC BY 4.0 | |
| DBpedia | CC BY-SA 3.0 | |
| Yahoo Answers | Unknown | No license metadata on the source card |
| AG News | Unknown | The card describes research/noncommercial use; its formal licensing section is unspecified |

Data revisions and license-card revisions are pinned separately. The converted
parquet commits used for training omit source cards, so these card findings do
not by themselves establish the permissions in force at training time, and they
do not clear any derived head.

## Reasoning

Missing dataset metadata is not an established prohibition on distributing
Venn heads, and dataset licenses do not automatically become weight licenses.
This audit has found no explicit ban on distributing these heads. Retraining is
not required solely because a card omits a license; assess the actual
applicable terms before a weight publication decision. The AG News
research/noncommercial wording is a specific training/use question to assess,
not proof of a blanket derived-weight ban.

The source-only geometric benchmark needs no Venn head weights, so its
publication does not depend on this review.

## Other corpora

SLURP text is CC BY 4.0, but its audio is CC BY-NC 4.0 according to upstream.
ESC-50 is noncommercial. Other media and adapter corpora have individual terms.
Their corpora, embeddings and fitted weights are excluded from the public
release. Yelp's custom terms have not been cleared for distributing the ordinal
evaluation head.

<!-- docs-pager:start -->
---

[← Previous: Comparative evaluation protocol](evaluation-plan.md) · [Documentation home](../README.md)
<!-- docs-pager:end -->
