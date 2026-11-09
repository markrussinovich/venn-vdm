# Design rationale

<!-- docs-navigation:start -->
[Documentation](README.md) · [Get started](getting-started.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md) · [Examples](../examples/README.md) · [Task recipes](cookbooks.md) · [API](api.md) · [Results](evidence.md)

<details>
<summary>Browse all documentation</summary>

**Project home:** [Home](../README.md)

**Start here:** [Documentation overview](README.md) · [Getting started](getting-started.md) · [Installation](installation.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md)

**Task guides:** [Task recipes](cookbooks.md) · [Outcome stores](outcome-api.md)

**Examples:** [Choose an example](../examples/README.md) · [Hot dog image classifier](../examples/hotdog/README.md) · [Precedent Desk](../examples/precedent_desk/README.md) · [Wine predictor](../examples/wine/README.md) · [Audio classifier](../examples/audio/README.md) · [Screen-only arena](arena-example.md) · [Demos and videos](demos.md)

**Reference:** [API reference](api.md) · **Design rationale** · [Venn 1 model card](model-card.md)

**Results and limits:** [Performance and cost](performance.md) · [VennSpace-1](benchmark.md) · [Results and evidence](evidence.md) · [Reproduction](reproduction.md) · [Semantic boundary findings](redteam.md) · [Limitations](limitations.md) · [Comparison video methods](same-memory-video.md) · [Comparison notebooks and renderer](../examples/same_memory_video/README.md)

**Project:** [Policies](policies.md) · [Licenses and attribution](licenses.md) · [Contributing to Venn](../CONTRIBUTING.md)

**Maintainers:** [Maintainer guide](maintainers/maintaining.md) · [Documentation site](maintainers/website.md) · [Release checklist](maintainers/release-checklist.md) · [Publishing the Python package](maintainers/pypi.md) · [Publishing the decision head](maintainers/publishing-the-head.md) · [Development evidence](maintainers/development-evidence.md) · [Comparative evaluation protocol](maintainers/evaluation-plan.md) · [License audit notes](maintainers/license-audit.md)

</details>

**Reference / Design rationale**
<!-- docs-navigation:end -->

<!-- docs-contents:start -->
<details>
<summary>On this page</summary>

- [The problem](#the-problem)
- [The idea](#the-idea)
- [What is learned, and where](#what-is-learned-and-where)
- [Space-level methods](#space-level-methods)
- [What this does not give you](#what-this-does-not-give-you)
- [Related work](#related-work)

</details>
<!-- docs-contents:end -->

This note explains why Venn is built the way it is. The [concepts](concepts.md)
page defines the terms; the [benchmark](benchmark.md), [semantic boundary
suite](redteam.md) and [evidence ledger](evidence.md) carry the numbers.

## The problem

Applications that already embed their content keep asking small questions
about it: which team owns this ticket, does this passage belong with these
others, has the mix of incoming requests changed, which of these indexed
records now match a new policy. Each question is cheap to state and expensive
to answer with a generative model, and the answer is needed for every item,
not once.

## The idea

A Vector Decision Model separates the embedding state from the decisions
made over it. A frozen reader encodes content into reusable vectors. A small decision layer
then answers typed questions against that state: Choice, Score and Noul
(yes/no) for a single item, and nearest reference, set membership, population
shift and streamed index decisions for collections. Because the state is
reused, a compatible stored representation can serve new questions without another
reader pass. Linear heads score collections with a matrix multiplication;
outcome lookup and kernel heads use their respective search and prediction paths.

Venn accepts the Decisions API request format used by TypeSafe's Jev for the
[tested Decisions clients](https://github.com/markrussinovich/venn-vdm/blob/main/tests/clients/README.md), and adds the space-level
question types as extensions those clients can ignore.

## What is learned, and where

- **The generic head** adds one learned "none of these" logit to centered
  cosine scores and applies fitted temperatures. It is the only trained
  component in the standard path and is 3 MB.
- **Families** are custom predictors fitted and persisted by the service:
  residual Choice heads, linear classifiers, numeric ridge or kernel heads,
  and heads compiled from outcome stores. The reader stays frozen.
- **Outcome stores** learn nothing in weights. The application records a
  vector, the option it chose and a numerical outcome; later decisions average
  the outcomes of similar past cases per option. Learning is the growth of the
  store, which the application owns.

Keeping these three separate makes it clear what changes when behavior
changes, and lets the headless outcome path run with no trained head at all.

## Space-level methods

Nearest reference returns the label of the most similar exemplar. Small-set
odd-one-out ranks leave-one-out centroid distances. Large-set membership uses
fifth-neighbor distances rather than assuming one centroid. Population shift
uses a permutation-calibrated RBF MMD test; its p-value is not a probability
that drift occurred. Indexed decisions prepare the question once and stream
every row rather than returning a top-k.

## What this does not give you

Embeddings are not propositional. Paraphrases, numbers, dates and negation can
cross a decision boundary while the intended label stays put; the semantic
boundary suite shows this. Conditional probabilities are not universally
calibrated. Embeddings can be inverted, so a vector is as sensitive as its
text. Venn is therefore a routing and triage layer with an explicit abstention
path, not a guardrail or an authorization mechanism.

## Related work

[SetFit](https://github.com/huggingface/setfit) fits few-shot classifiers on
sentence embeddings. [Semantic Router](https://github.com/aurelio-labs/semantic-router)
routes by embedding similarity. [ColBERT](https://github.com/stanford-futuredata/ColBERT)
reuses token-level representations with a late interaction step, and
[MTEB](https://github.com/embeddings-benchmark/mteb) benchmarks embedding
models across tasks. Venn's distinct choices are the typed Decisions interface,
the explicit abstention head, the space-level question types and the
application-owned outcome store. Whether the learned components beat simple
baselines on a given task is an empirical question; the
[evaluation plan](maintainers/evaluation-plan.md) lays out the matched
comparisons that would settle it.

<!-- docs-pager:start -->
---

[← Previous: API reference](api.md) · [Next: Venn 1 model card →](model-card.md) · [Documentation home](README.md)
<!-- docs-pager:end -->
