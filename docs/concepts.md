# Concepts

<!-- docs-navigation:start -->
[Documentation](README.md) · [Get started](getting-started.md) · **Concepts** · [Venn vs Jev](venn-and-jev.md) · [Examples](../examples/README.md) · [Task recipes](cookbooks.md) · [API](api.md) · [Results](evidence.md)

<details>
<summary>Browse all documentation</summary>

**Project home:** [Home](../README.md)

**Start here:** [Documentation overview](README.md) · [Getting started](getting-started.md) · [Installation](installation.md) · **Concepts** · [Venn vs Jev](venn-and-jev.md)

**Task guides:** [Task recipes](cookbooks.md) · [Outcome stores](outcome-api.md)

**Examples:** [Choose an example](../examples/README.md) · [Hot dog image classifier](../examples/hotdog/README.md) · [Precedent Desk](../examples/precedent_desk/README.md) · [Wine predictor](../examples/wine/README.md) · [Audio classifier](../examples/audio/README.md) · [Screen-only arena](arena-example.md) · [Demos and videos](demos.md)

**Reference:** [API reference](api.md) · [Design rationale](design.md) · [Venn 1 model card](model-card.md)

**Results and limits:** [Performance and cost](performance.md) · [VennSpace-1](benchmark.md) · [Results and evidence](evidence.md) · [Reproduction](reproduction.md) · [Semantic boundary findings](redteam.md) · [Limitations](limitations.md) · [Comparison video methods](same-memory-video.md) · [Comparison notebooks and renderer](../examples/same_memory_video/README.md)

**Project:** [Policies](policies.md) · [Licenses and attribution](licenses.md) · [Contributing to Venn](../CONTRIBUTING.md)

**Maintainers:** [Maintainer guide](maintainers/maintaining.md) · [Documentation site](maintainers/website.md) · [Release checklist](maintainers/release-checklist.md) · [Publishing the Python package](maintainers/pypi.md) · [Publishing the decision head](maintainers/publishing-the-head.md) · [Development evidence](maintainers/development-evidence.md) · [Comparative evaluation protocol](maintainers/evaluation-plan.md) · [License audit notes](maintainers/license-audit.md)

</details>

**Start here / Concepts**
<!-- docs-navigation:end -->

<!-- docs-contents:start -->
<details>
<summary>On this page</summary>

- [Vector Decision Models](#vector-decision-models)
- [Three ways to decide](#three-ways-to-decide)
- [Profiles and compatibility](#profiles-and-compatibility)
- [Changing the reader](#changing-the-reader)
- [Families](#families)
- [What an answer contains](#what-an-answer-contains)

</details>
<!-- docs-contents:end -->

## Vector Decision Models

A Vector Decision Model (VDM) separates reusable embedding state from the
decisions made over it. A **reader** encodes content once into vectors. A
small **decision layer** then answers typed questions against those vectors,
for one item at a time or for whole collections. Venn is an implementation of
this design. Its single-item question types, Choice, Score and Noul, follow the
Decisions API interface established by TypeSafe's Jev; its collection-level
types are Venn extensions.

| Term | Meaning |
| --- | --- |
| Reader | The frozen embedding model. Venn 1 uses EmbeddingGemma 2. |
| Profile | A pinned reader, revision, vector dimension and prompt protocol. Vectors are only comparable within one profile. |
| Head | A small model that turns vectors and a question into decision scores. The generic `venn-1` head adds abstention to cosine scoring. |
| Family | A custom predictor registered with the service: a classifier, numeric predictor or head compiled from an outcome store. |
| Outcome store | Your vector store of past cases and their numerical outcomes, used to decide without any fitted head. A store can also be compiled into a small head that answers the same question without a lookup. |
| Tag | Provenance attached to outcome-store vectors: profile, revision, representation, dimension and field order. |
| Choice | Pick one option from named criteria, with an optional "none of these". |
| Score | A level on an ordered scale, or the numeric prediction from a regression family. |
| Noul | A yes/no question, returned as the probability of yes. |

## Three ways to decide

```text
Content -> pinned frozen reader -> reusable vectors
                                      |
               +----------------------+----------------------+
               |                      |                      |
        Described options       Historical cases       Fitted family
        + generic head          + labels/outcomes      + saved head
               |                      |                      |
        Choice / Score / Noul   answer + case IDs      prediction
```

**Described options:** ask which queue owns a new ticket, supplying descriptions
such as “login and access” and “billing and payments.” Venn compares the ticket
with those descriptions. This route needs no archive of resolved tickets.

**Historical cases:** ask which team had the best outcomes on similar tickets.
An outcome store finds precedents for each team and averages their recorded
outcomes; the answer includes the supporting case IDs. Your application defines
the outcome and writes new results back. A reference index offers a simpler
alternative: return the label of the closest past case.

**Fitted families:** fit a classifier on resolved tickets, give it a name such
as `ticket_team`, and call that saved predictor for new tickets. This replaces
case lookup with a small model. A family can also predict a number, such as
resolution time, or compile an outcome store into a decision head. The embedding
reader stays frozen while the predictor learns from your examples.

For existing tagged vectors, the in-process `OutcomeService` needs neither a
reader nor a generic head. The HTTP service loads the generic head at startup.
See the [model card](model-card.md) for architectures and fitting details.

See [Venn and Jev](venn-and-jev.md) for when to use each approach and the
[examples](../examples/README.md) to run them.

## Profiles and compatibility

The profile is the unit of compatibility. `venn-1` is `google/embeddinggemma-2`
at revision `f9972ac22e696a105c482202646979e33e4b5a1e`, 768 dimensions, with the
Classification text prompt. Images use the same reader; image "thirds" are three
768-dimensional embeddings concatenated into 2304 dimensions and are not
interchangeable with text vectors.

The prompt is part of the representation: Classification and SentenceSimilarity
vectors use separate caches. Families with another prompt or text length inputs
require text requests so Venn can supply those inputs.

Every store, index and family belongs to one profile. The outcome API requires
an explicit representation tag on vector states and stores. Standard Decisions
vector requests use the service's fixed `venn-1` profile without a per-state tag.
A supplied tag is a caller assertion, not proof, so keep your embedding pipeline
pinned. Moving to a different reader
means re-embedding and re-registering; padding, truncating or relabelling
vectors is never valid. The arena example shows how an alternative image reader
can be used for outcome-store geometry under an explicitly `experimental:`
profile; such vectors cannot be mixed with `venn-1` stores or used with the
published generic head. Outcome heads fitted on the new representation remain
supported.

## Changing the reader

The reader and decision layer are separate components. Venn can use a better
or more specialized embedder through the vector-native outcome API: embed your
items externally, register a store with its representation tag, and send new
vectors with the same tag. This path supports other dimensions and can compile
the resulting history into a fitted outcome head. The [portable arena reader](arena-example.md#try-other-hardware-readers-and-levels)
is a working example with a selectable image model.

Custom readers use an `experimental:` profile, an immutable model revision and
their actual vector dimensions. The outcome service consumes those vectors;
your application controls how they are produced. Store adapters are a separate
extension point for choosing the database.

The published `venn-1` generic head and the standard `DecisionService` text
profile are tied to EmbeddingGemma 2. Its `embed` hook can replace the loading
or execution backend for that reader; it does not register an unrelated model.
Adopting another reader for this path requires a corresponding profile and
head. Changing readers means re-embedding the store and refitting task heads;
the application's outcome request shape stays the same.

## Families

A family is a registered custom predictor. Choice families train on every
example by default; an explicit split reserves selection and report rows. A
residual Choice family chooses weight decay and epochs by cross-validation
(grouped by `group_id`), fits a temperature on out-of-fold predictions, and records
cross-validated accuracy and expected calibration error, plus report-slice
metrics when you pass a split. Five examples per label is the minimum
to fit; it is not a calibration guarantee. A family is marked calibrated only
when a caller-supplied report slice is large enough and its bootstrap calibration-error
bound is low enough. Families are stored under `VENN_HOME` and reloaded on
startup. See the [model card](model-card.md#local-family-fitting) for the
defaults and the [cookbook](cookbooks.md#fit-and-reload-a-local-family) for a
runnable example.

A linear family (`form="linear"`) replaces the residual head with a multinomial
logistic regression on the state vectors. Its penalty is chosen by the same
grouped cross-validation, on log loss. It has no abstention, so its `none`
probability is always 0, and it can name only the labels it was registered
with. For plain classification it is usually the stronger choice. On CREMA-D
speech emotion, 70 training actors and 21 held out, it scored 0.565 accuracy,
matching a scikit-learn logistic-regression probe at 0.567, while the residual
head scored 0.383. Use the residual head when "none of these" matters or when
options change between questions.

A numeric family (`objective="regress"`) fits ridge or kernel ridge from vectors
to a number: a critic's score, a resolution time, a return. It is
registered from examples with a numeric field or from a store's outcomes,
chooses its penalty by cross-validation inside its train slice, and answers a
Score question that names it with a score, an 80% prediction interval and a
`calibrated` flag that turns on only when a large enough report slice confirms
the interval's coverage. See the [numeric family cookbook](cookbooks.md#fit-a-numeric-family).

## What an answer contains

Generic Decisions answers carry the decision, probabilities and a `venn`
block with abstention and calibration metadata. Registered numeric families
return a score, interval and calibration flag. Conditional Choice
probabilities are conditioned on the listed options unless you include a `null`
criterion for "none"; Venn-specific metadata may be dropped by standard
Decisions clients, so include a null criterion when abstention matters. Outcome
answers carry per-option evidence (mean outcome, count, spread, mean distance
and case IDs) instead of probabilities. Nothing in an answer authorizes an
action by itself; see the [limitations](limitations.md).

<!-- docs-pager:start -->
---

[← Previous: Installation](installation.md) · [Next: Venn vs Jev →](venn-and-jev.md) · [Documentation home](README.md)
<!-- docs-pager:end -->
