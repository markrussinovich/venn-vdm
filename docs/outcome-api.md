# Decisions from an app-owned outcome store

<!-- docs-navigation:start -->
[Documentation](README.md) · [Get started](getting-started.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md) · [Examples](../examples/README.md) · [Task recipes](cookbooks.md) · [Adapters](adapters.md) · [API](api.md) · [Results](evidence.md)

<details>
<summary>Browse all documentation</summary>

**Project home:** [Home](../README.md)

**Start here:** [Documentation overview](README.md) · [Getting started](getting-started.md) · [Installation](installation.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md)

**Task guides:** [Task recipes](cookbooks.md) · [Store adapters](adapters.md) · **Outcome stores**

**Examples:** [Choose an example](../examples/README.md) · [Hot dog image classifier](../examples/hotdog/README.md) · [Precedent Desk](../examples/precedent_desk/README.md) · [Wine predictor](../examples/wine/README.md) · [Audio classifier](../examples/audio/README.md) · [Screen-only arena](arena-example.md) · [Demos and videos](demos.md)

**Reference:** [API reference](api.md) · [Design rationale](design.md) · [Venn 1 model card](model-card.md)

**Results and limits:** [Performance and cost](performance.md) · [VennSpace-1](benchmark.md) · [Results and evidence](evidence.md) · [Reproduction](reproduction.md) · [Semantic boundary findings](redteam.md) · [Limitations](limitations.md) · [Comparison video methods](same-memory-video.md) · [Comparison notebooks and renderer](../examples/same_memory_video/README.md)

**Project:** [Policies](policies.md) · [Licenses and attribution](licenses.md) · [Contributing to Venn](../CONTRIBUTING.md)

**Maintainers:** [Maintainer guide](maintainers/maintaining.md) · [Documentation site](maintainers/website.md) · [Release checklist](maintainers/release-checklist.md) · [Publishing the Python package](maintainers/pypi.md) · [Publishing the decision head](maintainers/publishing-the-head.md) · [Development evidence](maintainers/development-evidence.md) · [Comparative evaluation protocol](maintainers/evaluation-plan.md) · [License audit notes](maintainers/license-audit.md)

</details>

**Task guides / Outcome stores**
<!-- docs-navigation:end -->

<!-- docs-contents:start -->
<details>
<summary>On this page</summary>

- [Run the vector example](#run-the-vector-example)
- [Tags and image batches](#tags-and-image-batches)
- [HTTP setup](#http-setup)
- [Freshness and filtering](#freshness-and-filtering)
- [Learning without weight updates](#learning-without-weight-updates)
- [Compile the store into a head](#compile-the-store-into-a-head)

</details>
<!-- docs-contents:end -->

The experimental Python `OutcomeService` chooses among options using numerical
outcomes recorded in an app-owned vector store. It runs without fitted Venn
heads, and it can compile a store into a small head that answers the same
question without a lookup. For already tagged vectors, it needs no reader or
GPU; raw image batches require the pinned reader on CUDA. The application owns
and updates the store.

Contract tests validate the interface, filtering and recorded-outcome
calculations. Application decision quality and latency need evaluation on your
own data. The existing Decisions routes remain available.
The [arena application](arena-example.md) provides a runnable screen-only
client, portable memory and instructions for other hardware, readers and levels.

## Run the vector example

After installing Venn, run `python examples/outcome_store.py` from the checkout.
It uses synthetic vectors on CPU and prints `support`. The code below shows the
same registration and query sequence.

```python
import faiss
import numpy as np
from venn import OutcomeService
from venn.adapters import FaissOutcomeAdapter
from venn.profiles import TEXT_TAG

# The app owns these mutable objects and serializes writes against decisions.
index = faiss.IndexFlatL2(768)
ids, outcomes = [], []
parts = {"support": {"index": index, "ids": ids, "outcomes": outcomes}}
service = OutcomeService()
store = service.register_store(FaissOutcomeAdapter(parts, tag=TEXT_TAG))

# vector must have been produced with TEXT_TAG's pinned reader and prompt.
vector = np.zeros(768, dtype=np.float32)
vector[0] = 1  # Synthetic illustration, not a real text embedding.
index.add(vector[None])
ids.append("resolved-1")
outcomes.append(1.0)
response = service.decide_batch(
    [{"vector": vector, "tag": TEXT_TAG.to_dict()}],
    {"queue": {"type": "nearest_outcome", "store": store,
               "options": ["support", "billing"], "k": 11,
               "min_support": 1, "goal": "maximize"}},
)
assert response["results"][0]["answers"]["queue"]["answer"] == "support"
```

For each option the API returns its nearest cases' mean outcome, count,
population standard deviation (`spread`), mean distance and evidence IDs.
`goal` is `maximize` or `minimize`. An option below `min_support` is
unsupported; if none qualifies, `answer` is `none`. Empty evidence has mean,
spread and distance `null`, not invented zeros. Outcomes are utilities, not
probabilities: `calibrated` is false and confidence/probabilities are null.
Up to 128 states and 32 named questions share a call.

`action` additionally applies the app-supplied per-state `epsilon` exploration
probability, breaking utility ties randomly. The in-process call accepts a
NumPy `rng` for reproducibility. The optional `unsupported_value` is an explicit
app utility prior for unseen options; it never becomes a measured mean.
`exploratory` identifies exploration or an unsupported chosen action.
The default `answer` still requires measured support.

An opt-in `exact_match: true` FAISS query uses the app's `exact_positions`
mapping (SHA256 of a float16 key to a row) to prefer that case over averaging
nearby cases. This reproduces episodic control's exact-state convention; it is
not the default k-neighbor estimator. The app supplies already-merged maximum
returns and updates its live outcomes list. It must use float16 query keys for
this convention. Other backends reject this option rather than approximate it.

## Tags and image batches

Every new-path vector carries `profile`, pinned `revision`, `representation`,
`dimensions` and ordered `fields`. All fields must equal the store's tag;
matching dimensions alone is insufficient. Raw-input decisions on the standard
routes also return `vector_tag`. Legacy cached vectors without provenance are marked
`untagged`, not silently certified as text or images.

Outcome-only tagged-vector geometry also accepts explicitly named
`experimental:` profiles with immutable 40-character reader revisions.
This permits an app to use another image-capable reader without claiming
compatibility with `venn-1`. The native raw-image path and trained-head
serving remain pinned; alternate readers do not become aliases or defaults.

The in-process image path accepts an `N x H x W x 3` NumPy **uint8** array.
`representation` is `image-70`, `image-280` or `image-thirds-70`. Thirds are
embedded in left/center/right order and concatenated then normalized (2304
dimensions); whole images have 768 dimensions. The pinned full reader runs
frozen bf16 on CUDA. A pinned host buffer, GPU crop/resize/normalization and
eager forward avoid PNG/JPEG encoding and CPU image preprocessing. Download
the pinned reader separately; this path never downloads implicitly.

For warm GPU throughput, construct `ImageBatchReader(compile_norms=True)` and
pass it as `OutcomeService(image_reader=reader)`. This opt-in compiles only
elementwise normalization kernels; reductions, precision, model weights and
representations stay unchanged. Initial calls compile kernels, so warm up the
intended batch shapes before measuring throughput. A supplied reader is
optimized in place; use a separate reader for an independent eager baseline.

```python
response = service.decide_batch(
    screens_uint8, questions, representation="image-thirds-70",
    rng=policy_rng, epsilon=exploration_probabilities,
)
# Append response["results"][i]["vector"] to the app's correctly tagged store.
```

## HTTP setup

The current HTTP routes are hosted by the combined `DecisionService`, which
loads the venn-1 head at startup. This requirement applies even if your
requests only use outcome-store decisions. There is currently no standalone
headless HTTP startup command. For operation without a head, use the Python
`OutcomeService` directly as in the example above.

For combined serving, follow [getting started](getting-started.md), then
construct `DecisionService(outcome_service=service, model_dir=...)` in your
server startup code. Attach the configured service with `create_app(service=...)`
and register adapters with `register_outcome_store`.
Clients refer only to returned handles, not arbitrary database credentials.
`POST /decide-batch` accepts JSON tagged states or an `application/octet-stream`
uint8 batch. The latter requires `X-Venn-Experimental: media` and an
`X-Venn-Batch` JSON header containing `shape`, `questions` and `representation`.
The Python `Client.decide_batch` handles this wire format without image encoding.
Limits are 128 images, 4M pixels per image, 64MB raw bytes; JSON retains the
existing 8MB limit. Single tagged states also work with the `nearest_outcome`
extension on `/api/alpha/decisions`.

## Freshness and filtering

Each adapter declares `read_after_write`; none is copied into a registered
snapshot. `filters` are finite scalar equality predicates. Both option and
metadata constraints are applied **before** top-k:

| Adapter | Visibility and filter path |
|---|---|
| `FaissOutcomeAdapter` | Live app-supplied squared-L2 index per option; app serializes updates. Metadata selects an exact eligible subset before search (CPU subset when filtering). GPU indexes require a separately installed FAISS GPU build. |
| `QdrantOutcomeAdapter` | Fresh indexed queries by default after the app completes `wait=True` upserts; `exact=True` opts into exhaustive search. Requires an unnamed Euclidean collection matching the tag. Reported distances are squared. |
| `LanceDBOutcomeAdapter` | Fresh table checkout and a new prefiltered L2 query after completed app writes. |
| `PgVectorOutcomeAdapter` | Parameterized option/metadata predicates precede distance ordering and limit. Visibility follows the app cursor's transaction isolation. |
| `WeaviateOutcomeAdapter` | Backend option/metadata filters on each query; replication consistency determines visibility. Distances retain backend semantics. |
| `AzureSearchOutcomeAdapter` | Fresh vector-only queries against the configured index with `preFilter` option/metadata predicates; `exhaustive=True` opts into brute-force search. Indexing is eventually consistent; completed uploads do not guarantee search visibility. |

The batch-reading adapters used for sweeps are separate from these outcome
adapters. Qdrant and LanceDB contracts exercise actual local stores;
PostgreSQL, Weaviate and Azure AI Search require separately provisioned services
for live integration validation. Azure adapter tests exercise the real Python
SDK and serialized HTTP requests against a local test transport, not a hosted index.

The [store adapter guide](adapters.md) gives setup and usage examples for every
backend, including [Microsoft Azure AI Search](adapters.md#microsoft-azure-ai-search),
with schema requirements, field mappings and write-visibility caveats.

## Learning without weight updates

The application closes the feedback loop:

1. Call `decide_batch` with tagged states and named outcome questions.
2. Execute the returned `action`, retaining its vector, option and evidence.
3. Observe the application outcome and write it with that vector and option.
4. Complete the write under the adapter's visibility contract before querying again.

Venn does not infer rewards, write feedback or update reader weights.
For routing, an outcome could be a measured success or resolution time; define
`goal` accordingly and evaluate feedback bias. A policy that observes only its
chosen option cannot be compared as if it received every correct label.

The [arena example](arena-example.md) stores discounted returns per screen
vector and action and uses `exact_match` and `unsupported_value` to reproduce
an episodic-control convention. This is store-based learning with a frozen
image reader. The section below is the one place the outcome path trains
anything, and what it trains is a head of a few tens of thousands of parameters.

**Optional logged-policy diagnostics.** For events with measured rewards and
recorded action propensities, `venn.offpolicy.estimate_logged_policy` reports IPS,
self-normalized IPS, effective sample size, maximum weight and observed rewards:

```python
from venn.offpolicy import estimate_logged_policy

report = estimate_logged_policy(
    rewards=[1., 0., 1., 0.],
    logging_probability=[.5, .5, .5, .5],
    target_probability=[1., 0., 1., 0.],
)
```

Every position must refer to the same logged event in all three arrays. The
probabilities are those assigned to the **action actually taken**, conditional on
its logged context, by the behavior and target policies respectively. Retain the
context, action, policy version, action propensity and eventual reward when
logging; epsilon alone is not the propensity, and `exploratory` is not a
probability. For example, with two actions, epsilon 0.1 and a unique greedy
winner, the behavior probabilities are 0.95 and 0.05, not 0.1. Ties and fallback
policies change that calculation. The current service does not return action
propensities, so existing outcomes without these records are insufficient.

The helper does not infer missing rewards, clip weights, fit a reward model or
provide confidence intervals. Small effective sample size warns of fragile
estimates. Valid propensities, conditional exchangeability and policy overlap
are external assumptions that these arrays cannot verify; results do not certify
a causal improvement or justify changing an experimental gate.

## Compile the store into a head

`register_family` trains a small head from every case the store exports and
registers it under a name. A `nearest_outcome` question that names that
`family` is then answered by the head alone: no lookup, no evidence, the same
answer shape otherwise.

```python
service.register_family("policy", from_store=store, options=["support", "billing"],
                        objective="awr", k=11, exact_match=False, unsupported_value=0)

response = service.decide_batch(
    [{"vector": vector, "tag": TEXT_TAG.to_dict()}],
    {"queue": {"type": "nearest_outcome", "family": "policy"}},
)
```

The head is a rank-16 bilinear form over the state vector and one vector per
option, plus a bias. `option_vectors` default to one-hot vectors; passing text
embeddings of the option names from the same profile gives the head an option
side it can generalize over. Two objectives are available. `bc` imitates the
option the store itself returns at each stored state under the given `k`,
`exact_match`, `unsupported_value` and `goal`, so the head is a compiled copy of
the store's policy. `awr` imitates the option that was actually taken, weighting
each case by `exp(sign * (outcome - baseline) / tau)`, where `sign` is +1 for
`maximize` and -1 for `minimize`, with the baseline the mean
outcome of the state's nearest cases across all options, `tau` the spread of
training outcomes, and weights clipped at 20, so cases that turned out well
count for more. The store must hold at least 320 cases (five 64-case chunks);
smaller stores are rejected. Within each option, cases are cut into chunks of
64 in insertion order, and whole chunks are assigned at random (seeded) to
training, early stopping and a held-out report in proportions of about
60/20/20, so neighbouring cases stay in the same slice. If small per-option
partitions round away a slice, their intact chunks are assigned together to
keep all three slices nonempty. Fitting rejects an empty required slice.
The summary returns the parameter count,
case count, fitting losses and held-out agreement.

The adapter must implement `export()`, which `FaissOutcomeAdapter` does.
`save_family` writes the head, options, option vectors, tag and report to one
file and `load_family` registers it again in a fresh service; `warm_start` names
a registered family to continue from when the store has grown. Answers from a
family carry `family`, `objective` and the per-option `head` scores, with
`options` null because no evidence was read; `epsilon` exploration still applies
and is flagged in `exploratory`. A family answers only states in its store's
profile and only under the goal it was registered with.

In the arena, the head compiled from a 380,000-case memory with the `awr`
objective scored 8.4 kills per game served alone, against 6.2 for the memory it
came from, over 250 scored games across five independent memories. The
[arena page](arena-example.md#compile-the-memory-into-a-head) has the command
and the measured table.

<!-- docs-pager:start -->
---

[← Previous: Store adapters](adapters.md) · [Next: Choose an example →](../examples/README.md) · [Documentation home](README.md)
<!-- docs-pager:end -->
