# Development evidence

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

**Maintainers:** [Maintainer guide](maintaining.md) · [Documentation site](website.md) · [Release checklist](release-checklist.md) · [Publishing the Python package](pypi.md) · [Publishing the decision head](publishing-the-head.md) · **Development evidence** · [Comparative evaluation protocol](evaluation-plan.md) · [License audit notes](license-audit.md)

</details>

**Maintainers / Development evidence**
<!-- docs-navigation:end -->

<!-- docs-contents:start -->
<details>
<summary>On this page</summary>

- [Application validation](#application-validation)
- [Episodic-control protocol](#episodic-control-protocol)
- [Timing measurements](#timing-measurements)

</details>
<!-- docs-contents:end -->

These are internal development measurements of the outcome API from November
2026, kept for maintainers. They are not public benchmarks and were not
independently reproduced. Public contract tests exercise the API and adapters;
the original game harness, checkpoints and raw traces behind these numbers
are not distributed. The public [arena example](../arena-example.md) now
provides a runnable implementation of the workflow. None of these rows is part of VennSpace-1.

## Application validation

The November 2026 development run exercised every Doom embedding, lookup and
action choice through the Python `OutcomeService`, with an app-owned exact
FAISS GPU index, in the ViZDoom basic and defend_the_center scenarios.

| Claim | Recorded result | Scope / status |
| --- | --- | --- |
| Shadow action agreement | basic 99.990%; defend_the_center 99.994% | 50,000 decisions per scenario; the direct game harness's actions drove the game |
| Exact FAISS immediate reads | 1,000/1,000 | Passed the read-after-write check; not a guarantee for other stores |
| API-native learning | basic success 100%; defend_the_center mean kills 6.18 | One seed, 431,173 decisions per scenario; 50 frozen episodes each; inside the pass criteria fixed before the run |
| Store wipe | All 20 random-agent episode records matched per scenario | Tests dependence on memory, not neural weight learning |
| Warm in-process full loop | 94.67 decisions/s vs eager direct game harness 76.97 | One A100, batch 128; passed the fixed >=81 decisions/s and >=90% relative criteria |
| HTTP + LanceDB full loop | 8.01 decisions/s; recall@11 99.36% | Report-only; different search/policy semantics; includes visibility probes |
| Banking77 outcome routing | 35.815% cumulative; 63.8% last 1,000 | 20,000 training-split interactions; below cosine 99.790% and fully labeled nearest reference 99.075%; unequal feedback |

Final validation checked checkpoint keys/returns, snapshot hashes and neighbor
traces with no future entry IDs. All pass criteria fixed before the run were
met; the HTTP and routing results are report-only.

The learning rows are one seed per scenario, not a five-seed API reproduction
or identical learning trajectories, and the model weights remained frozen.
The warm full-loop figure used opt-in exact elementwise fusion. None of these
numbers is a deployment SLA or a publicly reproduced benchmark.

A report-only 20,000-interaction Banking77 training stream reached 35.815%
cumulative outcome-routing accuracy (63.8% in the last 1,000), below fixed
zero-shot centered cosine at 99.790% and a fully labeled online nearest
reference at 99.075%. The outcome policy received only chosen-queue binary
feedback; the supervised reference received the true label after predicting.
Repeated training inputs and stronger baseline feedback prevent treating this
as held-out generalization or a routing advantage.

## Episodic-control protocol

The Doom application used model-free episodic control: 128 games shared one
exact FAISS GPU index per action, with k=11 and `min_support=1`. At episode
completion, the app computed discounted returns with gamma=0.99 and stored the
maximum observed return for each float16 state-key/action pair. Explicit
`exact_match=true` and `unsupported_value=0` preserved that application's
exact-state and unseen-action conventions, rather than changing API defaults.
Exploration decayed from 1.0 to 0.05 over the first 30% of the decision budget;
eight games were greedy throughout.

Each API-native run used 431,173 decisions from seed 0, followed by 50 frozen
evaluation episodes with no further store learning. Memory wipe evaluations
tested dependence on stored experience. This is store-based learning with a
frozen image reader, not neural policy training, a general
reinforcement-learning algorithm. The historical harness is not distributed;
use the [public arena example](../arena-example.md) for new runs.

## Timing measurements

Development measurements on an A100-SXM4-80GB distinguish the in-process
outcome API from a standard HTTP/store stack:

| Path | Warm full-loop decisions/s | Boundary |
| --- | --- | --- |
| Independent eager direct game harness | 76.97 | Frames, reader, exact lookup, policy, game stepping and terminal writes |
| Python outcome API + exact FAISS GPU | 94.67 | Same boundary; opt-in exact elementwise normalization fusion |
| Loopback HTTP + LanceDB IVF_FLAT | 8.01 | Binary uint8 requests, JSON response consumption, game loop and app-owned store writes; includes visibility probes |

Each timing restores the same basic seed-0 checkpoint and uses 128 games,
two warm-up batches and five timed batches. Setup, model loading, index
construction and initial compilation are excluded. The HTTP run's GPU had
no competing process before or after measurement. Other GPUs performed
separate work; this was not a dedicated-host or production-load study.

These are median batch rates over five timed batches (640 decisions), not
request-latency percentiles or sustained-load estimates. The API rate was
1.230 times the direct game harness rate, passing the fixed >=90% relative and
>=81 decisions/s absolute criteria. Normalization fusion is opt-in; this is not a same-optimization
comparison of API overhead. No GPU-kernel-only rate is substituted for the full
game loop.

HTTP/LanceDB recall@11 against exact search was 99.36% across 384 initial
query-option pairs. All 21 observed terminal-write transactions were visible
to fresh reads: write-start-to-visible p50/max 60.65/72.75 ms, or
commit-to-visible 23.92/26.64 ms, including client/query overhead. The backend
uses ordinary k-neighbor averaging, not FAISS's optional exact-key utility
override, and the HTTP server owns exploration RNG. Its rate, 10.41% of the
direct game harness rate, is therefore report-only, not an
identical-trajectory comparison or learning-equivalence result.

The HTTP result, below half the direct game harness rate, motivates a separate
proposal for upsert/delete
on a registered index rebuildable from the app store; that addition has not
been implemented. These are local measurements, not public reproduction, a
latency SLA, or a speed/cost comparison with Jev (TypeSafe's hosted decision
model, whose Decisions API request format Venn accepts).

See also the [outcome API](../outcome-api.md), the
[model card](../model-card.md#training-and-availability) and the
[performance guide](../performance.md).

<!-- docs-pager:start -->
---

[← Previous: Publishing the decision head](publishing-the-head.md) · [Next: Comparative evaluation protocol →](evaluation-plan.md) · [Documentation home](../README.md)
<!-- docs-pager:end -->
