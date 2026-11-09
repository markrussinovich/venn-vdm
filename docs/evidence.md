# Results and evidence

<!-- docs-navigation:start -->
[Documentation](README.md) · [Get started](getting-started.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md) · [Examples](../examples/README.md) · [Task recipes](cookbooks.md) · [API](api.md) · **Results**

<details>
<summary>Browse all documentation</summary>

**Project home:** [Home](../README.md)

**Start here:** [Documentation overview](README.md) · [Getting started](getting-started.md) · [Installation](installation.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md)

**Task guides:** [Task recipes](cookbooks.md) · [Outcome stores](outcome-api.md)

**Examples:** [Choose an example](../examples/README.md) · [Hot dog image classifier](../examples/hotdog/README.md) · [Precedent Desk](../examples/precedent_desk/README.md) · [Wine predictor](../examples/wine/README.md) · [Audio classifier](../examples/audio/README.md) · [Screen-only arena](arena-example.md) · [Demos and videos](demos.md)

**Reference:** [API reference](api.md) · [Design rationale](design.md) · [Venn 1 model card](model-card.md)

**Results and limits:** [Performance and cost](performance.md) · [VennSpace-1](benchmark.md) · **Results and evidence** · [Reproduction](reproduction.md) · [Semantic boundary findings](redteam.md) · [Limitations](limitations.md) · [Comparison video methods](same-memory-video.md) · [Comparison notebooks and renderer](../examples/same_memory_video/README.md)

**Project:** [Policies](policies.md) · [Licenses and attribution](licenses.md) · [Contributing to Venn](../CONTRIBUTING.md)

**Maintainers:** [Maintainer guide](maintainers/maintaining.md) · [Documentation site](maintainers/website.md) · [Release checklist](maintainers/release-checklist.md) · [Publishing the Python package](maintainers/pypi.md) · [Publishing the decision head](maintainers/publishing-the-head.md) · [Development evidence](maintainers/development-evidence.md) · [Comparative evaluation protocol](maintainers/evaluation-plan.md) · [License audit notes](maintainers/license-audit.md)

</details>

**Results and limits / Results and evidence**
<!-- docs-navigation:end -->

This ledger lists each headline claim, the method and artifacts behind it, and
how to reproduce it. It is not a new evaluation.

| Claim | Method / artifact | Reproduction | Status |
| --- | --- | --- | --- |
| Ten VennSpace-1 geometric metrics | [Protocol](../benchmarks/vennspace-v1.json), [rounded reference JSON](../benchmarks/vennspace-v1-reference.json), [methodology](benchmark.md) | `venn benchmark --output /tmp/venn-space --cache /tmp/venn-space.sqlite --gpus 1` then `python scripts/compare_reference.py /tmp/venn-space/summary.json` | Public headless code; original reader/data fetched separately |
| Sixteen semantic boundary rates | 32 authored cases in `venn/redteam.py`; [reference table](redteam.md) | `venn redteam --output /tmp/venn-boundary --cache /tmp/venn-boundary.sqlite --gpus 1` | Public headless code; finite authored suite |
| All 26 rates within reference CIs in an independent run | Summarized in [benchmark](benchmark.md) and [boundary findings](redteam.md) | Requires a new independent fresh-embedding run | Reported result; the original environment and raw-run archive are not distributed here |
| 787K generic-head parameters / 3.15 MB FP32 | Head architecture, [model card](model-card.md) | Run `venn download-model`, then load the head and count tensors | Architectural size, excluding reader/runtime; not a speed or quality result |
| Text reader: approximately 270M parameters, about 540 MB at 16-bit | [Google model card](https://huggingface.co/google/embeddinggemma-2), pinned checkpoint tensor headers | [Footprint inspection](performance.md#model-footprint) | Text/shared tensors: 542,005,296 BF16 bytes; default CPU loads float32 |
| Quantized reader: about 191 MB active RAM for text-only weights | [Google launch announcement](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/) | [Deployment configurations](performance.md#model-footprint) | Google measurement on Pixel 11 Pro; separate from Venn's current PyTorch reader |

Development measurements of the outcome API are recorded in the
[development evidence](maintainers/development-evidence.md).

## Comparative results

The [Venn and Jev guide](venn-and-jev.md) explains when these results matter.
The table below is the index to the current comparisons and runnable examples.

| Task / metric | Venn | Jev / comparison method | Evaluation scope | Source and how to try it |
| --- | --- | --- | --- | --- |
| Support routing, Banking77 · accuracy ↑ | References: 93.6% | Jev 1.13: 79.7% | 3,080 tickets | [Precedent Desk](../examples/precedent_desk/README.md), its `eval.py` runner and [recorded manifest](media/precedent-desk-manifest.json) |
| Support routing, CLINC150 · accuracy ↑ | References: 94.3% | Jev 1.13: 91.1% | 4,498 tickets | [Second-dataset results and runner](../examples/precedent_desk/README.md#what-was-measured) |
| CFPB refunds · AUROC ↑ | Neighbors: 0.768 | Jev-label historical rates: 0.702; trained Jev features: 0.816 | Predict outcomes using complaint history; the trained Jev-feature model scores highest | [History walkthrough](../examples/same_memory_video/history.ipynb), [numeric record](media/same-memory/history-eval.json); [outcome-store example](../examples/outcome_store.py) |
| SEC industry groups · accuracy ↑ | Linear family: 73.3% | Jev: 64.1% | 1,000 additional reports; report-only confirmation. The original 60-filing accuracy and useful-answer gates failed | [SIC walkthrough](../examples/same_memory_video/sec.ipynb), [numeric record](media/same-memory/sec-eval.json); [linear-family recipe](cookbooks.md#fit-a-linear-choice-family) |
| Wine critic score · RMSE ↓ | Kernel family: 1.803 | Jev 38-question pipeline: 1.794 | 3,000 fresh notes; cookbook runner uses a separate 1,200/800 split | [Wine walkthrough](../examples/same_memory_video/wine.ipynb), [numeric record](media/same-memory/wine-eval.json); `python examples/wine/wine.py --form kernel_ridge` |
| Wine price · log-price RMSE ↓ | 0.538 | Direct Jev: 0.531 | Accuracy and cached-vector runtime are separate measurements | [Price record](media/same-memory/wine-eval.json) and [workflow](same-memory-video.md#wine-accuracy-and-work-are-separate-measurements) |
| Audio emotion, CREMA-D · accuracy ↑ | Audio linear family: 56.5% | Jev on transcripts: 14.6% | 1,716 clips; inputs differ: sound versus words | [Sound walkthrough](../examples/same_memory_video/sound.ipynb), [numeric record](media/same-memory/sound-eval.json); [audio example](../examples/audio/README.md) |
| Grape naming · macro-F1 ↑ | 0.340 | Jev: 0.427 | November 6 comparison write-up; full evaluation runner not included | [Comparison discussion](venn-and-jev.md#choose-by-the-work-you-need-to-do) |
| Firefox component routing · accuracy ↑ | Head: 0.583 | Jev: 0.641 | November 6 comparison write-up; full evaluation runner not included | [Comparison discussion](venn-and-jev.md#choose-by-the-work-you-need-to-do) |

↑ Higher is better; ↓ lower is better. These are separate task evaluations,
not a single aggregate benchmark.

### Runtime and arena results

| Measurement | Recorded result | Scope / caveat | Source |
| --- | --- | --- | --- |
| Wine predictor build | Venn: 31 s on one A100; Jev discovery: 17.4 min | Separate five-round discovery run produced 35 questions; accuracy above uses the 38-question pipeline | [Build record and timing boundaries](same-memory-video.md#wine-accuracy-and-work-are-separate-measurements) |
| Wine price fit and prediction | About 7 s for 65,701 cached notes | Venn fitting and scoring on already embedded inputs; excludes fresh embedding | [Price workflow and record](same-memory-video.md#wine-accuracy-and-work-are-separate-measurements) |
| Arena action selection | Compiled head: 8.36 kills/game; memory: 6.20 | 250 scored games; comparison is head versus memory, not Jev | [Arena results and runner](arena-example.md), [recorded manifest](media/venn-arena-manifest.json) |

The comparison notebooks walk through recorded results. The wine, support desk,
audio and arena examples run the respective Venn workflows on source or local
data. Detailed experiment records stay on the [methods page](same-memory-video.md).
Scientific benchmark commands above continue to use the frozen VennSpace-1 protocol.

## Interpreting reproduction

| Reproduction detail | Meaning |
| --- | --- |
| Reference JSON | Transcription of the published two-decimal percentage table, with the frozen protocol hash; original raw measurements were not recovered |
| `compare_reference.py` | Checks each new point estimate against the rounded reference intervals |
| Exit `0` | All ten metrics are inside the intervals |
| Exit `1` | At least one metric is outside the intervals |
| Exit `2` | Input is invalid or incompatible |
| Interpretation limit | Consistency with rounded intervals does not establish statistical equivalence, generalization, or a learned-head advantage |

New runs emit predictions and source-row manifests locally. Source corpora,
reader weights and caches are not part of this repository. Record software
versions, hardware and fresh/cache status when reporting results. Do not
relabel synthetic sweeps as semantic evaluation.

Further comparisons are specified in
[the comparison protocol](maintainers/evaluation-plan.md). The
[performance and cost guide](performance.md) explains which work is reused,
which expenses remain, and how to measure latency and throughput.

<!-- docs-pager:start -->
---

[← Previous: VennSpace-1](benchmark.md) · [Next: Reproduction →](reproduction.md) · [Documentation home](README.md)
<!-- docs-pager:end -->
