# Performance and cost

<!-- docs-navigation:start -->
[Documentation](README.md) · [Get started](getting-started.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md) · [Examples](../examples/README.md) · [Task recipes](cookbooks.md) · [API](api.md) · [Results](evidence.md)

<details>
<summary>Browse all documentation</summary>

**Project home:** [Home](../README.md)

**Start here:** [Documentation overview](README.md) · [Getting started](getting-started.md) · [Installation](installation.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md)

**Task guides:** [Task recipes](cookbooks.md) · [Outcome stores](outcome-api.md)

**Examples:** [Choose an example](../examples/README.md) · [Hot dog image classifier](../examples/hotdog/README.md) · [Precedent Desk](../examples/precedent_desk/README.md) · [Wine predictor](../examples/wine/README.md) · [Audio classifier](../examples/audio/README.md) · [Screen-only arena](arena-example.md) · [Demos and videos](demos.md)

**Reference:** [API reference](api.md) · [Design rationale](design.md) · [Venn 1 model card](model-card.md)

**Results and limits:** **Performance and cost** · [VennSpace-1](benchmark.md) · [Results and evidence](evidence.md) · [Reproduction](reproduction.md) · [Semantic boundary findings](redteam.md) · [Limitations](limitations.md) · [Comparison video methods](same-memory-video.md) · [Comparison notebooks and renderer](../examples/same_memory_video/README.md)

**Project:** [Policies](policies.md) · [Licenses and attribution](licenses.md) · [Contributing to Venn](../CONTRIBUTING.md)

**Maintainers:** [Maintainer guide](maintainers/maintaining.md) · [Documentation site](maintainers/website.md) · [Release checklist](maintainers/release-checklist.md) · [Publishing the Python package](maintainers/pypi.md) · [Publishing the decision head](maintainers/publishing-the-head.md) · [Development evidence](maintainers/development-evidence.md) · [Comparative evaluation protocol](maintainers/evaluation-plan.md) · [License audit notes](maintainers/license-audit.md)

</details>

**Results and limits / Performance and cost**
<!-- docs-navigation:end -->

<!-- docs-contents:start -->
<details>
<summary>On this page</summary>

- [Model footprint](#model-footprint)
- [Alongside other local decision models](#alongside-other-local-decision-models)
- [Where the savings can come from](#where-the-savings-can-come-from)
- [A workload cost model](#a-workload-cost-model)
- [Measure the CPU sweep](#measure-the-cpu-sweep)
- [What a performance comparison must report](#what-a-performance-comparison-must-report)

</details>
<!-- docs-contents:end -->

Venn is designed to make repeated decisions over existing embeddings low in
latency and incremental compute cost. Its decision layer reuses a representation
that may already have been computed for search, routing or another task.

## Model footprint

Venn's generic decision head is 787,457 parameters, about 3.15 MB in float32.
It runs on CPU. With existing vectors, the reader can stay unloaded; the
outcome-store path also runs without the generic head.

For fresh text, Venn disables EmbeddingGemma 2's vision and audio towers with
`config_kwargs={"vision_config": None, "audio_config": None}`. Google's
[model card](https://huggingface.co/google/embeddinggemma-2) gives the text-only
model as approximately 270 million parameters. Google's
[launch announcement](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/)
reports that quantized text-only weights can occupy about 191 MB of active RAM
on a Pixel 11 Pro; the full multimodal model occupies about 567 MB in that
deployment.

| Configuration | Footprint | Measure |
| --- | ---: | --- |
| Google's quantized text-only deployment, Pixel 11 Pro | About 191 MB | Active RAM for weights, reported by Google |
| Google's quantized full multimodal deployment, Pixel 11 Pro | About 567 MB | Active RAM for weights, reported by Google |
| Generic Venn head, float32 | 3.15 MB | Weight storage |
| Text reader, 16-bit | About 540 MB | Weight storage |
| Text reader, float32 CPU default | About 1.08 GB | Weight storage |
| Full upstream multimodal checkpoint downloaded by the CLI | About 1.49 GB | Weight file |

Venn's current reader path uses PyTorch, with float32 on CPU. Google's smaller
quantized deployment uses a different execution path; Venn does not currently
provide that backend. Weight storage uses decimal MB/GB and excludes runtime
working memory. Google's active-RAM numbers also refer specifically to weights,
not an entire application's memory use.

The pinned checkpoint at revision `f9972ac22e696a105c482202646979e33e4b5a1e`
has a 1,488,915,288-byte safetensors file. Its text/shared tensors account for
542,005,296 bytes at BF16. To inspect the cached file without loading its
weights into a model, substitute the path printed by `venn download-embedder`:

```python
from math import prod
from pathlib import Path
from safetensors import safe_open

path = Path("/path/to/reader/model.safetensors")
elements = 0
with safe_open(path, framework="numpy") as weights:
    for name in weights.keys():
        if "audio" not in name and "vision" not in name:
            tensor = weights.get_slice(name)
            assert tensor.get_dtype() == "BF16"
            elements += prod(tensor.get_shape())
print({"text_shared_elements": elements, "bf16_bytes": elements * 2,
       "checkpoint_bytes": path.stat().st_size})
```

## Alongside other local decision models

The footprint advantage also matters against open-source language-model-based
alternatives. [Kev's model list](https://github.com/jaredpalmer/kev#models)
starts at 0.8B parameters and includes 4B, 9B and 27B versions;
[Open JEV](https://github.com/zhihz/openjev) uses Qwen3-4B. Venn's approximately
270M-parameter text reader is about three times smaller than a 0.8B model and
15 times smaller than a 4B model by parameter count. Quantized files and total
runtime memory depend on each implementation.

The architectural saving is the small decision computation after embedding.
Fitted linear heads score stored vectors directly; outcome decisions query
the store. The [arena head](arena-example.md) answered an embedded screen in
under 0.5 ms in the recorded run. The [wine price head](evidence.md#comparative-results)
fit and scored 65,701 stored vectors in about 7 seconds.

Local alternatives can also cache content, and projects such as
[jevos](https://github.com/feder-cr/jev) explicitly target CPU deployment.
Venn has not been timed against these projects on a common workload. Compare
their complete input-to-answer paths, including embedding and task accuracy,
before assigning a relative speedup.

The [reader can be replaced](concepts.md#changing-the-reader) independently of
the outcome decision API. A different reader can trade size, speed and quality
without requiring a different application-level decision loop.

## Where the savings can come from

Consider a ticket that needs a team, a priority and a topic. Its content can be
encoded once, then reused by each question. For a collection whose documents
are already embedded, a new question requires preparing its instructions/options
and scoring the stored vectors. It does not require re-encoding every document.
All vectors must belong to the supported profile.

| Component | When it is paid | Reuse opportunity |
| --- | --- | --- |
| Content encoding | New or changed content; profile changes | Cache each compatible state across decisions |
| Question preparation | New or changed instructions/options | Prewarm once for repeated rows |
| Decision scoring | Each requested decision | Small heads, CPU execution and batches |
| Storage and transport | Reading vectors and returning results | Local placement and bounded streaming |
| Family fitting | New or revised task-specific family | Reuse the fitted family across requests |
| Outcome evidence lookup | Each outcome question, separately per option | Reuse the reader state; keep indexes near the decision process |
| Outcome feedback writes | When the app observes a utility | App-owned store; include commit and read-visibility costs |

The generic head has 787,457 trainable parameters: about 3.15 MB of FP32
parameter storage. This excludes model buffers, the embedding reader, loaded
libraries, vector stores and request working memory. Vector-only serving can
execute on CPU without loading the reader. Raw text, including new textual
criteria, still requires encoding.

## A workload cost model

For N items and Q questions applied to each item, a simplified cost decomposition is:

```text
Total work = content encoding
           + question preparation
           + N × Q × per-decision scoring
           + vector I/O and result delivery
           + any family fitting
```

For an existing compatible vector collection, content encoding has already
been paid. For fresh content, amortize it over the actual number of decisions.
Report both total and incremental costs. Convert measured compute time to dollars
using the actual hardware/provider rate, including utilization and idle capacity;
local inference still consumes resources. The API reports zero in its token and
cost usage fields; they are not instrumented.

This is a cost-accounting model. Jev, TypeSafe's hosted decision model whose
Decisions API request format Venn accepts, supports multiple typed questions in
one request; comparisons should use that capability and any relevant caching.
See [TypeSafe's interface description](https://docs.typesafe.ai/introduction).

## Measure the CPU sweep

After installing Venn, run:

```bash
venn sweep --rows 1000 --batch-size 256
venn sweep --rows 1000000 --batch-size 8192
```

The larger run creates a temporary 1.536 GB float16 vector file plus working
memory. It uses seeded synthetic vectors and headless centered-cosine scoring.
No reader or head download is needed. Every output is consumed and hashed.

Read `generation_seconds`, `scoring_serialization_consumption_seconds`,
`rows_per_second`, `end_to_end_seconds` and `counts`. Counts must sum to `rows`.
Throughput covers scoring, serialization and consumption; generation is reported
separately. The report excludes model/service loading and final report writing.
This standalone path does not measure HTTP requests, a database adapter, learned
abstention, or semantic quality. To exercise the actual index streaming service, which
applies one prepared question to every vector in a registered index and streams
the results as NDJSON, use the trained-head path in the [combined demo](demos.md).

## What a performance comparison must report

- **Quality:** identical tasks and inputs, with accuracy and abstention coverage
  measured alongside speed. An inexpensive wrong decision is not a useful saving.
- **Latency:** cold startup, first request and warm p50/p95 latency, with batch
  size and concurrency. Aggregate rows/s is not single-request latency.
- **Throughput:** rows and questions per second, including full output consumption.
- **Resources:** CPU/GPU model, thread count, memory, software versions and power
  or hosting cost where measured.
- **Accounting:** reader, preparation, fitting, vector I/O, serialization and
  transport costs; distinguish already embedded inputs from fresh inputs.

The [evidence ledger](evidence.md#comparative-results) records current comparisons.
A wine head was built in 31 seconds on one A100; a separate five-round Jev
discovery run took 17.4 minutes and produced 35 questions. The accuracy
comparison uses the 38-question pipeline. For price, fitting and applying a
head over 65,701 already embedded notes took about 7 seconds. See
[Venn and Jev](venn-and-jev.md) for the workflows and results.

Reuse follows the representation. The winning wine score head uses
SentenceSimilarity embeddings and text length inputs; the price comparison
uses Classification embeddings. A family requiring text inputs cannot consume
a vector-only index. Kernel ridge compares query rows with its fitting vectors;
its cost grows with the fitting set. Linear ridge and Choice heads are fixed
projections suitable for matrix scoring across a collection.

Development measurements of the in-process outcome API are recorded in the
[development evidence](maintainers/development-evidence.md).

<!-- docs-pager:start -->
---

[← Previous: Venn 1 model card](model-card.md) · [Next: VennSpace-1 →](benchmark.md) · [Documentation home](README.md)
<!-- docs-pager:end -->
