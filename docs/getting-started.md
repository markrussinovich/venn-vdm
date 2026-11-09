# Getting started

<!-- docs-navigation:start -->
[Documentation](README.md) · **Get started** · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md) · [Examples](../examples/README.md) · [Task recipes](cookbooks.md) · [API](api.md) · [Results](evidence.md)

<details>
<summary>Browse all documentation</summary>

**Project home:** [Home](../README.md)

**Start here:** [Documentation overview](README.md) · **Getting started** · [Installation](installation.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md)

**Task guides:** [Task recipes](cookbooks.md) · [Outcome stores](outcome-api.md)

**Examples:** [Choose an example](../examples/README.md) · [Hot dog image classifier](../examples/hotdog/README.md) · [Precedent Desk](../examples/precedent_desk/README.md) · [Wine predictor](../examples/wine/README.md) · [Audio classifier](../examples/audio/README.md) · [Screen-only arena](arena-example.md) · [Demos and videos](demos.md)

**Reference:** [API reference](api.md) · [Design rationale](design.md) · [Venn 1 model card](model-card.md)

**Results and limits:** [Performance and cost](performance.md) · [VennSpace-1](benchmark.md) · [Results and evidence](evidence.md) · [Reproduction](reproduction.md) · [Semantic boundary findings](redteam.md) · [Limitations](limitations.md) · [Comparison video methods](same-memory-video.md) · [Comparison notebooks and renderer](../examples/same_memory_video/README.md)

**Project:** [Policies](policies.md) · [Licenses and attribution](licenses.md) · [Contributing to Venn](../CONTRIBUTING.md)

**Maintainers:** [Maintainer guide](maintainers/maintaining.md) · [Documentation site](maintainers/website.md) · [Release checklist](maintainers/release-checklist.md) · [Publishing the Python package](maintainers/pypi.md) · [Publishing the decision head](maintainers/publishing-the-head.md) · [Development evidence](maintainers/development-evidence.md) · [Comparative evaluation protocol](maintainers/evaluation-plan.md) · [License audit notes](maintainers/license-audit.md)

</details>

**Start here / Getting started**
<!-- docs-navigation:end -->

<!-- docs-contents:start -->
<details>
<summary>On this page</summary>

- [Install](#install)
- [Your first result](#your-first-result)
- [Ways to run Venn](#ways-to-run-venn)
- [Decide from an outcome store](#decide-from-an-outcome-store)
- [Serve text questions](#serve-text-questions)
- [Reproduce the measurements](#reproduce-the-measurements)
- [Troubleshooting](#troubleshooting)
- [Continue with a workflow](#continue-with-a-workflow)

</details>
<!-- docs-contents:end -->

Get a first CPU result without downloading a reader or head, then choose
a workflow for real vectors or text. The first example uses authored coordinates;
it demonstrates the interface, not semantic accuracy.

## Install

Venn requires Python 3.12. CPU workflows run on Linux and Windows, including
ARM64. On Linux, clone the repository and enter its root directory:

```bash
git clone https://github.com/markrussinovich/venn-vdm.git
cd venn
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only PyTorch
python -m pip install .
python -m pip check
```

If you already have a checkout, start with the virtual-environment command
from its root directory.

Skip the `pip install torch` line if you want CUDA PyTorch; the default
wheel on Linux includes the CUDA libraries and is several times larger. Neither
install downloads a model. Contributors use `python -m pip install -e '.[dev]'`.

### ARM64 and Windows

Use the dedicated [ARM64 and Windows setup](installation.md#arm64-and-windows)
for native Python and matching CPU PyTorch wheels. The
[installation guide](installation.md) also lists optional extras and download requirements.

## Your first result

Run the first example:

```bash
venn quickstart
```

The JSON response names `account` as the answer. Its `options` entries show
mean outcomes of **0.9 for account** and **0.4 for billing**, three cases each,
and the IDs of the supporting cases. Account wins because its nearby cases
had better recorded outcomes. Your application supplies what “good” means.

This example uses hand-written vectors to demonstrate the feedback-and-lookup
mechanism without downloading a model. To decide from actual ticket text,
continue to [Serve text questions](#serve-text-questions).

### Optional: measure a synthetic vector sweep

`venn sweep --rows 1000 --batch-size 256` generates normalized synthetic
768-dimensional vectors, scores four centered options for every row and consumes
all output. It reports `rows`, per-option counts that sum to `rows`, and
timings for generation and scoring. Use it as a CPU throughput probe; it loads
no model. Rows go up to one million, using about 1.5 KB of temporary disk per row.

## Ways to run Venn

| Path | Needs | Start with |
| --- | --- | --- |
| CPU quickstart and sweep | Nothing downloaded | `venn quickstart`, `venn sweep --rows 1000` |
| Decisions from an outcome store | Your tagged vectors and a store; no head, no GPU | [examples/outcome_store.py](../examples/outcome_store.py), [outcome API](outcome-api.md) |
| Text questions over HTTP | The reader and the head: `venn download-embedder`, `venn serve` | [examples/route_ticket.py](../examples/route_ticket.py), [API](api.md) |
| Benchmark and authored demos | `.[embedding,benchmark]` and a CUDA GPU | `venn benchmark`, `venn redteam`, `venn demos` |
| Arena example | `.[arena,embedding]` | [arena example](arena-example.md) |

## Decide from an outcome store

`OutcomeService` chooses among options using the outcomes recorded in your own
vector store. It needs no trained head, and for already-tagged vectors it needs
no reader or GPU:

```bash
python examples/outcome_store.py
```

It prints `support`. The script creates a FAISS store, records one resolved
case, and asks which queue a new ticket should go to. The [outcome API
guide](outcome-api.md) explains tags, adapters for other stores, image batches
and how your application closes the feedback loop.

## Serve text questions

Text questions need the reader and the `venn-1` head. The reader is
EmbeddingGemma 2, an Apache-2.0 model. The download contains about 1.49 GB of
multimodal weights. Text serving loads only the approximately 270M-parameter
text components: about 1.08 GB of weights in the default CPU float32 mode.
See [model footprint](performance.md#model-footprint) for sizes and precision.

```bash
python -m pip install '.[embedding]'
venn download-embedder
venn serve --host 127.0.0.1 --port 8000
```

`venn serve` downloads the head from `markrussinovich/venn-1` at the pinned
revision on first use. To use a local copy instead, pass `--model-dir` or set
`VENN_MODEL_DIR` to a directory holding `config.json`, `manifest.json` and
`universal.safetensors`. Server state (the embedding cache and registered
families) lives under `VENN_HOME`, which defaults to `~/.cache/venn`.
`venn serve` starts loading the reader in the background; pass `--no-preload`
to skip that on low-memory or vector-only servers. See
[server options](api.md#server-options-and-host-checks).

In another terminal:

```bash
python examples/route_ticket.py --text "I cannot sign in"
```

The response contains `answers.team.choice`, the probabilities and Venn's
abstention metadata. The first text request loads the reader and can take
several seconds; `GET /health` reports `reader_loaded` once it is ready. Add
the header `X-Venn-Return-Vector: 1` if you want the 768-dimensional state
vector back in the response. Including a criterion with the value `null`, as the example
does with `no_match`, gives the model an explicit way to say none of the above.
Vector-only requests do not need the reader, but every vector must come from
the exact `venn-1` profile.

## Reproduce the measurements

The benchmark, the semantic boundary suite and the authored demos embed text
with the pinned reader and currently require at least one CUDA GPU. Install
`.[embedding,benchmark,dev]` in a clean environment, run `venn download-embedder`,
then follow the [reproduction guide](reproduction.md). The reference run used
four A100 GPUs; requirements for smaller hardware have not been measured.

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| Python version rejected | Use Python 3.12 when creating the virtual environment |
| `venn serve` cannot download the head | Check network access to huggingface.co, or set `VENN_MODEL_DIR` to a local copy |
| Reader download fails | Check network access, disk space and the reported error. For an access-denied response, check the model repository's access requirements and authenticate with `hf auth login` if required |
| CUDA preflight fails | Pass `--gpus` with a visible device count; use `venn quickstart` and `venn sweep` on CPU |
| Import resolves to a diagram package | Create a clean virtual environment without the unrelated `venn` distribution |
| Profile or revision mismatch | Re-embed with the exact profile and re-register families and indexes |
| Client cannot connect | Start the server in another terminal and check `curl http://127.0.0.1:8000/health` |
| Text request returns 503 `reader_missing` | The reader is not downloaded: run `venn download-embedder`, then retry. `GET /health` reports `"reader": "absent"` until then |
| 400 `Invalid Host header` | The server accepts only loopback hostnames. After `--host 0.0.0.0`, allow the name or IP clients use with `--allowed-host NAME`; it replaces the loopback defaults, so repeat it for `127.0.0.1` if needed. See [host checks](api.md#server-options-and-host-checks) |
| 403 `Invalid Origin header` | A browser page on another origin called the API. Serve the page from the same scheme, host and port |

The server has no authentication and binds to loopback by default. Keep it local.

## Continue with a workflow

Read [concepts](concepts.md) to understand readers, profiles, stores and families.
Choose a runnable path in the [example catalog](../examples/README.md), follow a
[task recipe](cookbooks.md), or look up request fields in the [API reference](api.md).

<!-- docs-pager:start -->
---

[← Previous: Documentation overview](README.md) · [Next: Installation →](installation.md) · [Documentation home](README.md)
<!-- docs-pager:end -->
