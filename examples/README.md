# Examples

<!-- docs-navigation:start -->
[Documentation](../docs/README.md) · [Get started](../docs/getting-started.md) · [Concepts](../docs/concepts.md) · [Venn vs Jev](../docs/venn-and-jev.md) · **Examples** · [Task recipes](../docs/cookbooks.md) · [API](../docs/api.md) · [Results](../docs/evidence.md)

<details>
<summary>Browse all documentation</summary>

**Project home:** [Home](../README.md)

**Start here:** [Documentation overview](../docs/README.md) · [Getting started](../docs/getting-started.md) · [Installation](../docs/installation.md) · [Concepts](../docs/concepts.md) · [Venn vs Jev](../docs/venn-and-jev.md)

**Task guides:** [Task recipes](../docs/cookbooks.md) · [Outcome stores](../docs/outcome-api.md)

**Examples:** **Choose an example** · [Hot dog image classifier](hotdog/README.md) · [Precedent Desk](precedent_desk/README.md) · [Wine predictor](wine/README.md) · [Audio classifier](audio/README.md) · [Screen-only arena](../docs/arena-example.md) · [Demos and videos](../docs/demos.md)

**Reference:** [API reference](../docs/api.md) · [Design rationale](../docs/design.md) · [Venn 1 model card](../docs/model-card.md)

**Results and limits:** [Performance and cost](../docs/performance.md) · [VennSpace-1](../docs/benchmark.md) · [Results and evidence](../docs/evidence.md) · [Reproduction](../docs/reproduction.md) · [Semantic boundary findings](../docs/redteam.md) · [Limitations](../docs/limitations.md) · [Comparison video methods](../docs/same-memory-video.md) · [Comparison notebooks and renderer](same_memory_video/README.md)

**Project:** [Policies](../docs/policies.md) · [Licenses and attribution](../docs/licenses.md) · [Contributing to Venn](../CONTRIBUTING.md)

**Maintainers:** [Maintainer guide](../docs/maintainers/maintaining.md) · [Documentation site](../docs/maintainers/website.md) · [Release checklist](../docs/maintainers/release-checklist.md) · [Publishing the Python package](../docs/maintainers/pypi.md) · [Publishing the decision head](../docs/maintainers/publishing-the-head.md) · [Development evidence](../docs/maintainers/development-evidence.md) · [Comparative evaluation protocol](../docs/maintainers/evaluation-plan.md) · [License audit notes](../docs/maintainers/license-audit.md)

</details>

**Examples / Choose an example**
<!-- docs-navigation:end -->

| Example | Needs | What it shows |
| --- | --- | --- |
| [Outcome store](outcome_store.py) | Base install; no downloads | `python examples/outcome_store.py`: a decision and case IDs from a tiny authored store |
| [Text routing](route_ticket.py) | `.[embedding]`, reader and running `venn serve` | `python examples/route_ticket.py`: a Choice response with account/billing probabilities and a no-match option |
| [Precedent Desk](precedent_desk/README.md) | Reader, head and public support-ticket data; see its setup | Route from closed tickets, inspect precedents, add a queue and sweep a backlog |
| [Registered family](registered_family.py) | Base install; head downloaded automatically | `python examples/registered_family.py`: fit on authored labeled vectors, reload and compare answers |
| [Wine predictor](wine/README.md) | `.[embedding]`, reader, head and downloaded CSV; CPU or GPU | `python examples/wine/wine.py --form kernel_ridge`: fit on 1,200 notes, report RMSE on 800 others |
| [Index sweep](index_vectors.py) | Running server; `scripts/contract_server.py` is enough | `python examples/index_vectors.py`: register an index, stream every row and delete it |
| [Audio emotion](audio/README.md) | `.[embedding]`, reader, head and your labeled WAV clips; CPU | Fit a linear family from audio and print a new clip's emotion probabilities |
| [Hot dog or not](hotdog/README.md) | `.[embedding,dev]`, reader, head and your local photos; CPU or GPU | Visual local demo: train once, evaluate unseen photos, then judge photos with separate reader and decision timings |
| [Arena](../docs/arena-example.md) | `.[arena,embedding]`; GPU for the native reader | Play from screenshots and recorded outcomes; `--head awr` compiles memory into a small predictor |
| [Comparison walkthroughs](same_memory_video/README.md) | Numeric exports included; notebook environment for exploration | Inspect recorded history, SIC, wine and sound results; render the companion video |

Run commands from the repository root after `python -m pip install .` and the
extra listed for the example. Download the reader with `venn download-embedder`
when required. The offline outcome, index and registered-family examples use
authored coordinates to demonstrate interfaces. Text routing uses actual text
embeddings when connected to `venn serve`. Wine reports RMSE and timings; the
other small scripts print JSON.

Start with `venn quickstart` for a CPU example with no downloads. For choosing a
workflow, read [Venn and Jev](../docs/venn-and-jev.md); for the request shapes,
see the [cookbooks](../docs/cookbooks.md).

<!-- docs-pager:start -->
---

[← Previous: Outcome stores](../docs/outcome-api.md) · [Next: Hot dog image classifier →](hotdog/README.md) · [Documentation home](../docs/README.md)
<!-- docs-pager:end -->
