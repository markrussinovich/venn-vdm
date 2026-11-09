# Licenses and attribution

<!-- docs-navigation:start -->
[Documentation](README.md) · [Get started](getting-started.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md) · [Examples](../examples/README.md) · [Task recipes](cookbooks.md) · [API](api.md) · [Results](evidence.md)

<details>
<summary>Browse all documentation</summary>

**Project home:** [Home](../README.md)

**Start here:** [Documentation overview](README.md) · [Getting started](getting-started.md) · [Installation](installation.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md)

**Task guides:** [Task recipes](cookbooks.md) · [Outcome stores](outcome-api.md)

**Examples:** [Choose an example](../examples/README.md) · [Hot dog image classifier](../examples/hotdog/README.md) · [Precedent Desk](../examples/precedent_desk/README.md) · [Wine predictor](../examples/wine/README.md) · [Audio classifier](../examples/audio/README.md) · [Screen-only arena](arena-example.md) · [Demos and videos](demos.md)

**Reference:** [API reference](api.md) · [Design rationale](design.md) · [Venn 1 model card](model-card.md)

**Results and limits:** [Performance and cost](performance.md) · [VennSpace-1](benchmark.md) · [Results and evidence](evidence.md) · [Reproduction](reproduction.md) · [Semantic boundary findings](redteam.md) · [Limitations](limitations.md) · [Comparison video methods](same-memory-video.md) · [Comparison notebooks and renderer](../examples/same_memory_video/README.md)

**Project:** [Policies](policies.md) · **Licenses and attribution** · [Contributing to Venn](../CONTRIBUTING.md)

**Maintainers:** [Maintainer guide](maintainers/maintaining.md) · [Documentation site](maintainers/website.md) · [Release checklist](maintainers/release-checklist.md) · [Publishing the Python package](maintainers/pypi.md) · [Publishing the decision head](maintainers/publishing-the-head.md) · [Development evidence](maintainers/development-evidence.md) · [Comparative evaluation protocol](maintainers/evaluation-plan.md) · [License audit notes](maintainers/license-audit.md)

</details>

**Project / Licenses and attribution**
<!-- docs-navigation:end -->

Venn code is MIT-licensed; see [LICENSE](../LICENSE) and [NOTICE](../NOTICE). Reader weights, datasets
and trained heads have separate terms. The code license does not relicense them.

| Component | License | Notes |
| --- | --- | --- |
| Venn code | MIT | [LICENSE](../LICENSE) and [NOTICE](../NOTICE) accompany source downloads |
| EmbeddingGemma 2 reader | Apache-2.0 | Google DeepMind. Downloaded separately; read the upstream model card before downloading |
| NumPy | BSD, with MIT, Zlib and CC0 components | Runtime dependency |
| PyTorch | BSD-3-Clause | Runtime dependency |
| FAISS | MIT | Runtime dependency |
| FastAPI | MIT | Runtime dependency |
| Uvicorn | BSD-3-Clause | Runtime dependency |
| orjson | Apache-2.0 or MIT | Runtime dependency |
| safetensors | Apache-2.0 | Runtime dependency |
| Hugging Face Hub | Apache-2.0 | Runtime dependency |
| Freedoom assets in the arena video | Freedoom license | [License notice](media/freedoom-copying.txt) and [credits](media/freedoom-credits.txt) accompany the recording; no commercial Doom assets are used |
| ViZDoom | Own license | Optional. The arena example uses an installed engine and does not bundle one; custom levels keep their own terms |
| Evaluation dataset texts | Per dataset | Fetched from pinned original sources at run time; not bundled |
| Precedent Desk embedding snapshots | Banking77: CC BY 4.0; CLINC: CC BY 3.0 | Separately distributed reproduction inputs; see [sources and attribution](../examples/precedent_desk/README.md) |

Optional dependencies retain their own licenses and notices; inspect the exact
installed distributions, including bundled and transitive components.
Alternative image readers are selected and downloaded separately under their
own terms.

## Venn-1 head redistribution

Redistribution of the published `venn-1` head weights is permitted. The project
owner confirmed this on November 7, 2026. The Hub card records this permission
under a custom license notice; the MIT code license is separate.
See the [Hub model card](https://huggingface.co/markrussinovich/venn-1#redistribution)
for the actual grant. This page does not expand that grant to unspecified
use, modification or commercial-use rights.

The head was trained on CLINC, MASSIVE, Banking77, DBpedia, Yahoo Answers and
AG News. Source attributions and the historical review are retained in the
[license audit](maintainers/license-audit.md). The embedding reader and datasets
keep their upstream terms.

<!-- docs-pager:start -->
---

[← Previous: Policies](policies.md) · [Next: Contributing to Venn →](../CONTRIBUTING.md) · [Documentation home](README.md)
<!-- docs-pager:end -->
