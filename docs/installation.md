# Installation

<!-- docs-navigation:start -->
[Documentation](README.md) · [Get started](getting-started.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md) · [Examples](../examples/README.md) · [Task recipes](cookbooks.md) · [API](api.md) · [Results](evidence.md)

<details>
<summary>Browse all documentation</summary>

**Project home:** [Home](../README.md)

**Start here:** [Documentation overview](README.md) · [Getting started](getting-started.md) · **Installation** · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md)

**Task guides:** [Task recipes](cookbooks.md) · [Outcome stores](outcome-api.md)

**Examples:** [Choose an example](../examples/README.md) · [Hot dog image classifier](../examples/hotdog/README.md) · [Precedent Desk](../examples/precedent_desk/README.md) · [Wine predictor](../examples/wine/README.md) · [Audio classifier](../examples/audio/README.md) · [Screen-only arena](arena-example.md) · [Demos and videos](demos.md)

**Reference:** [API reference](api.md) · [Design rationale](design.md) · [Venn 1 model card](model-card.md)

**Results and limits:** [Performance and cost](performance.md) · [VennSpace-1](benchmark.md) · [Results and evidence](evidence.md) · [Reproduction](reproduction.md) · [Semantic boundary findings](redteam.md) · [Limitations](limitations.md) · [Comparison video methods](same-memory-video.md) · [Comparison notebooks and renderer](../examples/same_memory_video/README.md)

**Project:** [Policies](policies.md) · [Licenses and attribution](licenses.md) · [Contributing to Venn](../CONTRIBUTING.md)

**Maintainers:** [Maintainer guide](maintainers/maintaining.md) · [Documentation site](maintainers/website.md) · [Release checklist](maintainers/release-checklist.md) · [Publishing the Python package](maintainers/pypi.md) · [Publishing the decision head](maintainers/publishing-the-head.md) · [Development evidence](maintainers/development-evidence.md) · [Comparative evaluation protocol](maintainers/evaluation-plan.md) · [License audit notes](maintainers/license-audit.md)

</details>

**Start here / Installation**
<!-- docs-navigation:end -->

<!-- docs-contents:start -->
<details>
<summary>On this page</summary>

- [Linux setup](#linux-setup)
- [ARM64 and Windows](#arm64-and-windows)
- [Verify the installation](#verify-the-installation)

</details>
<!-- docs-contents:end -->

Use this page for Python, PyTorch, platform-specific setup and optional extras.
For the shortest route to a result, follow [getting started](getting-started.md).

## Linux setup

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

## ARM64 and Windows

Use native ARM64 Python, not an x86-64 interpreter under emulation. The CPU
quickstart, sweep and development tests have been exercised on an ARM64 Windows
laptop both natively and in Ubuntu under WSL 2. This does not establish GPU,
trained-head text serving, media or optional-adapter compatibility. The pinned
reader has also been smoke-tested for CPU text embedding on both platforms.

For CPU text support and development, install
[uv](https://docs.astral.sh/uv/getting-started/installation/) and use the two-step
recipe below: install matching `torch` and `torchvision` packages from the
official CPU index, then install Venn from your regular package feed. Neither
step downloads model weights.

On Ubuntu ARM64, including WSL, run these commands from the clone:

```bash
uv venv --python cpython-3.12-linux-aarch64-gnu .venv
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e ".[embedding,dev]"
uv pip check --python .venv/bin/python
.venv/bin/python -m venn quickstart
.venv/bin/python -m venn sweep --rows 1000 --batch-size 256
```

For native Windows ARM64, install uv and run these commands from the clone in
PowerShell. The explicit interpreter target avoids accidentally testing x86-64
emulation:

```powershell
uv venv --python cpython-3.12-windows-aarch64-none .venv
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv\Scripts\python.exe -e ".[embedding,dev]"
uv pip check --python .venv\Scripts\python.exe
.\.venv\Scripts\python.exe -c "import platform; print(platform.machine())"
.\.venv\Scripts\python.exe -m venn quickstart
.\.venv\Scripts\python.exe -m venn sweep --rows 1000 --batch-size 256
```

The architecture check should print `ARM64`. Install CPU PyTorch first from its
official index: the default package feed may not carry Windows ARM64 wheels,
and Linux package resolution may otherwise pull in unnecessary CUDA packages.
Install `torch` and `torchvision` together so the resolver can select compatible
versions; the available pair may differ between platforms. For a smaller,
reader-free environment, install `torch` alone and use `.[dev]` instead.
If you use a package mirror, it must also carry Venn's required reader
dependencies, including `transformers>=5.22` and `sentence-transformers>=6.1`.
The PyTorch CPU index does not replace your regular package feed for these.
The benchmark and native arena GPU paths require CUDA; these CPU checks do not
enable Qualcomm GPU or NPU acceleration.

Extras add optional capabilities:

| Extra | Adds |
| --- | --- |
| `embedding` | The EmbeddingGemma 2 reader, for text, image and audio input |
| `benchmark` | Datasets and tooling for `venn benchmark`, `venn redteam` and `venn demos` |
| `adapters` | Qdrant, Weaviate, LanceDB and pgvector clients |
| `mcp` | The MCP bridge |
| `arena` | ViZDoom and video encoding for the arena example |

For task selection, read [Venn and Jev](venn-and-jev.md). For runnable workflows,
see the [example index](../examples/README.md).


## Verify the installation

Run `venn quickstart` for an authored CPU example with no model downloads.
Then choose a workflow in the [documentation overview](README.md#choose-a-workflow)
or the [example catalog](../examples/README.md). For download and runtime issues,
see [troubleshooting](getting-started.md#troubleshooting).

<!-- docs-pager:start -->
---

[← Previous: Getting started](getting-started.md) · [Next: Concepts →](concepts.md) · [Documentation home](README.md)
<!-- docs-pager:end -->
