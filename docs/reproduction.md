# Reproduction

<!-- docs-navigation:start -->
[Documentation](README.md) · [Get started](getting-started.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md) · [Examples](../examples/README.md) · [Task recipes](cookbooks.md) · [API](api.md) · [Results](evidence.md)

<details>
<summary>Browse all documentation</summary>

**Project home:** [Home](../README.md)

**Start here:** [Documentation overview](README.md) · [Getting started](getting-started.md) · [Installation](installation.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md)

**Task guides:** [Task recipes](cookbooks.md) · [Outcome stores](outcome-api.md)

**Examples:** [Choose an example](../examples/README.md) · [Hot dog image classifier](../examples/hotdog/README.md) · [Precedent Desk](../examples/precedent_desk/README.md) · [Wine predictor](../examples/wine/README.md) · [Audio classifier](../examples/audio/README.md) · [Screen-only arena](arena-example.md) · [Demos and videos](demos.md)

**Reference:** [API reference](api.md) · [Design rationale](design.md) · [Venn 1 model card](model-card.md)

**Results and limits:** [Performance and cost](performance.md) · [VennSpace-1](benchmark.md) · [Results and evidence](evidence.md) · **Reproduction** · [Semantic boundary findings](redteam.md) · [Limitations](limitations.md) · [Comparison video methods](same-memory-video.md) · [Comparison notebooks and renderer](../examples/same_memory_video/README.md)

**Project:** [Policies](policies.md) · [Licenses and attribution](licenses.md) · [Contributing to Venn](../CONTRIBUTING.md)

**Maintainers:** [Maintainer guide](maintainers/maintaining.md) · [Documentation site](maintainers/website.md) · [Release checklist](maintainers/release-checklist.md) · [Publishing the Python package](maintainers/pypi.md) · [Publishing the decision head](maintainers/publishing-the-head.md) · [Development evidence](maintainers/development-evidence.md) · [Comparative evaluation protocol](maintainers/evaluation-plan.md) · [License audit notes](maintainers/license-audit.md)

</details>

**Results and limits / Reproduction**
<!-- docs-navigation:end -->

<!-- docs-contents:start -->
<details>
<summary>On this page</summary>

- [CPU contracts and packaging](#cpu-contracts-and-packaging)
- [Fresh headless scientific run](#fresh-headless-scientific-run)
- [Outputs and recovery](#outputs-and-recovery)
- [Trained-head evaluation](#trained-head-evaluation)
- [Upstream client validation](#upstream-client-validation)

</details>
<!-- docs-contents:end -->

There are three distinct checks: executable contracts, headless scientific
measurements, and trained-head results. Passing one does not establish the others.
See [the evidence ledger](evidence.md) for the availability of each claim.

## CPU contracts and packaging

From the checkout, using Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pip check
python scripts/check_docs.py
venn quickstart
venn sweep --rows 19 --batch-size 8
python -m pytest -q
python -m build
```

Tests use deterministic fake embeddings and untrained fixtures. They require
no real reader or head download. Optional adapter tests skip without their
extras. CI also exercises upstream clients and the installed wheel outside the
checkout. Use [getting started](getting-started.md) for CPU-only PyTorch setup.

## Fresh headless scientific run

Use an isolated virtual environment so pinned versions are not shadowed by
user-site packages. CUDA is required by the current fresh-embedding worker.
Choose a visible GPU count; the default is one. Preflight checks devices,
required distributions and output/cache writability before dataset or reader
downloads.

```bash
python3.12 -m venv /tmp/venn-science-env
/tmp/venn-science-env/bin/python -m pip install '.[embedding,benchmark,dev]'
/tmp/venn-science-env/bin/python -m pip check
/tmp/venn-science-env/bin/venn download-embedder
/tmp/venn-science-env/bin/venn benchmark \
  --output /tmp/venn-space-run --cache /tmp/venn-fresh-cache.sqlite --gpus 1
/tmp/venn-science-env/bin/python scripts/compare_reference.py /tmp/venn-space-run/summary.json
/tmp/venn-science-env/bin/venn redteam \
  --output /tmp/venn-boundary-run --cache /tmp/venn-fresh-cache.sqlite --gpus 1
/tmp/venn-science-env/bin/python -m pip freeze > /tmp/venn-science-environment.txt
```

Run these commands from the root of the source checkout: `venn benchmark`
reads its protocol from `benchmarks/vennspace-v1.json`, which the wheel does
not contain, and the comparison script lives in `scripts/`.

Read the pinned reader's upstream terms before downloading. Original datasets
are fetched at the revisions in the frozen protocol. Output/cache paths must be
new for a fresh run. Sharing the newly generated cache with the boundary suite
is explicit reuse within the same run; copying another person's embeddings is
not independent fresh-embedding reproduction.

The reference run used four A100 GPUs. Minimum VRAM, RAM, disk, elapsed time
and download size have not been qualified for other hardware. The code reduces
embedding batches after CUDA out-of-memory errors, but this does not make every
GPU suitable. Changing GPU count preserves task definitions.

## Outputs and recovery

| File | Contents |
| --- | --- |
| `environment.json` | Python/platform, key package versions, CUDA/device names and invoked command |
| `protocol.json` | Frozen task or authored-suite definition |
| `*-splits.json` | Original source rows, labels and normalized-text hashes; no corpus text |
| `*-predictions.npz` | Benchmark predictions, targets and sampled row identities |
| `summary.json` | Geometric metrics, intervals, protocol hash and embedding telemetry |
| `results.json` | Boundary-suite or combined-demo outputs; authored example text may be present |

Completed embedding batches are committed to the SQLite cache. Repeating a
command with the same paths reuses exact cache keys; a changed frozen protocol
is rejected. Final result files are overwritten on a rerun, so use a new output
directory to preserve comparisons. `environment.json` describes the latest
invocation. Archive the protocol, outputs, environment and file checksums after
a successful run. Never describe a cache replay as fresh embedding.

`compare_reference.py` reports each metric's observed value, reference value,
difference, interval and pass/fail. It exits 0 if all ten are within the rounded
reference intervals, 1 for a mismatch, or 2 for invalid input. This reproduces
the documented consistency criterion, not a claim of statistical equivalence.
The reference JSON is transcribed from the existing table, not raw-run precision.

## Trained-head evaluation

The venn-1 head is published on Hugging Face at `markrussinovich/venn-1`
(revision `5e512eb799058d33ca47dd1c80863e7785960160`); `venn download-model`
fetches it and prints the assets path. The
[family example](../examples/registered_family.py) exercises the downloaded
head. There is no public ordinal evaluation runner yet. Follow the
[comparison protocol](maintainers/evaluation-plan.md) before making new model-quality claims.

## Upstream client validation

First terminal, from the checkout with the dev environment active:

```bash
python scripts/contract_server.py --port 8000
```

Second terminal, with that environment active (Node 24 is used by CI):

```bash
python examples/route_ticket.py
python examples/index_vectors.py
cd tests/clients
npm ci --ignore-scripts
npm run contract -- http://127.0.0.1:8000
node playground.mjs http://127.0.0.1:8000
python -m pip install -r requirements.txt
python plugin.py http://127.0.0.1:8000
```

The contract server uses fake embeddings and temporary untrained head fixtures.
These outputs validate interfaces, not semantic routing. The upstream-client
checks restrict SDK traffic to the local origin. Stop the server with Ctrl-C.

<!-- docs-pager:start -->
---

[← Previous: Results and evidence](evidence.md) · [Next: Semantic boundary findings →](redteam.md) · [Documentation home](README.md)
<!-- docs-pager:end -->
