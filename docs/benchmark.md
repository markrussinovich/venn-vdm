# VennSpace-1

<!-- docs-navigation:start -->
[Documentation](README.md) · [Get started](getting-started.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md) · [Examples](../examples/README.md) · [Task recipes](cookbooks.md) · [API](api.md) · [Results](evidence.md)

<details>
<summary>Browse all documentation</summary>

**Project home:** [Home](../README.md)

**Start here:** [Documentation overview](README.md) · [Getting started](getting-started.md) · [Installation](installation.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md)

**Task guides:** [Task recipes](cookbooks.md) · [Outcome stores](outcome-api.md)

**Examples:** [Choose an example](../examples/README.md) · [Hot dog image classifier](../examples/hotdog/README.md) · [Precedent Desk](../examples/precedent_desk/README.md) · [Wine predictor](../examples/wine/README.md) · [Audio classifier](../examples/audio/README.md) · [Screen-only arena](arena-example.md) · [Demos and videos](demos.md)

**Reference:** [API reference](api.md) · [Design rationale](design.md) · [Venn 1 model card](model-card.md)

**Results and limits:** [Performance and cost](performance.md) · **VennSpace-1** · [Results and evidence](evidence.md) · [Reproduction](reproduction.md) · [Semantic boundary findings](redteam.md) · [Limitations](limitations.md) · [Comparison video methods](same-memory-video.md) · [Comparison notebooks and renderer](../examples/same_memory_video/README.md)

**Project:** [Policies](policies.md) · [Licenses and attribution](licenses.md) · [Contributing to Venn](../CONTRIBUTING.md)

**Maintainers:** [Maintainer guide](maintainers/maintaining.md) · [Documentation site](maintainers/website.md) · [Release checklist](maintainers/release-checklist.md) · [Publishing the Python package](maintainers/pypi.md) · [Publishing the decision head](maintainers/publishing-the-head.md) · [Development evidence](maintainers/development-evidence.md) · [Comparative evaluation protocol](maintainers/evaluation-plan.md) · [License audit notes](maintainers/license-audit.md)

</details>

**Results and limits / VennSpace-1**
<!-- docs-navigation:end -->

<!-- docs-contents:start -->
<details>
<summary>On this page</summary>

- [V1 reference leaderboard](#v1-reference-leaderboard)
- [Independent source-only reproduction](#independent-source-only-reproduction)
- [Compare a new run](#compare-a-new-run)

</details>
<!-- docs-contents:end -->

VennSpace-1 v1 freezes geometric vector-space tasks, not a composite model
ranking. The protocol is `benchmarks/vennspace-v1.json`. Its seed, exact source
revisions, profile/prompt, sample counts, algorithms, scoring and uncertainty
are fixed before measurement. Changing these requires a new benchmark version.

The loader downloads Banking77 and CLINC from their original pinned sources.
Source text, input vectors and caches remain local. Normalized text is deduplicated
within and across official train/test splits, with training taking precedence.
The CLINC out-of-scope class is explicitly excluded. Split manifests contain
original row numbers, labels and hashes, not corpus text.

| Task | Frozen protocol | Score and interval |
| --- | --- | --- |
| Nearest reference | 20 distinct official training examples/intent; 20 test queries/intent | Top-1 accuracy, 2,000 intent-label cluster bootstraps |
| Small membership | 2,000 sets of seven same-intent training examples and one intruder; shuffled positions | Leave-one-out centroid odd-one-out accuracy, Wilson 95% CI |
| Large membership | 200 sets: 190 distinct training examples from five intents plus ten from other intents | Recall@10 using fifth-neighbor squared distance, 2,000 set bootstraps |
| Population shift | 64 null and 64 changed-mixture trials; 48 test examples/population, ten intents, no within-trial overlap | RBF-MMD rejection rate at alpha .05; 199 permutations; Wilson 95% CI |

Membership uses training-split inputs only as a **fixed evaluation corpus**:
these geometric tasks fit no classifier, head or threshold. Nearest-reference
queries and drift populations use the official test split. Intervals for
membership are conditional on this finite corpus and the sampled sets; sets
can share underlying rows. They are not independent population-generalization
or probability-calibration guarantees.

The population-shift null fixes the same balanced intent counts in both
populations and samples disjoint rows within each trial. Its zero observed
alarms and Wilson interval describe these 64 stratified finite-corpus trials,
not IID operational false-alarm calibration at alpha .05. Trials may reuse
corpus rows; deployment mixtures and sampling assumptions require separate
validation.

The EmbeddingGemma profile uses Classification, text-only towers, the pinned
reader revision, float16 cache storage and normalized float32 replay. These
geometric tasks need no head weights or training pipeline.

```bash
python -m pip install ".[embedding,benchmark,dev]"
venn download-embedder
venn benchmark \
  --output /path/to/new/run --cache /path/to/exact-reader-cache.sqlite --gpus 1
```

Run it from a source checkout: the protocol, `benchmarks/vennspace-v1.json`,
is not in the wheel. `venn benchmark` finds it beside an editable install or in
the current directory; `--protocol` names another file.

Protocol and split manifests precede embedding/measurement. Cache hits are
reused exactly; GPU workers only encode missing keys and commit completed
batches. Output includes all numeric predictions/targets, sampled row IDs,
permutation p-values, summary CIs and cache telemetry. The loader never
downloads a redistributed copy of the data from this project.
New runs also record the available Git source revision, dirty-worktree flag
and local Python implementation SHA-256 fingerprints in `environment.json`
and `summary.json`. Without a Git checkout, revision/dirty fields are null;
file fingerprints still identify the available code. Historical reference
artifacts are not retroactively assigned an implementation identity.

The leaderboard reports every task, including weak drift power; there is no
single weighted score, post-hoc threshold or discarded losing task. These are
measurements of the published benchmark protocol.

## V1 reference leaderboard

Local reference run; all percentages show 95% intervals.

| Dataset | Nearest reference | 7+1 accuracy | 190+10 recall@10 | Null alarm rate | Shift detection |
| --- | --- | --- | --- | --- | --- |
| Banking77 | 92.79 [91.36,94.09] | 99.90 [99.64,99.97] | 96.05 [95.25,96.80] | 0.00 [0.00,5.66] | 92.19 [82.98,96.62] |
| CLINC | 92.20 [90.70,93.80] | 98.45 [97.81,98.91] | 82.45 [80.95,83.90] | 0.00 [0.00,5.66] | 89.06 [79.10,94.60] |

Sources: [PolyAI Banking77](https://huggingface.co/datasets/PolyAI/banking77)
(CC BY 4.0) and [CLINC OOS](https://huggingface.co/datasets/clinc/clinc_oos)
(CC BY 3.0). See [licenses](licenses.md) and the upstream papers and cards
for authorship and attribution.

## Independent source-only reproduction

The frozen suite was rerun unchanged in a fresh isolated environment on four
A100 GPUs with fresh embeddings. All ten geometric and all sixteen
semantic-suite aggregates fell within their reference 95% intervals. CLINC
nearest accuracy was 92.17% versus 92.20%, one different answer among 3,000
queries.

## Compare a new run

The [reference JSON](../benchmarks/vennspace-v1-reference.json) transcribes the
rounded table above and records the frozen protocol SHA-256. After evaluation:

```bash
python scripts/compare_reference.py /path/to/new/run/summary.json
```

This checks all ten point estimates against the published rounded intervals;
it is a consistency check, not a test of statistical equivalence. Missing or
nonfinite metrics and incompatible protocol hashes fail.

Protocol-hash compatibility does not check implementation identity: inspect
the new provenance separately; missing fingerprints in historical references
remain a limitation, not a reason to change the frozen comparator criterion.
See the [evidence ledger](evidence.md) for artifact availability and the
[reproduction guide](reproduction.md) for environment and output details.

<!-- docs-pager:start -->
---

[← Previous: Performance and cost](performance.md) · [Next: Results and evidence →](evidence.md) · [Documentation home](README.md)
<!-- docs-pager:end -->
