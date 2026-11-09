# Emotion from labeled audio

<!-- docs-navigation:start -->
[Documentation](../../docs/README.md) · [Get started](../../docs/getting-started.md) · [Concepts](../../docs/concepts.md) · [Venn vs Jev](../../docs/venn-and-jev.md) · [Examples](../README.md) · [Task recipes](../../docs/cookbooks.md) · [API](../../docs/api.md) · [Results](../../docs/evidence.md)

<details>
<summary>Browse all documentation</summary>

**Project home:** [Home](../../README.md)

**Start here:** [Documentation overview](../../docs/README.md) · [Getting started](../../docs/getting-started.md) · [Installation](../../docs/installation.md) · [Concepts](../../docs/concepts.md) · [Venn vs Jev](../../docs/venn-and-jev.md)

**Task guides:** [Task recipes](../../docs/cookbooks.md) · [Outcome stores](../../docs/outcome-api.md)

**Examples:** [Choose an example](../README.md) · [Hot dog image classifier](../hotdog/README.md) · [Precedent Desk](../precedent_desk/README.md) · [Wine predictor](../wine/README.md) · **Audio classifier** · [Screen-only arena](../../docs/arena-example.md) · [Demos and videos](../../docs/demos.md)

**Reference:** [API reference](../../docs/api.md) · [Design rationale](../../docs/design.md) · [Venn 1 model card](../../docs/model-card.md)

**Results and limits:** [Performance and cost](../../docs/performance.md) · [VennSpace-1](../../docs/benchmark.md) · [Results and evidence](../../docs/evidence.md) · [Reproduction](../../docs/reproduction.md) · [Semantic boundary findings](../../docs/redteam.md) · [Limitations](../../docs/limitations.md) · [Comparison video methods](../../docs/same-memory-video.md) · [Comparison notebooks and renderer](../same_memory_video/README.md)

**Project:** [Policies](../../docs/policies.md) · [Licenses and attribution](../../docs/licenses.md) · [Contributing to Venn](../../CONTRIBUTING.md)

**Maintainers:** [Maintainer guide](../../docs/maintainers/maintaining.md) · [Documentation site](../../docs/maintainers/website.md) · [Release checklist](../../docs/maintainers/release-checklist.md) · [Publishing the Python package](../../docs/maintainers/pypi.md) · [Publishing the decision head](../../docs/maintainers/publishing-the-head.md) · [Development evidence](../../docs/maintainers/development-evidence.md) · [Comparative evaluation protocol](../../docs/maintainers/evaluation-plan.md) · [License audit notes](../../docs/maintainers/license-audit.md)

</details>

**Examples / Audio classifier**
<!-- docs-navigation:end -->

Fit a linear Choice family on your own recordings, then classify a new clip.
The reader embeds the sound; `group_id` keeps each speaker together during
cross-validation. Audio support is experimental.

From the repository root:

```bash
python -m pip install -e '.[embedding]'
venn download-embedder
python examples/audio/emotion.py --labels /path/to/clips.csv --clip /path/to/new.wav
```

The first run also downloads the Venn head. The reader runs on CPU and caches
each clip's vector. Clips must be mono 16 kHz WAV, at most 30 seconds and 6 MB.
The example does not resample or download recordings.

Create a CSV with these columns; paths are relative to the CSV:

```csv
path,label,speaker
recordings/a-angry-01.wav,angry,speaker-a
recordings/b-happy-01.wav,happy,speaker-b
```

Supply at least five distinct clips per label, at least two labels, and several
speakers represented across labels. The two rows above show the format; add your
remaining recordings before running. Keep the prediction clip out of the CSV.

The script prints JSON with the fitting clip count, cross-validation accuracy
and the new clip's `answer` and `probabilities`. It saves the family under
`VENN_HOME`, so later applications can reload it. See the
[linear-family cookbook](../../docs/cookbooks.md#fit-a-linear-choice-family)
for its request shape and the [comparison guide](../../docs/venn-and-jev.md#sound-carries-information-that-words-leave-out)
for the CREMA-D example.

<!-- docs-pager:start -->
---

[← Previous: Wine predictor](../wine/README.md) · [Next: Screen-only arena →](../../docs/arena-example.md) · [Documentation home](../../docs/README.md)
<!-- docs-pager:end -->
