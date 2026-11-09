# Decisions over stored embeddings: review assets

<!-- docs-navigation:start -->
[Documentation](../../docs/README.md) · [Get started](../../docs/getting-started.md) · [Concepts](../../docs/concepts.md) · [Venn vs Jev](../../docs/venn-and-jev.md) · [Examples](../README.md) · [Task recipes](../../docs/cookbooks.md) · [API](../../docs/api.md) · [Results](../../docs/evidence.md)

<details>
<summary>Browse all documentation</summary>

**Project home:** [Home](../../README.md)

**Start here:** [Documentation overview](../../docs/README.md) · [Getting started](../../docs/getting-started.md) · [Installation](../../docs/installation.md) · [Concepts](../../docs/concepts.md) · [Venn vs Jev](../../docs/venn-and-jev.md)

**Task guides:** [Task recipes](../../docs/cookbooks.md) · [Outcome stores](../../docs/outcome-api.md)

**Examples:** [Choose an example](../README.md) · [Hot dog image classifier](../hotdog/README.md) · [Precedent Desk](../precedent_desk/README.md) · [Wine predictor](../wine/README.md) · [Audio classifier](../audio/README.md) · [Screen-only arena](../../docs/arena-example.md) · [Demos and videos](../../docs/demos.md)

**Reference:** [API reference](../../docs/api.md) · [Design rationale](../../docs/design.md) · [Venn 1 model card](../../docs/model-card.md)

**Results and limits:** [Performance and cost](../../docs/performance.md) · [VennSpace-1](../../docs/benchmark.md) · [Results and evidence](../../docs/evidence.md) · [Reproduction](../../docs/reproduction.md) · [Semantic boundary findings](../../docs/redteam.md) · [Limitations](../../docs/limitations.md) · [Comparison video methods](../../docs/same-memory-video.md) · **Comparison notebooks and renderer**

**Project:** [Policies](../../docs/policies.md) · [Licenses and attribution](../../docs/licenses.md) · [Contributing to Venn](../../CONTRIBUTING.md)

**Maintainers:** [Maintainer guide](../../docs/maintainers/maintaining.md) · [Documentation site](../../docs/maintainers/website.md) · [Release checklist](../../docs/maintainers/release-checklist.md) · [Publishing the Python package](../../docs/maintainers/pypi.md) · [Publishing the decision head](../../docs/maintainers/publishing-the-head.md) · [Development evidence](../../docs/maintainers/development-evidence.md) · [Comparative evaluation protocol](../../docs/maintainers/evaluation-plan.md) · [License audit notes](../../docs/maintainers/license-audit.md)

</details>

**Results and limits / Comparison notebooks and renderer**
<!-- docs-navigation:end -->

The [video and methods page](../../docs/same-memory-video.md) links each beat
to its source demo, numeric evidence and limitations. These assets are a
private release preview. The owner confirmed clearances on November 7, 2026,
and authorized uploading both MP4s to the private `media-v1` release. Public
launch remains on hold.

## Recompose

Use the checkout's existing dependencies; the arena extra provides Pillow
and imageio-ffmpeg. Install Lato (the arena video's font) and DejaVu Sans Mono
system fonts to reproduce the typography.

```bash
pip install -e ".[arena]"
python examples/same_memory_video/render.py --preview --output /tmp/same-memory-preview
python examples/same_memory_video/render.py --output /tmp/same-memory-render
```

The renderer reads `docs/media/same-memory/*-eval.json`, `head-execution.json`
and the displayed function in `head_demo.py`. It makes no network or model
calls, and writes both landscape and vertical cuts, posters,
storyboards, SRT/VTT captions and a SHA-256 manifest. Temporary synthesized
music WAVs are removed after encoding. Source speech is never played.
The motion-led revision follows an inspection of the actual encoded Doom cut:
continuous record flow, retrieval trails, data morphs and processing lanes,
with rhythmic music and synchronized effects. Frames are
rendered throughout each shot, not frozen after an entrance animation.
There are no on-screen publication overlays or fine-print footers; those
disclosures remain on the methods page. All text is at least 48 pixels at
the final output scale.
Frame rendering and encoding are CPU work; there is no GPU workload to
distribute. Existing output files are overwritten.
The landscape cut runs 2:30 and the vertical cut 1:00. As with the arena video,
keep MP4s out of Git and publish approved binaries on GitHub Releases.
Ignored copies under `docs/media` are for local review; do not upload these
drafts until the owner explicitly requests publication. Source links use the intended release URLs, which are not
available until upload. The site builder prefers local MP4s when present and
otherwise uses those release URLs, just as a source-only deployment requires.

The opening sets up the comparison, explains Jev's question-and-choice
interface, and then shows Venn with the identical example on a separate
screen. Input, question, choices, reader, embeddings and head appear in order,
with moving signals and a focus cue guiding the eye before the answer appears.
The input and highlighted choice are authored illustrations, not
recorded API responses. All frames are checked for overlapping text, padded
answer-box and node containment, metric-column bounds and separation from
the reserved narration area. Answer choices are stacked in both formats.
Narration uses one cool-blue regular-weight caption block, visually distinct
from graphic labels, without sentence-ending periods. Connected title lines
share the same font and color; redundant explanation below the architecture
is removed. Site caption tracks are opt-in, not a second default overlay
over the video's existing captions. Score error is labeled in points
with "Lower is better", without an on-screen metric acronym.
The three examples then show why the difference
matters: historical outcomes, learned label conventions and reusable custom
heads. The audio beat and its closing claims are removed from both cuts.
Its 20 landscape seconds and 16 portrait seconds are redistributed to reading
time without lengthening either cut. A subsequent four-second landscape trim
removes the summary animation and shortens the end card, not the explanations.
Architecture stages now unfold over eight
seconds; the code highlight makes a ten-second pass and holds the completed
registration for four seconds. The filing gallery reveals companies one at a
time, then connects them to the shared recorded Business Services label.
Each new dataset and target is introduced before its comparison. Complaint
label grouping precedes outcome retrieval; its wipe affects only that lookup,
and the vertical cut restores it before closing. SEC fitting precedes its
separately labeled test sets, and the company gallery does not attribute a
linear-head prediction to retrieved neighbors. Wine preparation is explicitly
a separate historical comparison, and price shows its distinct cached vectors.

The code scene is taken verbatim from `register_head` in `head_demo.py`.
It was executed on the original fitting notes using their existing
SentenceSimilarity cache, not a mock API. The execution record includes
the source hash and head checksum; the renderer refuses a stale record.
The replay is compressed and does not display a real-time clock.
To execute it with your own local fitting notes and prepared embedding cache:

```bash
python examples/same_memory_video/head_demo.py \
  --examples /path/to/fitting-examples.json \
  --model-dir /path/to/venn-head \
  --cache /path/to/cache.sqlite \
  --output /tmp/head-execution.json
```

Writing to `docs/media/same-memory/head-execution.json` would replace the
tracked record that the renderer uses.

The examples file is a JSON list of `{"state": "note text", "points": 88}`
records. The cache must already contain their pinned SentenceSimilarity
embeddings; cache misses fail instead of starting another reader run.
This only fits the head. It does not rerun or replace the held-out studies.
Do not treat a run on different examples as reproduction of the video results.

The three active notebooks can be opened in Jupyter from this checkout, or their code
cells can be executed without Jupyter:

```bash
python - <<'PY'
import json
from pathlib import Path

for beat in ("history", "sec", "wine"):
    notebook = json.loads(Path(f"examples/same_memory_video/{beat}.ipynb").read_text())
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), beat, "exec"), namespace)
PY
```

These notebooks verify exported **point estimates**, not end-to-end study
reproduction, original wall-clock performance or bootstrap intervals. Numeric
exports carry original source hashes. The SEC notebook reports the failed
primary gate rather than promoting confirmation. The archived sound notebook
and eval file retain the earlier draft's provenance but are not used by the
current renderer. No new fit is run during rendering.

The public export deliberately excludes narratives, tasting notes, source
audio, raw vectors, checkpoints and credentials. Waveform envelopes retain
CREMA-D's ODbL/DbCL terms and credit, as documented on the methods page.
Keep the methods page's local-review and publication-hold notices until release is authorized.

<!-- docs-pager:start -->
---

[← Previous: Comparison video methods](../../docs/same-memory-video.md) · [Next: Policies →](../../docs/policies.md) · [Documentation home](../../docs/README.md)
<!-- docs-pager:end -->
