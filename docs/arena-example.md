# A screen-only arena application

<!-- docs-navigation:start -->
[Documentation](README.md) · [Get started](getting-started.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md) · [Examples](../examples/README.md) · [Task recipes](cookbooks.md) · [API](api.md) · [Results](evidence.md)

<details>
<summary>Browse all documentation</summary>

**Project home:** [Home](../README.md)

**Start here:** [Documentation overview](README.md) · [Getting started](getting-started.md) · [Installation](installation.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md)

**Task guides:** [Task recipes](cookbooks.md) · [Outcome stores](outcome-api.md)

**Examples:** [Choose an example](../examples/README.md) · [Hot dog image classifier](../examples/hotdog/README.md) · [Precedent Desk](../examples/precedent_desk/README.md) · [Wine predictor](../examples/wine/README.md) · [Audio classifier](../examples/audio/README.md) · **Screen-only arena** · [Demos and videos](demos.md)

**Reference:** [API reference](api.md) · [Design rationale](design.md) · [Venn 1 model card](model-card.md)

**Results and limits:** [Performance and cost](performance.md) · [VennSpace-1](benchmark.md) · [Results and evidence](evidence.md) · [Reproduction](reproduction.md) · [Semantic boundary findings](redteam.md) · [Limitations](limitations.md) · [Comparison video methods](same-memory-video.md) · [Comparison notebooks and renderer](../examples/same_memory_video/README.md)

**Project:** [Policies](policies.md) · [Licenses and attribution](licenses.md) · [Contributing to Venn](../CONTRIBUTING.md)

**Maintainers:** [Maintainer guide](maintainers/maintaining.md) · [Documentation site](maintainers/website.md) · [Release checklist](maintainers/release-checklist.md) · [Publishing the Python package](maintainers/pypi.md) · [Publishing the decision head](maintainers/publishing-the-head.md) · [Development evidence](maintainers/development-evidence.md) · [Comparative evaluation protocol](maintainers/evaluation-plan.md) · [License audit notes](maintainers/license-audit.md)

</details>

**Examples / Screen-only arena**
<!-- docs-navigation:end -->

<!-- docs-contents:start -->
<details>
<summary>On this page</summary>

- [Watch the arena](#watch-the-arena)
- [Install and check the application](#install-and-check-the-application)
- [Run the native Venn image path](#run-the-native-venn-image-path)
- [Try other hardware, readers and levels](#try-other-hardware-readers-and-levels)
- [Compile the memory into a head](#compile-the-memory-into-a-head)
- [Outputs and repetition](#outputs-and-repetition)

</details>
<!-- docs-contents:end -->

The [arena example](../examples/arena.py) shows how to build an application
around Venn's outcome API and an ordinary vector memory. Venn returns a move,
per-move utilities and evidence, and the screen's vector. The client steps the
game, controls exploration and saves each completed game's discounted returns.
The reader is never trained. With `--head`, the client also compiles the memory
into a small head through the same service and evaluates it on its own.

It uses ViZDoom with its bundled Freedoom assets, not commercial game assets.
The app supports bundled scenarios and custom `.cfg` levels with discrete
buttons. A move presses one button; continuous controls and arbitrary action
combinations need a different client. Third-party levels and readers have
their own terms.

## Watch the arena

<!-- video: venn-arena -->

https://github.com/user-attachments/assets/b6e0a5da-d703-47db-8fc4-9a387d74b7d5

[Download the video](https://github.com/markrussinovich/venn-vdm/releases/download/media-v1/venn-arena.mp4)
or read [what the video shows and how it was measured](demos.md#screen-only-arena-video).

## Install and check the application

Use Python 3.12 and a clean environment without the unrelated `venn` package:

```bash
python -m pip install ".[arena,embedding]"
python examples/arena.py --smoke-test --output /tmp/venn-arena-smoke
```

The smoke test takes four game steps through `OutcomeService` using simple
screen descriptors. It downloads no reader, learns no semantic policy and
reports no evaluation score. This checks the game/API plumbing, not gameplay
quality. Choose a new output directory for each invocation.

## Run the native Venn image path

The video uses a frozen pinned reader, thirds of each screen and app-owned
exact FAISS GPU memory. This command selects those runtime components:

```bash
python examples/arena.py --reader-mode native --device cuda:0 \
  --dtype bfloat16 --compile-norms --index-device cuda:0 \
  --games 128 --decisions 431173 --level defend_the_center \
  --seed 0 --eval-episodes 50 --record --download-reader \
  --output /tmp/venn-arena-native
```

GPU indexing requires a separately installed FAISS GPU build compatible with
your CUDA environment; the default `faiss-cpu` distribution does not provide it.
Use `--index-device cpu` to keep the store on the host. Smaller GPUs may need
fewer games, an eager reader without `--compile-norms`, or a smaller budget.
There is no automatic batch-size change hidden in the results.

`--download-reader` explicitly permits fetching the pinned upstream reader.
Otherwise the reader must already be available in the local Hub cache.
The native path calls `OutcomeService.decide_batch` on raw uint8 screenshots,
so Venn performs preprocessing, encoding, evidence lookup and action selection.
It uses the existing `venn-1` image tag and does not require fitted heads.

This example is a reimplementation of the application shown in the video.
Instrumentation, evaluation scheduling and recording differ, so matching the
reader and settings does not guarantee the same learning trajectory or scores.

## Try other hardware, readers and levels

The portable path accepts an image-capable SentenceTransformer reader on CPU
or GPU. For example, substitute an immutable commit for a CLIP image encoder:

```bash
python examples/arena.py --reader-mode portable --device cpu --dtype float32 \
  --model sentence-transformers/clip-ViT-B-32 --revision FULL_40_CHARACTER_COMMIT \
  --layout whole --games 4 --decisions 20000 --level basic \
  --eval-episodes 20 --download-reader --record --output /tmp/venn-arena-clip
```

Replace the revision placeholder; a branch or floating tag is rejected.
Image support and the reader's access requirements must be checked separately.
The portable path uses the selected model's ordinary image processor and
passes tagged vectors to Venn. It does not claim the native GPU preprocessing
protocol or native performance. CPU float32 is supported; CPU float16 is not.
CPU runs are much slower than the native GPU path: in two short checks on a
96-core host shared with other jobs, this CLIP command made about one to one
and a half decisions a second, so its 20,000 decisions take several hours.

Every portable reader gets a distinct `experimental:` profile derived from
its model and preprocessing protocol, with its pinned revision, dimensions
and field order. Its vectors cannot be relabeled as `venn-1` or mixed with
another store. These tags are supported only by outcome-store geometry;
they do not enable other readers for the trained head, standard text serving
or the native image path.

Change `--level` to another bundled scenario or an existing `.cfg` file.
Set `--episode-timeout` and `--frame-skip` explicitly for the intended task.
The reward function comes from the level; maximum discounted returns need
not be a good utility estimator for every task.

## Compile the memory into a head

`--head awr` (or `--head bc`) calls `OutcomeService.register_family` on the
final memory, saves the head as `head.pt`, and runs the final evaluation twice:
once answered by the memory and once answered by the head alone, with the store
never consulted. `--head-every 25600` additionally re-registers the head every
25,600 training decisions, warm-started from the previous one, and serves it
during training from the first registration on; the memory keeps being written
either way. The option vectors are text embeddings of the button names from the
same frozen reader, so the head has 49,153 parameters for the thirds layout.

```bash
python examples/arena.py --reader-mode native --device cuda:0 \
  --dtype bfloat16 --compile-norms --index-device cuda:0 \
  --games 128 --decisions 431173 --level defend_the_center \
  --seed 0 --eval-episodes 50 --head awr --download-reader \
  --output /tmp/venn-arena-head
```

`result.json` then has `head` with the objective, SHA-256, parameter and case
counts, registration seconds, fitting losses and the head-only evaluation.

In the development runs behind the video (the head study), heads were compiled
from five independent 431,173-decision memories (about 380,000 cases each) and
scored over 50 fresh games per memory, 250 games in total, with the memory and
heads frozen. The memory row is that study's same-day memory-only control:

| Served by (head study) | Kills per game | Against the memory (paired 95% interval) |
| --- | ---: | --- |
| Memory, k = 11 | 6.20 | |
| Head, `awr` objective, 49,153 parameters | 8.36 | +2.16 [+1.54, +2.95] |
| Head, `bc` objective, 49,153 parameters | 8.26 | +2.06 [+1.26, +2.78] |
| Head compiled from a raw-pixel memory | 3.92 | |

The memories themselves were built in an earlier study of five runs with 50
games each, which also measured the controls. The video's chart takes its
memory, raw-pixel and random bars from it (see
[arena measurement details](demos.md#arena-footage-and-measurement-details)):

| Played by (earlier five-run study) | Kills per game | Score per game (kills minus the one death) |
| --- | ---: | ---: |
| Memory, k = 11 | 6.17 | 5.17 |
| Memory of raw pixels | 4.94 | 3.94 |
| Random moves | 1.31 | 0.31 |

The research report compared the pixel head's 3.92 kills with the pixel
memory's 3.94, which is a score per game; in kills the pixel memory scored
4.94, so the pixel head is about one kill below its memory.

Both heads beat the memory on every one of the five seeds. Registration took
about 55 seconds per memory on one A100, including the label pass through the
adapter, and a head answers a single screen in under half a millisecond. The
same bilinear form trained to regress outcomes instead of imitating the
store's answer scored 5.18, so the objective, not the head's size, is what
matters. These numbers come from the research runs, not from this example
script; an independent run with this command is a new experiment.

Re-registering during play did not add to this. In the same study, a loop
that served the head from the start of a further 431,173 decisions and
re-registered it every 25,600 ended at 6.2 kills against 6.3 for the
memory-only loop with the same budget. The head opened at the compiled
head's level and converged to the memory's: its agreement with the memory's
greedy move rose from 0.49 to 0.70 across the re-registrations, because the
`awr` objective imitates the moves the head itself had taken. Serving a head
compiled from a young store was worse: five loops that registered their first
head after 25,600 decisions from about 19,000 cases all collapsed to a single
move. Use `--head-every` with those results in mind; compiling a mature
memory once with `--head` is the demonstrated use.

## Outputs and repetition

`result.json` records the model protocol, hardware choices, seed, budget,
completed training episodes, online greedy-window curve and separate final
evaluations. `progress.json` reports progress during training. `memory.npz`
and `memory.json` retain the app-owned vectors and maximum returns; use
`--load-memory /path/to/previous/output` to start fresh games with that memory.
Action layout and complete representation tags must match. Unfinished games
and policy RNG are not restored, so this is memory reuse, not exact resumption.
`--record` writes `evaluation-first.mp4` from the first final evaluation game;
with `--head`, it also writes `evaluation-first-head.mp4` from the first game
of the head-only evaluation.

Store growth is not capped. Memory consumption scales with completed unique
state/action cases and vector dimension; monitor RAM/VRAM and reduce the
budget on smaller hardware. An interrupted run may have a progress report
but no final memory checkpoint. Model startup, encoding, training and final
evaluation costs must be reported separately when comparing hardware.

For repeated runs, use distinct `--seed` and `--output` values. With multiple
GPUs, assign independent runs to different `--device cuda:N` and
`--index-device cuda:N` pairs. Each GPU can run a separate seed if it has
enough memory. The app does not aggregate runs. Changed hardware, readers, budgets or levels produce new experiments, not
automatic reproductions of the video's claims.

<!-- docs-pager:start -->
---

[← Previous: Audio classifier](../examples/audio/README.md) · [Next: Demos and videos →](demos.md) · [Documentation home](README.md)
<!-- docs-pager:end -->
