<img src="docs/media/venn-logo.svg" alt="Venn logo: two overlapping circles" width="88" align="right">

# Venn

<!-- docs-navigation:start -->
[Documentation](docs/README.md) · [Get started](docs/getting-started.md) · [Concepts](docs/concepts.md) · [Venn vs Jev](docs/venn-and-jev.md) · [Examples](examples/README.md) · [Task recipes](docs/cookbooks.md) · [Adapters](docs/adapters.md) · [API](docs/api.md) · [Results](docs/evidence.md)

<details>
<summary>Browse all documentation</summary>

**Project home:** **Home**

**Start here:** [Documentation overview](docs/README.md) · [Getting started](docs/getting-started.md) · [Installation](docs/installation.md) · [Concepts](docs/concepts.md) · [Venn vs Jev](docs/venn-and-jev.md)

**Task guides:** [Task recipes](docs/cookbooks.md) · [Store adapters](docs/adapters.md) · [Outcome stores](docs/outcome-api.md)

**Examples:** [Choose an example](examples/README.md) · [Hot dog image classifier](examples/hotdog/README.md) · [Precedent Desk](examples/precedent_desk/README.md) · [Wine predictor](examples/wine/README.md) · [Audio classifier](examples/audio/README.md) · [Screen-only arena](docs/arena-example.md) · [Demos and videos](docs/demos.md)

**Reference:** [API reference](docs/api.md) · [Design rationale](docs/design.md) · [Venn 1 model card](docs/model-card.md)

**Results and limits:** [Performance and cost](docs/performance.md) · [VennSpace-1](docs/benchmark.md) · [Results and evidence](docs/evidence.md) · [Reproduction](docs/reproduction.md) · [Semantic boundary findings](docs/redteam.md) · [Limitations](docs/limitations.md) · [Comparison video methods](docs/same-memory-video.md) · [Comparison notebooks and renderer](examples/same_memory_video/README.md)

**Project:** [Policies](docs/policies.md) · [Licenses and attribution](docs/licenses.md) · [Contributing to Venn](CONTRIBUTING.md)

**Maintainers:** [Maintainer guide](docs/maintainers/maintaining.md) · [Documentation site](docs/maintainers/website.md) · [Release checklist](docs/maintainers/release-checklist.md) · [Publishing the Python package](docs/maintainers/pypi.md) · [Publishing the decision head](docs/maintainers/publishing-the-head.md) · [Development evidence](docs/maintainers/development-evidence.md) · [Comparative evaluation protocol](docs/maintainers/evaluation-plan.md) · [License audit notes](docs/maintainers/license-audit.md)

</details>

**Project / Home**
<!-- docs-navigation:end -->

<!-- docs-contents:start -->
<details>
<summary>On this page</summary>

- [What you can do](#what-you-can-do)
- [Venn or Jev?](#venn-or-jev)
- [Install](#install)
- [Try it in a minute](#try-it-in-a-minute)
- [Ask questions about text](#ask-questions-about-text)
- [Learn more](#learn-more)
- [Watch it learn to play](#watch-it-learn-to-play)
- [Project](#project)

</details>
<!-- docs-contents:end -->

**A Vector Decision Model: typed decisions over reusable embeddings.**

The [documentation website](https://markrussinovich.github.io/venn-vdm/) includes
the guides, examples, API reference, results and demo videos. The same
[documentation](docs/README.md) is browsable here or as a
[local searchable website](docs/maintainers/website.md#preview-locally).

Venn is a Vector Decision Model (VDM), a small decision layer that answers
structured questions over embeddings rather than generating text. Give it a
vector, or text that its frozen reader embeds, and it returns a typed answer:
which team owns this ticket, does this passage belong with these, has the mix
of incoming requests shifted, which indexed records match a new policy. Because
it reuses the vectors you have already computed, new questions can run through
the small decision layer without another reader pass. The generic head is
3.15 MB, and existing-vector decisions run on CPU without loading the reader.

When a lookup is not enough, the same API trains a small custom head on your
own data and serves it under the question you were already asking. Your
application code does not change: the question names the head, and the answer
comes back in the same typed shape. In the Doom arena, a 49,000-parameter head
compiled from the game's outcome store played with the store switched off and
averaged 8.4 kills per game, against 6.2 for the store it learned from.

The standard text path uses the pinned `venn-1` reader. The outcome API also
accepts explicitly tagged vectors from other readers; see
[profiles and compatibility](docs/concepts.md#profiles-and-compatibility).

## What you can do

- **Route and classify.** Ask Choice, Score and Noul (yes/no) questions about
  text or vectors, with an explicit "none of these" option. Requests use the
  [Decisions API](https://docs.typesafe.ai/introduction) format, with compatibility
  checked against [these client versions](https://github.com/markrussinovich/venn-vdm/blob/main/tests/clients/README.md).
- **Decide from recorded outcomes.** Point Venn at your own vector store of past
  cases and their outcomes. It picks the option whose similar past cases went
  best, and your application writes new outcomes back. No trained head is needed.
- **Train custom heads through the same API.** Four kinds, all through
  `register_family` (or `PUT /families/{name}` over HTTP):
  - a linear Choice head (`form="linear"`), a multinomial logistic regression on
    your labeled vectors and the best default for plain classification;
  - the residual Choice head, which can also answer "none of these";
  - a numeric head (`objective="regress"`) that predicts a score with an 80%
    interval, from examples or a store's outcomes;
  - a head compiled from an outcome store that answers `nearest_outcome` without
    a lookup.

  Choice heads train on every supplied example by default; an explicit split
  reserves selection and report rows. They choose their settings by
  cross-validation, grouped by `group_id` when related examples must stay
  together. Each fits in seconds to minutes on one machine, is saved and reloaded
  with the service, and answers the same question type the rest of your code
  already sends. The reader stays frozen; only the head is trained. See the
  [family cookbook](docs/cookbooks.md#fit-and-reload-a-local-family), the
  [linear family cookbook](docs/cookbooks.md#fit-a-linear-choice-family), the
  [numeric family cookbook](docs/cookbooks.md#fit-a-numeric-family) and the
  [outcome API](docs/outcome-api.md#compile-the-store-into-a-head).
- **Work on collections.** Find the nearest labeled reference, spot the
  intruder in a set, test whether a population has shifted, or stream one
  question over every row of an index.

A few words you will see throughout: a **reader** is the frozen embedding model; a
**profile** pins the reader, its revision and the vector format; a **head** is the
small model that turns vectors and a question into an answer; **Noul** is the
Decisions API's name for a yes/no question.

## Venn or Jev?

Jev evaluates the content and context supplied in a request. Venn lets your
application keep and reuse the representation behind its decisions, through
historical cases, custom predictors and collection operations. Its small local
decision layer also provides an alternative to running a larger open-source
decision model. EmbeddingGemma 2's text-only reader has about 270 million
parameters; image and audio towers are loaded only when needed.

Start with Jev when useful category descriptions and the item's content carry
the answer. Start with Venn when you have useful labeled history, need your own
predictor, or expect repeated decisions over the same embedded collection.
Read [Venn and Jev: when to use which](docs/venn-and-jev.md) for the trade-offs,
examples and results.

<!-- video: venn-same-memory-review -->

https://github.com/user-attachments/assets/ae29eb8c-ee45-46d1-b21c-9b6d9689ccf5

[Watch the comparison](docs/same-memory-video.md) with examples of
historical outcomes, filing conventions and custom predictors.

Venn is a research preview. Probabilities are not calibrated across tasks, and the
"none of these" option is a signal rather than a filter: an off-topic question can
still land on a listed option with high confidence. Read the
[limitations](docs/limitations.md) before acting on a probability.

## Install

Python 3.12 on Linux or Windows. CPU workflows also run on ARM64, including
Ubuntu under WSL and native Windows; see [platform setup](docs/getting-started.md#arm64-and-windows).

```bash
git clone https://github.com/markrussinovich/venn-vdm.git venn
cd venn
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only PyTorch; omit for CUDA
python -m pip install .
```

## Try it in a minute

`venn quickstart` runs on CPU and downloads nothing. It registers a small store
of past support tickets with their outcomes, then routes a new ticket to the
team whose similar tickets went best:

```bash
venn quickstart
```

```json
{
  "answer": "account",
  "options": {
    "account": {"mean": 0.9, "count": 3, "evidence_ids": ["acct-102", "acct-101", "acct-103"]},
    "billing": {"mean": 0.4, "count": 3, "evidence_ids": ["bill-201", "bill-202", "bill-203"]}
  },
  "note": "authored vectors, not text embeddings; nothing was downloaded"
}
```

The same API with your own vectors is in [examples/outcome_store.py](examples/outcome_store.py)
and the [outcome API guide](docs/outcome-api.md).

## Ask questions about text

Text questions use the `venn-1` profile: the EmbeddingGemma 2 reader plus a
3 MB decision head. Install the reader extra, download both and start the server:

```bash
python -m pip install '.[embedding]'
venn download-embedder      # EmbeddingGemma 2 (Apache-2.0), about 1.49 GB of weights
venn serve                  # downloads the venn-1 head on first run
```

In another terminal:

```bash
curl http://127.0.0.1:8000/api/alpha/decisions -H 'Content-Type: application/json' -d '{
  "model": "venn-1",
  "state": "I cannot sign in",
  "questions": {"team": {"type": "choice", "instructions": "Who owns this ticket?",
                          "criteria": {"account": "Login or access", "billing": "Payments", "no_match": null}}}
}'
```

The response names a team, gives its probability and says whether Venn
abstained. The first text request also loads the reader, which takes a few
seconds. Later requests avoid that cold load; fresh content still needs embedding. [examples/route_ticket.py](examples/route_ticket.py) does the same from
Python. The [API reference](docs/api.md) covers Score, Noul, the space-level
question types, the JavaScript client and the MCP bridge.

## Learn more

- [Documentation overview](docs/README.md): choose a path through concepts, guides, examples and reference.
- [Getting started](docs/getting-started.md): install and obtain a first result without model downloads.
- [Installation](docs/installation.md): CPU PyTorch, Windows/ARM64 and optional extras.
- [Example catalog](examples/README.md): pick a complete workflow by its prerequisites.
- [Task recipes](docs/cookbooks.md) and [outcome stores](docs/outcome-api.md): build common operations.
- [Concepts](docs/concepts.md) and [API reference](docs/api.md): understand the model and request contracts.
- [Performance and cost](docs/performance.md), [evidence](docs/evidence.md) and [limitations](docs/limitations.md): assess the results and their scope.

## Watch it learn to play

<!-- video: venn-arena -->

https://github.com/user-attachments/assets/42b3ddeb-0c9a-42f3-ad5a-e1d9d8734088

The arena example drives a ViZDoom player with nothing but Venn's outcome API
and a FAISS store. Venn embeds each screen, looks up how similar moments turned
out for each move, and returns the move with the best recorded outcome.
Finished games are written back to the store. The reader is never trained.

One more call compiles that store into a head of 49,153 parameters that answers
the same question with no lookup. Served alone, the head averaged 8.4 kills per
game over 250 scored games against 6.2 for the memory it learned from.
[Run it yourself](docs/arena-example.md) or read
[what the video shows](docs/demos.md#screen-only-arena-video).

For text, the [Precedent Desk example](examples/precedent_desk/README.md) points
Venn at a support desk's 10,003 closed tickets and routes new ones from them,
with precedents, a queue-fit flag and a one-pass backlog sweep. On the same
3,080 public test tickets the closed tickets scored 93.6% against 79.7% for a
matched run of Jev 1.13, and the pattern repeats on a second dataset.
[Video, 2 minutes 36 seconds](https://github.com/markrussinovich/venn-vdm/releases/download/media-v1/venn-precedent-desk.mp4).

## Project

- Code is MIT-licensed. The EmbeddingGemma 2 reader and the `venn-1` head have
  their own terms; see [licenses](docs/licenses.md).
- The distribution is named `venn-vdm` and the import name is `venn`. Do not
  install it alongside the unrelated `venn` package.
- Cite with [CITATION.cff](CITATION.cff). See [contributing](CONTRIBUTING.md)
  and [security](SECURITY.md).

<!-- docs-pager:start -->
---

[Next: Documentation overview →](docs/README.md) · [Documentation home](docs/README.md)
<!-- docs-pager:end -->
