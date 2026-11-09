# Precedent Desk: your closed tickets are the model

<!-- docs-navigation:start -->
[Documentation](../../docs/README.md) · [Get started](../../docs/getting-started.md) · [Concepts](../../docs/concepts.md) · [Venn vs Jev](../../docs/venn-and-jev.md) · [Examples](../README.md) · [Task recipes](../../docs/cookbooks.md) · [API](../../docs/api.md) · [Results](../../docs/evidence.md)

<details>
<summary>Browse all documentation</summary>

**Project home:** [Home](../../README.md)

**Start here:** [Documentation overview](../../docs/README.md) · [Getting started](../../docs/getting-started.md) · [Installation](../../docs/installation.md) · [Concepts](../../docs/concepts.md) · [Venn vs Jev](../../docs/venn-and-jev.md)

**Task guides:** [Task recipes](../../docs/cookbooks.md) · [Outcome stores](../../docs/outcome-api.md)

**Examples:** [Choose an example](../README.md) · [Hot dog image classifier](../hotdog/README.md) · **Precedent Desk** · [Wine predictor](../wine/README.md) · [Audio classifier](../audio/README.md) · [Screen-only arena](../../docs/arena-example.md) · [Demos and videos](../../docs/demos.md)

**Reference:** [API reference](../../docs/api.md) · [Design rationale](../../docs/design.md) · [Venn 1 model card](../../docs/model-card.md)

**Results and limits:** [Performance and cost](../../docs/performance.md) · [VennSpace-1](../../docs/benchmark.md) · [Results and evidence](../../docs/evidence.md) · [Reproduction](../../docs/reproduction.md) · [Semantic boundary findings](../../docs/redteam.md) · [Limitations](../../docs/limitations.md) · [Comparison video methods](../../docs/same-memory-video.md) · [Comparison notebooks and renderer](../same_memory_video/README.md)

**Project:** [Policies](../../docs/policies.md) · [Licenses and attribution](../../docs/licenses.md) · [Contributing to Venn](../../CONTRIBUTING.md)

**Maintainers:** [Maintainer guide](../../docs/maintainers/maintaining.md) · [Documentation site](../../docs/maintainers/website.md) · [Release checklist](../../docs/maintainers/release-checklist.md) · [Publishing the Python package](../../docs/maintainers/pypi.md) · [Publishing the decision head](../../docs/maintainers/publishing-the-head.md) · [Development evidence](../../docs/maintainers/development-evidence.md) · [Comparative evaluation protocol](../../docs/maintainers/evaluation-plan.md) · [License audit notes](../../docs/maintainers/license-audit.md)

</details>

**Examples / Precedent Desk**
<!-- docs-navigation:end -->

<!-- docs-contents:start -->
<details>
<summary>On this page</summary>

- [What was measured](#what-was-measured)
- [Run it](#run-it)
- [What this does and does not claim](#what-this-does-and-does-not-claim)

</details>
<!-- docs-contents:end -->

A support desk routes tickets to queues. Northwind Bank's card-ops desk has
10,003 closed tickets, each with the queue it was resolved in. This demo points
Venn at those tickets and asks the same question a text-reading decision model
answers, with nothing trained: which queue handles this ticket. The same lookup
returns the precedents, flags a ticket that does not fit its queue, learns a new
queue from one resolved ticket, and sweeps the whole backlog in one pass.

The desk is played by [Banking77](https://huggingface.co/datasets/PolyAI/banking77)
(CC BY 4.0): the training split is the store of closed tickets, the test split
is the incoming stream, and the 77 intents are the queues. It is the one public
dataset with a published Jev number, so the comparison is on the same 3,080
rows. The resolution notes beside each precedent are authored illustrations,
not data. This harness retains the official splits: seven test rows have
normalized-text matches in training, all with the same label. The 13,083
rows contain 13,071 distinct normalized texts. Historical scores are not
deduplicated or rescored; this overlap limits the held-out interpretation.

## What was measured

Every number on screen is read from `results.json`, written by `eval.py` on
CPU from the published vectors. Accuracy is top-1 by queue over all 3,080 test
tickets. Intervals cover test-row variation (1,000-row bootstrap) and which
tickets the store kept (25 store draws), pooled, except for the held-out-queue
intervals described below. The original protocol and thresholds were fixed
before its run; later amendments and exploratory analyses are not all covered
by that registration.

| What | Result |
| --- | ---: |
| Jev 1.13, matched run on the 3,080 test tickets, one fixed prompt over the 77 queue names | 79.7% [78.3, 81.1] |
| Venn, zero-shot: the 77 queue names only, no store | 92.1% [91.1, 93.0] |
| Venn, nearest reference, 5 closed tickets per queue | 91.2% [89.9, 92.4] |
| 10 per queue | 92.3% [91.2, 93.4] |
| 20 per queue (full test split, not the VennSpace-1 cell) | 92.8% [91.8, 93.8] |
| 50 per queue | 93.3% [92.3, 94.3] |
| All 10,003 closed tickets | 93.6% [92.7, 94.4] |
| Fine-tuned classifiers (published, different model family) | about 93.8% |
| Store with 10% / 20% / 30% of queue labels wrong | 84.1% / 74.8% / 65.4% |

Provenance of the numbers: the 92.8% that circulated before this demo is
VennSpace-1's Banking77 nearest-reference row, a 20-references-per-queue,
20-queries-per-queue cell (1,540 queries). The full-split cells measured here
are different numbers: 92.1% zero-shot, 93.6% with all closed tickets, 79.7%
for Jev's matched run. This harness's 20-per-queue point estimate rounds to
the same 92.8%, but uses 3,080 queries and pooled row/store-draw intervals,
not VennSpace-1's 1,540 queries and intent-label bootstrap.

The crossover sentence the design asked for holds at the smallest stop tested:
five resolved tickets per queue already clear Jev's interval. Jev saw none of
the training rows; Venn's store was all of them. Same rows, different
information.

**The zero-shot row is not what it looks like.** Two gates were run before any
claim about it. Replacing the 77 original queue names with fresh one-line
descriptions written without those names drops zero-shot from 92.1% to 69.0%
[67.3, 70.6], below Jev; opaque codes give 0.7%, chance. On queues held out of
the store, the original name alone reaches 67% of a held-out queue's tickets
where a fresh description reaches 32%. The original label strings carry far
more than their meaning, which is what memorized dataset labels look like, and
Banking77 is a standard embedding benchmark. The number the demo treats as
Venn's no-history baseline is therefore the description-based 69.0%; the
92.1% is shown in the app with this caveat and nowhere else.

**A second dataset (later amendment; history curve exploratory).** On CLINC150 (pinned clinc_oos "plus", 4,498 in-scope
test rows, 150 intents), Jev's matched run scored 91.1% [90.2, 91.9] and
Venn zero-shot over the 150 intent names 69.2% [67.8, 70.5]. The history curve
replicates with a later crossover: 87.8 / 90.6 / 92.3 / 93.8 / 94.3% at 5 /
10 / 20 / 50 / all (100) training rows per intent, clearing Jev's interval from
50 per intent; the 11-vote variant reaches 94.6%. So the pattern that holds
on these two measured datasets is "the desk's closed tickets outscore the
decision model once there are enough of them", not "the queue names outscore
it". Item k's history curve and 11-vote result were added after seeing item j;
they are exploratory, not an independently pre-registered confirmation.

| CLINC150, 4,498 test rows | Accuracy |
| --- | ---: |
| Jev 1.13, matched run | 91.1% [90.2, 91.9] |
| Venn zero-shot, intent names only | 69.2% [67.8, 70.5] |
| Nearest reference, 5 / 10 / 20 / 50 / all per intent | 87.8 / 90.6 / 92.3 / 93.8 / 94.3% |

**Label noise and the vote.** With 10 / 20 / 30% of store labels reassigned,
the best-single-reference rule falls to 84.1 / 74.8 / 65.4%, below Jev from
20% on. A vote among the 11 nearest stored tickets holds at 93.7 / 93.7 /
93.6%, the same as with clean labels; the app's "Show the label-noise lines"
checkbox shows both.

**One new experience.** Seven queues were held out of the store entirely (ten
random choices of seven). A ticket from a held-out queue arrives with no
precedent: the nearest-reference margin fell below the abstention threshold on
81% of first arrivals [71, 90] (threshold set for 90% coverage on known
queues; the set-membership flag fired on 94%). Writing that one resolved ticket
back routes the queue's other test tickets at 57% [51, 63]; two tickets 72%,
three 77%, five 81%. The pre-registered gate for using the word "learning"
(one-shot lower bound above 50%, abstention on at least 80% of first arrivals)
passed. The counter never scores the ticket that was written.

That is the historical gate result under the registered cell bootstrap:
70 intent-within-choice cells (ten choices of seven), not ten choice clusters.
Cells share stores and thresholds, and intents can recur across choices.
The interval is descriptive of that cell resampling, not a cluster-aware
generalization guarantee. Whether a choice-cluster or repeated-intent analysis
changes the gate is unmeasured and deferred; thresholds and results are retained.
The abstention gate uses the point rate, not its interval's lower bound.

**Backlog sweep.** One `index_apply` pass over all 13,083 ticket rows takes
0.34 s in process and 0.34 s over loopback HTTP including client parsing
(index upload 6 s, timed separately), on 8 CPU cores. The pre-registered
two-option question ("affected by the FX-fee change / not", threshold 0.9) is a
poor sweep: precision 5%, recall 21%, 9,505 rows sent to review. Re-tagging
the backlog with the 77 queue names and filing the two FX queues, added after
that result and labelled exploratory, finds 361 tickets at 98.1% precision and
98.6% recall. The demo shows the measured row it was registered with and the
exploratory row beside it.

**Registration (app only).** A family head fitted on 20 or 35 closed tickets
per queue, 25 draws each, scored on the 3,080 test tickets against nearest
reference over the same store. These rows were recorded when `register_family`
split the examples 60/20/20 (train/selection/report) within each queue by
default. It now trains on every example unless a `split` is given, so item d
passes that 60/20/20 split explicitly. A one-draw check with the current code
reproduced the report-slice sizes (308 and 539 items) and the `calibrated`
flags; its fits took 323 s and 415 s on a shared machine. The fit times in the
table are from the earlier run:

| Examples per queue | Head | Nearest reference, same store | Head − nearest | `calibrated` | Test ECE | p ≥ 0.95 on the report slice | Fit time (recorded) |
| --- | ---: | ---: | ---: | --- | ---: | --- | ---: |
| 20 | 92.3% | 92.8% | −0.5 [−0.6, −0.4] | never (308 report items) | 0.061 | precision 100%, coverage 95.9% [95.1, 96.7] | 95 s |
| 35 | 92.4% | 93.1% | −0.7 [−0.8, −0.6] | always (539 report items) | 0.056 | precision 100.0%, coverage 96.5% [95.9, 97.1] | 153 s |

No accuracy gain over nearest reference is claimed; the head is half a point
behind it. The `calibrated` flag and the p ≥ 0.95 figures come from the
service's own held-out report slice, which is drawn from the training split;
on the test tickets the same head's ECE is 0.056 to 0.061, ten times the
report-slice value, so an auto-action threshold read off the report slice is an
estimate with an interval and a known gap, never a guarantee.

**The wrong tool, on purpose.** "I was NOT charged twice. One charge, but my
statement shows two lines." Venn's precedents say duplicate charge; a text
reader can read the negation. The closing rule: route by Venn when a precedent
clears the margin threshold, fall through to Jev when it does not, write every
resolution back.

## Run it

```bash
pip install -e ".[embedding,benchmark,dev]"
for f in banking77-vectors banking77-gate-vectors clinc-vectors clinc-train-vectors; do
  curl -LO "https://github.com/markrussinovich/venn-vdm/releases/download/media-v1/$f.npz"
done
python examples/precedent_desk/eval.py --vectors banking77-vectors.npz --output /tmp/precedent-desk \
    --model-dir "$(venn download-model)" --items abcfghijk --check-texts
python examples/precedent_desk/app.py --vectors banking77-vectors.npz \
    --results /tmp/precedent-desk/results.json --model-dir "$(venn download-model)"
```

Then open http://127.0.0.1:8765.

`eval.py` reads the gate and CLINC files from beside `banking77-vectors.npz`
(or `--gate-vectors`, `--clinc-vectors` and `--clinc-train-vectors`). Item h
needs `banking77-gate-vectors.npz`: it measures the 69.0% description baseline
and the label-string gate that qualifies the 92.1% zero-shot figure. Items j and
k need the two CLINC files. The app also reads `banking77-gate-vectors.npz` for
its "fresh queue descriptions" option; without it, that option is disabled and
the app routes with the original names. `--items abcfghijk` writes every number
in the tables above except three: Jev's matched Banking77 run (79.7%), Jev's
CLINC150 run (91.1%) and the registration table.

- **Jev, 79.7%:** item e. It needs `OPENROUTER_API_KEY` and costs about $0.21
  for the 3,080 requests, cached in `<output>/jev-responses.jsonl`. With no key,
  item e reads a complete cache from an earlier run and makes no requests; with
  neither, it writes the third-party 81.0% placeholder, and the app shows that
  81.0% with its source instead of 79.7%. Items run with e first, so
  `--items eabcfghijk` uses the matched run throughout. Pass the cache to the app
  with `--jev-cache` so the Jev column shows the matched answers.
- **Jev on CLINC150, 91.1%:** item j with `OPENROUTER_API_KEY` set; without a
  key, item j measures only Venn's side.
- **Registration:** item d fits 50 family heads (two sizes, 25 draws) and
  scores each on the 3,080 test tickets. One draw of each size took 13 minutes
  in the check above, so the default 25 draws take several hours; `--draws`
  lowers the number of draws.

The items take a few minutes each on 8 CPU cores, except d. On a
release-source checkout, the app fetches ticket texts from the original pinned
Hugging Face revision and checks hashes and labels against the vectors before
caching them in `--state` (default `$VENN_HOME/precedent-desk`, with
`VENN_HOME` defaulting to `~/.cache/venn`) as `tickets.json`; `--tickets`
chooses another path. No upstream ticket texts are shipped in release source.
Typing a ticket of your own needs the pinned reader locally
(`venn download-embedder`).

The vectors file holds EmbeddingGemma 2 vectors (float16, `venn-1` profile,
`Classification` prompt) for the 13,083 Banking77 rows plus the option and
instruction vectors the questions use. It carries Banking77's CC BY 4.0
attribution; it is an explicitly attributed derived-vector release exception
to the no-dataset-text boundary, not permission to redistribute upstream text.
The vectors file is not part of the source package.

New evaluation outputs keep the legacy Banking77 `vectors_sha256` and also
record consumed gate/CLINC vector files, available cached comparison inputs,
head artifacts, source pins and implementation fingerprints. Item j verifies
ordered CLINC hashes, labels and original row IDs against the pinned source.
Item k checks both splits when `--check-texts` is enabled; otherwise it retains
the local-vector-only CPU path and explicitly reports source-row verification
as unchecked. These checks do not remeasure historical results.

## What this does and does not claim

Said: on the same 3,080 public rows, the desk's history scored 93.6% and Jev
79.7%, with the information difference stated; nothing was trained; five
resolved tickets per queue already outscore the decision model; one resolved
ticket of a new kind routes the next ones at 57%, measured on held-out rows;
the same vector answers precedent, queue fit and the backlog sweep.

Not said: that Venn beats Jev at classification in general (JevBench's public
slice is 55% against 82 to 84%); that any of this is zero-shot when the store
is used; "learns" about a written ticket being retrieved; which handling path
works (there is no outcome data; nearest outcome is not in this demo); that an
auto-action threshold is proven or safe; any sweep time without its boundary.
Banking77 rows may be in the reader's training data, as disclosed for
VennSpace-1; labels are expert-assigned, so the public number is an upper bound
on what a desk's own clicked queue field would give (the noise rows are the
honest answer).

<!-- docs-pager:start -->
---

[← Previous: Hot dog image classifier](../hotdog/README.md) · [Next: Wine predictor →](../wine/README.md) · [Documentation home](../../docs/README.md)
<!-- docs-pager:end -->
