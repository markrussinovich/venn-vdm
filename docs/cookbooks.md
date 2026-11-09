# Task recipes

<!-- docs-navigation:start -->
[Documentation](README.md) · [Get started](getting-started.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md) · [Examples](../examples/README.md) · **Task recipes** · [API](api.md) · [Results](evidence.md)

<details>
<summary>Browse all documentation</summary>

**Project home:** [Home](../README.md)

**Start here:** [Documentation overview](README.md) · [Getting started](getting-started.md) · [Installation](installation.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md)

**Task guides:** **Task recipes** · [Outcome stores](outcome-api.md)

**Examples:** [Choose an example](../examples/README.md) · [Hot dog image classifier](../examples/hotdog/README.md) · [Precedent Desk](../examples/precedent_desk/README.md) · [Wine predictor](../examples/wine/README.md) · [Audio classifier](../examples/audio/README.md) · [Screen-only arena](arena-example.md) · [Demos and videos](demos.md)

**Reference:** [API reference](api.md) · [Design rationale](design.md) · [Venn 1 model card](model-card.md)

**Results and limits:** [Performance and cost](performance.md) · [VennSpace-1](benchmark.md) · [Results and evidence](evidence.md) · [Reproduction](reproduction.md) · [Semantic boundary findings](redteam.md) · [Limitations](limitations.md) · [Comparison video methods](same-memory-video.md) · [Comparison notebooks and renderer](../examples/same_memory_video/README.md)

**Project:** [Policies](policies.md) · [Licenses and attribution](licenses.md) · [Contributing to Venn](../CONTRIBUTING.md)

**Maintainers:** [Maintainer guide](maintainers/maintaining.md) · [Documentation site](maintainers/website.md) · [Release checklist](maintainers/release-checklist.md) · [Publishing the Python package](maintainers/pypi.md) · [Publishing the decision head](maintainers/publishing-the-head.md) · [Development evidence](maintainers/development-evidence.md) · [Comparative evaluation protocol](maintainers/evaluation-plan.md) · [License audit notes](maintainers/license-audit.md)

</details>

**Task guides / Task recipes**
<!-- docs-navigation:end -->

<!-- docs-contents:start -->
<details>
<summary>On this page</summary>

- [Choose a recipe](#choose-a-recipe)
- [Route a support ticket](#route-a-support-ticket)
- [Register, stream and delete an index](#register-stream-and-delete-an-index)
- [Fit and reload a local family](#fit-and-reload-a-local-family)
- [Fit a linear choice family](#fit-a-linear-choice-family)
- [Fit a numeric family](#fit-a-numeric-family)
- [Filter retrieved passages](#filter-retrieved-passages)
- [Sweep a real vector store](#sweep-a-real-vector-store)
- [Inspect a finite drift example](#inspect-a-finite-drift-example)

</details>
<!-- docs-contents:end -->

Start with [getting started](getting-started.md) for installation and a first result.
For complete applications, use the [example catalog](../examples/README.md).
Each recipe below states whether
it needs the head, the reader or only synthetic vectors. Run the commands from
the checkout after installation.

## Choose a recipe

| Task | Recipe | Needs |
| --- | --- | --- |
| Ask a typed question about text | [Route a support ticket](#route-a-support-ticket) | Reader, head and running server |
| Stream questions over stored vectors | [Index lifecycle](#register-stream-and-delete-an-index) | Contract server for the authored example |
| Fit and reload a classifier | [Local family](#fit-and-reload-a-local-family) | Head; no reader for the authored example |
| Fit a plain linear classifier | [Linear choice family](#fit-a-linear-choice-family) | Labeled vectors and the service |
| Predict a number | [Numeric family](#fit-a-numeric-family) | Labeled vectors or text; reader for text |
| Use retrieval or drift checks | [Filter passages](#filter-retrieved-passages), [drift example](#inspect-a-finite-drift-example) | Embedding/benchmark extras and CUDA for the combined demo |
| Connect an existing vector store | [Store sweep](#sweep-a-real-vector-store) | A compatible adapter and pinned vectors |

## Route a support ticket

Prerequisites: a running local server with the reader and head downloaded.
Start it following [serve text questions](getting-started.md#serve-text-questions).

```bash
python examples/route_ticket.py --text "I cannot sign in"
```

[The complete script](../examples/route_ticket.py) sends named account/billing
criteria and `no_match: null`. Inspect `answers.team.choice`, `probabilities`
and `venn.calibrated`; route a no-match answer to review. A high probability
alone does not establish task calibration. Use your own held-out examples to
validate the question. The same script can exercise the contract server, but
its fake embeddings do not produce meaningful semantic predictions.

## Register, stream and delete an index

For a download-free interface exercise, run the fixture server in terminal one:

```bash
python scripts/contract_server.py --port 8000
```

In terminal two:

```bash
python examples/index_vectors.py
```

[The script](../examples/index_vectors.py) constructs three synthetic vectors,
registers stable IDs and outcomes, asks a nearest-reference question, consumes
NDJSON in batches of two and deletes the index in a `finally` block. Expect
`consumed_ids: ["a", "b", "c"]`, including the final partial batch.

For real data, supply normalized vectors from the exact pinned profile to the
same operations. Keep the original document IDs so evidence can be retrieved
from its source. Nearest-reference margins describe geometric separation,
not a corpus-wide probability of being out of distribution.

## Fit and reload a local family

Prerequisites: the `venn-1` head, which the example downloads if `--model-dir`
is not given. No reader is needed for this synthetic vector example. Use a
writable state directory:

```bash
export VENN_HOME=/tmp/venn-family-state
python examples/registered_family.py
```

[The complete example](../examples/registered_family.py) makes 30 distinct
synthetic vectors across three labels, fits a family, queries it, reloads the
saved family in a new service and verifies identical answers. It then deletes
the temporary family. Expected split counts are `[30, 0, 0]` (every example
trains; settings come from cross-validation), with
`reloaded_identically: true` and `calibrated: false`. This is lifecycle training,
not evidence of semantic or ordinal quality.

For real fitting, replace synthetic vectors with your labeled, exact-profile
examples and reserve an external evaluation set. The service trains on every
example and chooses its settings by cross-validation; related variants should
share `group_id` so they stay in one fold.
Five examples per label is a fitting minimum. A custom ordinal rubric requires
separate ordering validation. Re-register after changing profile, labels,
instructions or option vectors. The [API](api.md#family-lifecycle) documents
the corresponding HTTP requests and persistence behavior.

## Fit a linear choice family

For a complete local image application, try [Hot dog or not](../examples/hotdog/README.md).
It fits once on supplied training photos, evaluates separate test photos, and
reuses the saved family in a browser. The [audio example](../examples/audio/README.md)
uses the same linear-family path for labeled WAV clips.

For plain classification, register the family with `form="linear"`: a
multinomial logistic regression on your labeled vectors.

```python
service.register_family(
    "emotion",
    [{"state": {"vector": v}, "label": label, "group_id": speaker}
     for v, label, speaker in zip(vectors, labels, speakers)],
    options=[{"label": name, "vector": name} for name in EMOTIONS],
    instructions="Which emotion is the speaker expressing?",
    form="linear",
)
registered = service.families["emotion"]["options"]
answer = service.decide({"state": {"vector": vector}, "questions": {"emotion": {
    "type": "Choice", "family": "emotion", "instructions": "Which emotion is the speaker expressing?",
    "options": [{"label": n, "vector": v} for n, v in zip(EMOTIONS, registered)]}}})
```

Here is how it trains and answers:
- **Training:** it trains on every example. The penalty C is chosen from 0.01
  to 100 by 5-fold cross-validated log loss; `group_id` keeps a speaker, a
  customer or a duplicate set in one fold.
- **Options:** `split`, `cv` and `class_weight` work as for the residual head.
- **Answering:** answer with the registered option vectors, as above. Options
  re-embedded from their labels may not match exactly and are refused.
- **No abstention:** there is no "none" option, and its probability is 0.
- **Persistence:** weights save beside the family's JSON and reload with the
  service.

On CREMA-D speech emotion, with 70 training actors and 21 held out, the linear
family scored 0.565 accuracy. A scikit-learn logistic-regression probe on the
same vectors scored 0.567, and the two predicted the same emotion for 99.1% of
clips.

For a complete clip-to-prediction workflow, run the
[audio example](../examples/audio/README.md) on your labeled WAV files.

## Fit a numeric family

When the answer is a number (a score, a resolution time, a return), register a
regression family. Examples carry a vector or text and a numeric field;
`target` names the field.

```python
service.register_family(
    "wine_points",
    [{"state": {"vector": v}, "points": p} for v, p in zip(vectors, points)],
    objective="regress", target="points", form="ridge",
)
answer = service.decide({"state": {"vector": vector},
                         "questions": {"points": {"type": "Score", "family": "wine_points"}}})
# {"score": 88.4, "interval": [86.9, 89.9], "interval_level": 0.8, "calibrated": false, "source": "family_head"}
```

The head is a ridge regression fitted in closed form. Rows are split 60/20/20
into train, selection and report slices unless you pass `split` with your own
row indices; the penalty is chosen by 5-fold cross-validation repeated 3 times
inside the train slice only. The response reports RMSE, MAE and Spearman on the
report slice and an 80% prediction interval from report-slice residuals, with
its coverage measured on held-back halves of that slice. `calibrated` is true
only when the report slice has at least 500 rows and the coverage interval sits
within 0.80 ± 0.05. `from_store` trains from an index registered with numeric
outcomes, or from an outcome store's partition named by `target`; the store's
vectors must come from the `venn-1` profile. A `{"state": {"vectors": [...]}}`
request answers many rows in one call, and `index_apply` streaming accepts a
regression family the same way.

On the TypeSafe wine-score cookbook's split (1,200 notes to fit, 800 held out),
a regression family on `venn-1` vectors reached 1.873 RMSE. The cookbook's
first-round Jev pipeline published 1.869 and its final 38-question pipeline
1.772.

Three options close most of that gap in the same one call:

```python
service.register_family(
    "wine_points",
    [{"state": note, "points": p} for note, p in zip(notes, points)],
    objective="regress", target="points", form="kernel_ridge",
    prompt="SentenceSimilarity", inputs=["log_words", "log_chars"],
)
answer = service.decide({"state": {"texts": new_notes},
                         "questions": {"points": {"type": "Score", "family": "wine_points"}}})
```

What each option does:
- **`prompt`** embeds the notes with the reader's sentence-similarity prompt.
- **`form="kernel_ridge"`** lets the head bend, so it no longer shrinks the notes critics rave about toward the
  mean.
- **`inputs`** add each note's length, which correlates 0.57 with the critic's score.

**On the cookbook's 800 held-out notes,** this family reached 1.765 RMSE, against 1.772 published for the
five-round, 38-question pipeline. Its settings were chosen by cross-validation on the 1,200 training notes.

**On 3,000 notes nobody had scored,** it reached 1.803 against 1.794 for that pipeline, rerun with Jev answering its
38 questions on the same notes. That is within 0.5%, with a 90% CI of −1.7% to +2.6%.

It used one `register_family` call, no LLM calls and no questions.

Run this configuration with `python examples/wine/wine.py --form kernel_ridge`.
The [wine example](../examples/wine/README.md) downloads the notes, keeps the
1,200 fitting notes separate from the 800 evaluation notes, and reports RMSE.

## Filter retrieved passages

Prerequisites: `.[embedding,benchmark]` and CUDA; no Venn head is required.

```bash
venn demos --output /tmp/venn-demos --cache /tmp/venn-demo-cache.sqlite --gpus 1 --rows 1000
```

Read `results.json`: `rag.before` contains the six retrieved passage IDs;
`rag.after` contains only members of that set assigned to the account-access
topic. Compare precision, recall and returned count. An empty filtered set has
undefined precision (`null`) and zero recall. The authored relevance labels
and original control are included. The demo does not generate answers or
measure downstream hallucinations.

## Sweep a real vector store

Use `venn.adapters.FaissAdapter`, `PgVectorAdapter`, `QdrantAdapter`,
`WeaviateAdapter` or `LanceDBAdapter` to read batches with explicit profile and
revision pins. Keep IDs, prewarm options once, consume every output and
checkpoint the source cursor/snapshot identity. A changing collection is not
a fixed evaluation corpus. The pgvector adapter needs a caller-owned server-side
cursor with vector types registered; it does not own the transaction.

For a minimal standalone throughput illustration, `venn sweep --rows 1000`
needs neither a store nor a server. This measures a synthetic local CPU workload,
not the latency of your database or production deployment.

## Inspect a finite drift example

The combined demo above writes `drift.windows`: eight 48-row windows with
Bonferroni alpha .05/8 and 199 permutations. The later windows switch from
account support to shipping. This authored finite-support example demonstrates
the interface; use the [benchmark](benchmark.md) for distinct-row evaluation.
A p-value is not the probability that drift occurred. An ongoing monitor needs
its own sequential alert policy and representative validation.

<!-- docs-pager:start -->
---

[← Previous: Venn vs Jev](venn-and-jev.md) · [Next: Outcome stores →](outcome-api.md) · [Documentation home](README.md)
<!-- docs-pager:end -->
