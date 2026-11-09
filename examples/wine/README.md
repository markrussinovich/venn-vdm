# Wine scores from tasting notes

<!-- docs-navigation:start -->
[Documentation](../../docs/README.md) · [Get started](../../docs/getting-started.md) · [Concepts](../../docs/concepts.md) · [Venn vs Jev](../../docs/venn-and-jev.md) · [Examples](../README.md) · [Task recipes](../../docs/cookbooks.md) · [API](../../docs/api.md) · [Results](../../docs/evidence.md)

<details>
<summary>Browse all documentation</summary>

**Project home:** [Home](../../README.md)

**Start here:** [Documentation overview](../../docs/README.md) · [Getting started](../../docs/getting-started.md) · [Installation](../../docs/installation.md) · [Concepts](../../docs/concepts.md) · [Venn vs Jev](../../docs/venn-and-jev.md)

**Task guides:** [Task recipes](../../docs/cookbooks.md) · [Outcome stores](../../docs/outcome-api.md)

**Examples:** [Choose an example](../README.md) · [Hot dog image classifier](../hotdog/README.md) · [Precedent Desk](../precedent_desk/README.md) · **Wine predictor** · [Audio classifier](../audio/README.md) · [Screen-only arena](../../docs/arena-example.md) · [Demos and videos](../../docs/demos.md)

**Reference:** [API reference](../../docs/api.md) · [Design rationale](../../docs/design.md) · [Venn 1 model card](../../docs/model-card.md)

**Results and limits:** [Performance and cost](../../docs/performance.md) · [VennSpace-1](../../docs/benchmark.md) · [Results and evidence](../../docs/evidence.md) · [Reproduction](../../docs/reproduction.md) · [Semantic boundary findings](../../docs/redteam.md) · [Limitations](../../docs/limitations.md) · [Comparison video methods](../../docs/same-memory-video.md) · [Comparison notebooks and renderer](../same_memory_video/README.md)

**Project:** [Policies](../../docs/policies.md) · [Licenses and attribution](../../docs/licenses.md) · [Contributing to Venn](../../CONTRIBUTING.md)

**Maintainers:** [Maintainer guide](../../docs/maintainers/maintaining.md) · [Documentation site](../../docs/maintainers/website.md) · [Release checklist](../../docs/maintainers/release-checklist.md) · [Publishing the Python package](../../docs/maintainers/pypi.md) · [Publishing the decision head](../../docs/maintainers/publishing-the-head.md) · [Development evidence](../../docs/maintainers/development-evidence.md) · [Comparative evaluation protocol](../../docs/maintainers/evaluation-plan.md) · [License audit notes](../../docs/maintainers/license-audit.md)

</details>

**Examples / Wine predictor**
<!-- docs-navigation:end -->

`wine.py` predicts a wine critic's score (80–100) from the tasting note with a
Venn regression family. It follows the data and split of TypeSafe's
[Autoresearch feature discovery](https://docs.typesafe.ai/cookbooks/autoresearch_feature_discovery)
cookbook: 2,000 winemag reviews, the first 1,200 after a seed-0 shuffle to fit
and the other 800 held out.

```bash
pip install -e ".[embedding]"
venn download-embedder
python examples/wine/wine.py                    # the API's default reader, on CPU
python examples/wine/wine.py --device cuda:0    # the same reader on a GPU, bf16
python examples/wine/wine.py --form kernel_ridge  # SentenceSimilarity + length inputs
python examples/wine/wine.py --form kernel_ridge --device cuda:0
```

The first run downloads the Venn decision head automatically. Set `VENN_HOME`
to choose where the embedding cache and fitted family are saved.

The script downloads the pinned CSV from Hugging Face
([`GroNLP/ik-nlp-22_winemag`](https://huggingface.co/datasets/GroNLP/ik-nlp-22_winemag),
revision `90eb39f3…`, CC BY-SA 4.0, derived from Wine Enthusiast reviews) each
time it runs. No notes, vectors or predictions are stored in this repository.
It embeds the 2,000 notes with the `venn-1` reader, registers a regression
family on the 1,200 fitting notes, scores the 800 held-out notes and prints the
held-out RMSE.

The default family is a ridge regression on the note's 768-dimensional vector. Its
penalty is chosen by 5-fold cross-validation repeated 3 times on the fitting
notes, the same protocol the cookbook uses to select its features. A Score
question that names the family returns a score and an 80% prediction interval:

```python
service = DecisionService(model_dir=model_dir)
vectors = service.embed(notes, prompt="Classification")
examples = [{"state": {"vector": vector.tolist()}, "points": score}
            for vector, score in zip(vectors[:1200], points[:1200])]
service.register_family("wine_points", examples, objective="regress", target="points",
                        split={"train": list(range(1200))})
service.decide({"state": {"vectors": vectors[1200:1712].tolist()},
                "questions": {"points": {"type": "Score", "family": "wine_points"}}})
```

Here `notes` and `points` are the 2,000 tasting notes and scores, and
`DecisionService` comes from `venn`.

## Results on the 800 held-out notes

| Method | Held-out RMSE | Source |
| --- | ---: | --- |
| Predict the mean of the fitting notes | 3.088 | TypeSafe, published |
| The note as word counts, CatBoost | 2.466 | TypeSafe, published |
| Ask Jev for the score, shifted | 2.145 | TypeSafe, published |
| **Venn regression family, `venn-1` vectors** | **1.873** (GPU) / **1.872** (CPU) | this example |
| **Venn kernel family, SentenceSimilarity + length** | **1.765** | recorded kernel configuration, now available with `--form kernel_ridge` |
| 18 Jev questions from one proposal round, CatBoost | 1.869 | TypeSafe, published |
| 38 Jev questions after five rounds, CatBoost | 1.772 | TypeSafe, published |

TypeSafe's numbers are copied from the
[cookbook page](https://docs.typesafe.ai/cookbooks/autoresearch_feature_discovery).
Their five-round pipeline is more accurate than the default ridge baseline.
The kernel configuration reaches similar accuracy in one registration call.
It uses the reader's SentenceSimilarity prompt and the note's log word and
character counts. The script caches that prompt separately, fits only on the
first 1,200 notes, and sends the remaining notes as text for prediction.
Its JSON API returns both the predicted score and an 80% interval.
On 3,000 additional notes, the recorded kernel head scored 1.803 RMSE versus
1.794 for the 38-question pipeline. See [Venn and Jev](../../docs/venn-and-jev.md)
and the [result record](../../docs/media/same-memory/wine-eval.json).

## Recorded ridge timings

| | GPU (one A100 80GB) | CPU (default reader, 24 threads) |
| --- | ---: | ---: |
| Download the CSV | 1.2 s | 1.4 s |
| Embed 2,000 notes | 2.9 s | 83 s |
| Register the family (1,200 notes) | 3.3 s | 3.2 s |
| Predict 800 notes | 0.1 s | 0.1 s |
| Whole run, including Python start-up and loading the reader | 25 s | 92 s |

The CPU column is the API's default reader in float32 on an AMD EPYC 7V12 host
with 2 sockets of 48 cores. A batch of 32 or more texts raises PyTorch to 24
threads, or to `VENN_EMBED_THREADS` if set, and restores the one-thread pin used
for serving afterwards. On this host one thread took 722 s to embed the notes
and all 96 took 1,017 s, because threads spanning both sockets stall each other.

The table records the earlier ridge-only script. The current script prints
download, embedding, registration and prediction timings for the selected
configuration, plus total elapsed time including model startup. Kernel fitting
does more work than ridge. Repeat runs reuse the embedding cache.

<!-- docs-pager:start -->
---

[← Previous: Precedent Desk](../precedent_desk/README.md) · [Next: Audio classifier →](../audio/README.md) · [Documentation home](../../docs/README.md)
<!-- docs-pager:end -->
