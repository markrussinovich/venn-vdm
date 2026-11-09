# API

<!-- docs-navigation:start -->
[Documentation](README.md) · [Get started](getting-started.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md) · [Examples](../examples/README.md) · [Task recipes](cookbooks.md) · **API** · [Results](evidence.md)

<details>
<summary>Browse all documentation</summary>

**Project home:** [Home](../README.md)

**Start here:** [Documentation overview](README.md) · [Getting started](getting-started.md) · [Installation](installation.md) · [Concepts](concepts.md) · [Venn vs Jev](venn-and-jev.md)

**Task guides:** [Task recipes](cookbooks.md) · [Outcome stores](outcome-api.md)

**Examples:** [Choose an example](../examples/README.md) · [Hot dog image classifier](../examples/hotdog/README.md) · [Precedent Desk](../examples/precedent_desk/README.md) · [Wine predictor](../examples/wine/README.md) · [Audio classifier](../examples/audio/README.md) · [Screen-only arena](arena-example.md) · [Demos and videos](demos.md)

**Reference:** **API reference** · [Design rationale](design.md) · [Venn 1 model card](model-card.md)

**Results and limits:** [Performance and cost](performance.md) · [VennSpace-1](benchmark.md) · [Results and evidence](evidence.md) · [Reproduction](reproduction.md) · [Semantic boundary findings](redteam.md) · [Limitations](limitations.md) · [Comparison video methods](same-memory-video.md) · [Comparison notebooks and renderer](../examples/same_memory_video/README.md)

**Project:** [Policies](policies.md) · [Licenses and attribution](licenses.md) · [Contributing to Venn](../CONTRIBUTING.md)

**Maintainers:** [Maintainer guide](maintainers/maintaining.md) · [Documentation site](maintainers/website.md) · [Release checklist](maintainers/release-checklist.md) · [Publishing the Python package](maintainers/pypi.md) · [Publishing the decision head](maintainers/publishing-the-head.md) · [Development evidence](maintainers/development-evidence.md) · [Comparative evaluation protocol](maintainers/evaluation-plan.md) · [License audit notes](maintainers/license-audit.md)

</details>

**Reference / API reference**
<!-- docs-navigation:end -->

<!-- docs-contents:start -->
<details>
<summary>On this page</summary>

- [First request](#first-request)
- [Question types](#question-types)
- [Index lifecycle](#index-lifecycle)
- [Family lifecycle](#family-lifecycle)
- [Errors and operational limits](#errors-and-operational-limits)
- [Server options and host checks](#server-options-and-host-checks)
- [MCP bridge](#mcp-bridge)
- [App-owned outcome decisions](#app-owned-outcome-decisions)

</details>
<!-- docs-contents:end -->

Start the server using [getting started](getting-started.md). The following
requests use loopback port 8000. Text requests require the downloaded head and reader (see getting started);
the fixture server in [reproduction](reproduction.md) exercises contracts only.
The compatible client versions are listed in [client checks](https://github.com/markrussinovich/venn-vdm/blob/main/tests/clients/README.md).
Venn serves the Decisions contract at `/api/alpha/decisions`; the current
TypeSafe `/v1/systemone` endpoint is a separate interface.
Interactive OpenAPI documentation is available locally at `/docs` and the
machine-readable schema at `/openapi.json` while the server is running.

## First request

```bash
curl --fail-with-body http://127.0.0.1:8000/api/alpha/decisions \
  -H 'Content-Type: application/json' \
  -d '{"model":"venn-1","state":"I cannot sign in","questions":{"team":{"type":"choice","instructions":"Who owns this ticket?","criteria":{"account":"Login or access","billing":"Payments","no_match":null}}}}'
```

Equivalent Python: `python examples/route_ticket.py`. Equivalent JavaScript
from the checkout (Node 20 or later; CI uses Node 24):

```js
import { VennClient } from "./clients/venn.mjs";
const response = await new VennClient().decide("I cannot sign in", {
  team: {type: "choice", instructions: "Who owns this ticket?",
         criteria: {account: "Login or access", billing: "Payments", no_match: null}}
});
console.log(response.answers.team);
```

Run this as an ES module (`.mjs`). Response fields include `id`, `model`,
`provider`, named `answers`, `usage` and top-level `venn` profile metadata.
The state vector is omitted by default; send `X-Venn-Return-Vector: 1` (or
`Client(...).decide(..., return_vector=True)` in Python, with
`from venn.client import Client`) to include it as `venn.vector`.
For Choice, inspect `answers.team.choice`, `confidence`, `probabilities`
and `venn`. The returned choice is a criterion key, including `no_match` when
selected. Null is the criterion value in the request, not the response key.
Values depend on the head and input; fixture predictions are not example accuracy.
For orientation, this is the answer object from the deterministic **fake**
contract server for the first request (rounded here):

```json
{
  "type": "choice",
  "choice": "billing",
  "confidence": 0.3791,
  "probabilities": {"account": 0.2896, "billing": 0.3791, "no_match": 0.3313},
  "venn": {
    "calibrated": false,
    "none_probability": 0.3313,
    "abstained": false,
    "probabilities_conditioned_on_not_none": false
  }
}
```

The deliberately nonsemantic fixture result illustrates why wire-format tests
must not be interpreted as model-quality tests.

## Question types

`POST /api/alpha/decisions`:

```json
{
  "model": "venn-1",
  "state": "The checkout page is broken",
  "questions": {
    "defect": {"type": "noul", "instructions": "Is the software broken?"},
    "team": {
      "type": "choice",
      "instructions": "Who owns this ticket?",
      "criteria": {"account": "Login problems", "billing": "Checkout and payments"}
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent?",
      "criteria": ["Can wait", "Should be fixed soon", "Blocking"]
    }
  }
}
```

Noul is a yes/no question type; the name comes from the Decisions API.
Choice returns `choice` and `probabilities`; Noul returns numeric `noul`
(probability of yes); Score returns
the probability-weighted zero-based level, `legend` and `probabilities`.
Structured JSON content is canonically serialized, not automatically
field-routed. No text is generated.

Generic distributions are not certified calibrated. Unless Choice has
an explicit null abstention criterion, the standard distribution is
conditional on the described options. Venn-specific abstention/calibration
metadata is under `venn`; the OpenRouter TypeScript SDK strips unknown
metadata. Applications needing abstention must use an explicit null option
or a client preserving Venn metadata. Score's generic path is experimental;
ordered registered families need independent task validation.
Registered Score is marked nonexperimental after the ordinal capability
gate; this does not certify a custom family's ordering or probabilities.

Venn extensions:

| Type | Inputs |
| --- | --- |
| `nearest_reference` | candidate state and `store` index handle |
| `set_membership` | candidate state and `set` vectors, or vector-set state with `method: "knn5"` and `outliers` |
| `population_shift` | vector-population state and `batch_b`; permutation p-value |
| `index_apply` | indexed state, options, row cap and batch size |

Use explicit `{"vector": [...]}`, `{"vectors": [...]}` or
`{"index_ref": "..."}` state for geometry.

**Many items in one request:** each single-item state has a plural form, embedded in bulk:

| One item | Many items, up to 512 |
| --- | --- |
| `{"vector": [...]}` | `{"vectors": [[...], ...]}` |
| `{"text": "..."}` | `{"texts": ["...", ...]}` |
| `{"image": "BASE64"}` | `{"images": ["BASE64", ...]}` |
| `{"audio": "BASE64"}` | `{"audios": ["BASE64", ...]}` |

- **Questions:** a plural state gets one answer per row from an `IndexApply` question and from a `Score`
  question that names a regression family. A registered field family reads a `vectors` state as the fields of
  one item. `Choice`, `Noul` and `Score` on the generic head, and families registered with single-vector
  states, answer one item: a plural state returns 422 (`unsupported_input` or `family_state`), so ask
  `IndexApply` instead.
- **Embedding:** Venn embeds the cache misses in batches, writes them to the cache in one transaction, and answers
  each question once per row.
- **Validation:** any invalid item rejects the whole request before anything is embedded.
- **In Python:** `DecisionService.embed(texts)` and `DecisionService.embed_media(payloads, kind="image")` return a
  matrix of unit vectors. Register vectors/IDs/outcomes
with `POST /indexes`; use `/index-apply/{handle}` for NDJSON above 512
output rows. Registrations and source stores must use `venn-1`'s exact
profile.
Without members, the default set method identifies one centroid outlier;
`knn5` ranks fifth-neighbor squared distances and returns the requested
number of row indices. Density ranking has no probabilities or probability
threshold. It is the method used by the benchmark's large-membership task.

Raw `{"image": "BASE64"}` or `{"audio": "BASE64"}` state requires
`X-Venn-Experimental: media`. Audio is mono 16 kHz, <=30 seconds; media is
<=6 MB, images <=4 million pixels. Referenced assets and Box are not
implemented. No network media URLs are fetched.
Typed media parts nested in JSON arrays/objects are rejected explicitly,
not silently embedded as their JSON text.

OpenRouter SDK 1.5.33 requires:

```js
await sdk.alpha.decisions.create(
  { decisionsRequest: request },
  { serverURL: "http://127.0.0.1:8000" }
);
```

Its constructor-level URL does not override this operation's default
server. Use a transport allowlist when testing local requests.

## Index lifecycle

The executable [index example](../examples/index_vectors.py) includes complete
requests and streamed-response consumption. Index streaming applies one prepared
question to every vector in a registered index and streams the results as
NDJSON. The sequence is:

1. `POST /indexes` with `vectors: {"vectors": [...]}`, distinct `ids`, and optional `outcomes`.
2. Read the returned `index_ref` handle.
3. Ask a Decisions `nearest_reference` question with `store: handle`, or stream `POST /index-apply/{handle}`.
4. For streaming, send an `index_apply` question with `instructions_vec`, labeled vector `options` and `batch_size`.
5. Consume each NDJSON object with `offset` and `rows`; every row retains its original `id`.
6. `DELETE /indexes/{handle}` after consumption.

Indexes and prewarmed questions live in server memory and do not survive a
restart. Re-register from the authoritative store. `POST /questions/{name}`
prepares an `index_apply` question once and returns a `question_ref` to reuse
across streams: send `{"type": "index_apply", "question_ref": "..."}` (with
`batch_size` if needed) in place of the instructions and options. Changing its
options or instructions requires re-registration. Above 512 output rows, use
NDJSON rather than the bounded Decisions response.

## Family lifecycle

Use `PUT /families/{name}` with `examples`, `options`, `instructions_vec` and
optional `seed`. Examples contain `state` (a vector or consistent vector set)
and `label`; optionally give every example a string `group_id`. Options contain
`label` and `vector`. Supply at least five distinct examples per label and at
least ten overall. Labels must be distinct and cannot use the reserved `none`.

By default, the head trains on every supplied example. With an explicit
`split`, only the training slice fits the head; selection and report rows
remain separate. Weight decay and the number of epochs are
chosen by cross-validation inside the examples: 5 folds by default, grouped by
`group_id` when given, so a group never sits on both sides of a fold. Optional
fields: `split` with `train`, `selection` and `report` row indices (a group may
not span slices; the selection slice then fits the temperature and the report
slice is scored), `cv` with `folds` (2–20) and `max_epochs` (1–100, default
24), and `class_weight` as `"balanced"` or `{label: weight}`.

`"form": "linear"` replaces the residual head with a multinomial logistic
regression on the state vectors (single-vector states only). Its penalty C is
chosen from 0.01–100 by the same cross-validation, on log loss, and it accepts
the same `split`, `cv` and `class_weight`. A linear family has no abstention:
its `none` probability is always 0. It scores only the labels it was registered
with.

The response includes split counts, the chosen settings and their
cross-validation curves, the temperature and its source, cross-validated
accuracy and ECE, report accuracy/ECE when a report slice is given,
calibration status and additional report items needed. Without a report
slice the family is never marked calibrated. Use `family` in subsequent questions and preserve registered
option vectors. Fitting performs synchronous CPU work, so allow a longer client
timeout for larger tasks. See [the executable family example](../examples/registered_family.py)
for fitting, response inspection and save/reload verification.

Families persist under `VENN_HOME/registry/families` by default. They are loaded
on service startup and checked against the profile. `DELETE /families/{name}`
removes a registration. Protect this application state and back it up as needed.
Custom registries and caches can be passed to the Python `DecisionService`.
A family existing or an ordinal capability flag does not certify its quality.

A regression family is registered with the same route and
`"objective": "regress"`: send `examples` that each carry, beside `state`, a
numeric field named by `target`, such as
`{"state": {"vector": [...]}, "points": 88}` (a field inside `state` is
rejected with 422), or `from_store` with an index or outcome-store handle; and
optionally `split` (`train`, `selection`, `report` row indices) and `cv`
(`folds`, `repeats`). It answers Decisions `score` questions that name it,
without criteria: `{"type": "score", "family": "wine_points"}` returns `score`,
`interval`, `interval_level` (0.8), `calibrated` and `source: "family_head"`.

Three options shape a regression family's inputs and head:
- **`"form": "kernel_ridge"`** fits an RBF kernel ridge instead of the linear ridge. Alpha (0.01–10) and gamma
  (0.5–4) are chosen by the same cross-validation.
- **`"prompt"`** embeds text states with another of the reader's documented task prompts: `Classification` (the
  default), `SentenceSimilarity` or `Clustering`.
- **`"inputs": ["log_words", "log_chars"]`** appends numeric inputs computed from the state text: log word count
  and log character count, standardised. Their scale (0.03–0.3) is chosen by the same cross-validation.

A family with a non-default `prompt` or with `inputs` registers from text examples and answers from text only.
Send a string, `{"text": ...}` or `{"texts": [...]}` with up to 512 texts. Venn embeds each with the family's
prompt and computes the inputs itself, so the caller sends nothing new. It refuses vector states, because it can't
tell which prompt made a vector. Cached vectors are keyed by prompt.

On the wine cookbook's 800 held-out notes, one call reached 1.765 RMSE, against 1.772 published for the cookbook's
five-round, 38-question pipeline. Its settings were chosen on the 1,200 training notes. On 3,000 notes nobody had
scored, it reached 1.803 against 1.794 for that pipeline rerun on Jev: within 0.5%, with a 90% CI of −1.7% to +2.6%.

```python
register_family("wine_points", examples, objective="regress", target="points", form="kernel_ridge",
                prompt="SentenceSimilarity", inputs=["log_words", "log_chars"])
```

### Temperature calibration

`PUT /calibrations/{family}` with `logits` (200–10,000 rows), integer `targets`
and optional `type` (default `Choice`) fits a softmax temperature, stores it in
`VENN_HOME/registry/calibrations.json` and returns `{"temperature": ...}`.
Later questions of that type that name `family` use it in place of the generic
temperature. `DELETE /calibrations/{family}?type=Choice` removes it. A stored
temperature does not mark answers `calibrated`.

## Errors and operational limits

Contract failures return HTTP 422 by default; missing handles use 404. Error
objects contain a machine-readable code and message. Inspect the response body
rather than retrying every 4xx response. Profile mismatches require compatible
vectors, not truncation or padding. Use 1–32 named questions per request and
at most 256 criteria. Index registration accepts up to one million vectors;
large real stores should be handled in bounded application batches.

The Python client defaults to a 60-second timeout; set `Client(timeout=...)`
for your workload. The JavaScript client is a small fetch wrapper without its
own timeout/retry policy. Neither client promises server-side cancellation.
`GET /health` confirms service availability, not model accuracy. The local
server has no authentication, quotas or tenant isolation.

## Server options and host checks

`venn serve` takes `--host` and `--port` (default `127.0.0.1:8000`),
`--model-dir` (default `VENN_MODEL_DIR`, else the published head is
downloaded) and `--site-dir` to mount a built documentation site at `/guide`.
It loads the text reader in a background thread as it starts; `GET /health`
reports `reader` as `loading`, `ready`, `absent` (not downloaded) or `error`.
`--no-preload` skips that load for low-memory or vector-only deployments, and
the reader then loads on the first text request. Without a downloaded reader,
text requests return 503 `reader_missing`; run `venn download-embedder`.

The server accepts only requests whose `Host` header names `127.0.0.1`,
`localhost` or `[::1]`; any other hostname gets 400 `Invalid Host header`. A
browser `Origin` header must match the request's scheme, hostname and port,
or the request gets 403. Native clients may omit `Origin`. These checks block
DNS rebinding and cross-origin pages; they are not authentication.

Binding with `--host 0.0.0.0` does not change the accepted hostnames: a client
that reaches the server by IP address or DNS name gets 400 until that name is
allowed. `--allowed-host NAME` (repeatable) accepts an exact hostname and
replaces the loopback defaults, so add `--allowed-host 127.0.0.1` as well if
local clients still need it. Wildcards are rejected at startup. In Python, pass
`create_app(..., allowed_hosts=[...])`. See [security](../SECURITY.md) before
exposing the server beyond loopback.

## MCP bridge

Install `.[mcp]`, start a local Venn API and then start the stdio bridge:

```bash
venn mcp --endpoint http://127.0.0.1:8000
```

Configure an MCP client to launch command `venn` with arguments
`["mcp", "--endpoint", "http://127.0.0.1:8000"]` in the installed environment.
The `decide` tool takes `state`, `questions`, optional `model: "venn-1"` and
`experimental_media: false`; its questions use the same shape as the first
request above. The bridge does not launch the API or download the head.

## App-owned outcome decisions

The experimental [outcome API](outcome-api.md) adds `nearest_outcome`, tagged
representations, fresh filtered adapters and batched uint8 image decisions.
The Python `OutcomeService` works without a trained head. The HTTP routes run
inside `DecisionService`, which loads the `venn-1` head at startup even when
requests only use outcome stores. Use the in-process service for headless
operation; see [setup and prerequisites](outcome-api.md#http-setup).

| Entry point | Input / response |
| --- | --- |
| `OutcomeService.decide_batch` | Up to 128 tagged vector states or RGB uint8 images and 32 named `nearest_outcome` questions |
| `POST /decide-batch` | JSON `states`/`questions`, or binary images with `X-Venn-Batch` metadata; returns per-state vectors, tags, answers and evidence |
| `POST /api/alpha/decisions` | Single tagged state with a `nearest_outcome` question; preserves the Decisions response envelope |

An outcome question specifies a registered `store`, option names, `k`,
`min_support`, `goal` and optional scalar equality `filters`. Utilities are not
probabilities: inspect the supported `answer`, policy `action`, `exploratory`
flag and per-option evidence rather than treating a mean as calibrated confidence.
The Python client exposes `Client.decide_batch`; the JavaScript thin client and
MCP bridge do not expose the binary batch interface.

Outcome stores are registered by application startup code using adapters;
`POST /indexes` remains the separate snapshot-index API. Outcome handles do not
survive a service restart: re-register the adapter against the authoritative
app store. Writes remain the application's responsibility; no upsert/delete
endpoint for a mutable registered outcome index is implemented.

<!-- docs-pager:start -->
---

[← Previous: Demos and videos](demos.md) · [Next: Design rationale →](design.md) · [Documentation home](README.md)
<!-- docs-pager:end -->
