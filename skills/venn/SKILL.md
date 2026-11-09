---
name: venn
description: Ask a local Venn server typed Choice, Score and Noul (yes/no) questions, plus space-level questions over vector stores.
---

# Venn decisions

Use the configured local Venn endpoint or the Venn MCP `decide` tool. Never
send private vectors or text to an unapproved remote endpoint.

Question types: `choice` picks one category from named criteria; `noul` returns
the probability that a yes/no condition holds; `score` returns a level on an
ordered scale. Include complete instructions and criteria. Space-level types are
`nearest_reference` with a registered `store`, `set_membership` with a `set`,
`population_shift` with `batch_b`, and `index_apply` with an indexed state.

Pin `model: venn-1`. Vectors are only comparable within one profile and
revision; a different reader requires re-embedding and re-registering.
Media input is experimental and requires explicit user opt-in.

Read the abstention and calibration metadata when present. Standard Decisions
clients may discard Venn metadata and return only conditional probabilities.
Probabilities and shift p-values do not authorize an action, and Venn is not a
privacy or security boundary. Do not invent benchmark scores, calibration
claims or latency guarantees.
