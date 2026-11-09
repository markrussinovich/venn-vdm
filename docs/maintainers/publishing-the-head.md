# Publishing the decision head

<!-- docs-navigation:start -->
[Documentation](../README.md) · [Get started](../getting-started.md) · [Concepts](../concepts.md) · [Venn vs Jev](../venn-and-jev.md) · [Examples](../../examples/README.md) · [Task recipes](../cookbooks.md) · [API](../api.md) · [Results](../evidence.md)

<details>
<summary>Browse all documentation</summary>

**Project home:** [Home](../../README.md)

**Start here:** [Documentation overview](../README.md) · [Getting started](../getting-started.md) · [Installation](../installation.md) · [Concepts](../concepts.md) · [Venn vs Jev](../venn-and-jev.md)

**Task guides:** [Task recipes](../cookbooks.md) · [Outcome stores](../outcome-api.md)

**Examples:** [Choose an example](../../examples/README.md) · [Hot dog image classifier](../../examples/hotdog/README.md) · [Precedent Desk](../../examples/precedent_desk/README.md) · [Wine predictor](../../examples/wine/README.md) · [Audio classifier](../../examples/audio/README.md) · [Screen-only arena](../arena-example.md) · [Demos and videos](../demos.md)

**Reference:** [API reference](../api.md) · [Design rationale](../design.md) · [Venn 1 model card](../model-card.md)

**Results and limits:** [Performance and cost](../performance.md) · [VennSpace-1](../benchmark.md) · [Results and evidence](../evidence.md) · [Reproduction](../reproduction.md) · [Semantic boundary findings](../redteam.md) · [Limitations](../limitations.md) · [Comparison video methods](../same-memory-video.md) · [Comparison notebooks and renderer](../../examples/same_memory_video/README.md)

**Project:** [Policies](../policies.md) · [Licenses and attribution](../licenses.md) · [Contributing to Venn](../../CONTRIBUTING.md)

**Maintainers:** [Maintainer guide](maintaining.md) · [Documentation site](website.md) · [Release checklist](release-checklist.md) · [Publishing the Python package](pypi.md) · **Publishing the decision head** · [Development evidence](development-evidence.md) · [Comparative evaluation protocol](evaluation-plan.md) · [License audit notes](license-audit.md)

</details>

**Maintainers / Publishing the decision head**
<!-- docs-navigation:end -->

The `venn-1` head lives in the Hugging Face model repo `markrussinovich/venn-1`,
separate from the source tree and the Python package. The repository layout is:

```text
README.md                      model card (built from docs/model-card.md)
assets/config.json             reader pin, temperatures, default head
assets/manifest.json           SHA-256 of the two files below
assets/universal.safetensors   the CosineAbstention weights
```

`venn download-model` and `venn serve` fetch `assets/` at the revision pinned in
`venn/profiles.py` (`HEAD_REPO_ID`, `HEAD_REVISION`).

## Publish or update

```bash
python scripts/publish_head.py --model-dir /path/to/assets --output /tmp/venn-head --dry-run
python scripts/publish_head.py --model-dir /path/to/assets --output /tmp/venn-head
```

The script refuses to stage anything whose weights or loader configuration differ
from the checksums in `venn/profiles.py`, strips internal keys from the config,
writes a fresh manifest and builds the model card. Repositories are created
private; pass `--public` to create a public one, or change visibility in the Hub
settings later. After a successful upload, update `HEAD_REVISION` in
`venn/profiles.py` to the printed commit and update the model card's pinned
revision.

A different head is a new profile. Train and evaluate it, record its checksums
in `venn/profiles.py` under a new profile and publish it to a new repository
rather than overwriting `venn-1`.

The owner approved head redistribution on November 7, 2026. The model card
records that permission separately from the MIT code license. During
private release preparation, keep the Hub repository private and pass the
intended account's token explicitly if `HF_TOKEN` names another account.

## Credentials

Use a write token for the account that owns the repository. The script reads
`HF_TOKEN` or the token stored by `hf auth login`; do not commit tokens.

<!-- docs-pager:start -->
---

[← Previous: Publishing the Python package](pypi.md) · [Next: Development evidence →](development-evidence.md) · [Documentation home](../README.md)
<!-- docs-pager:end -->
