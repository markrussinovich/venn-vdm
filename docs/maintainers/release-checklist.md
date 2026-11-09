# Release checklist

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

**Maintainers:** [Maintainer guide](maintaining.md) · [Documentation site](website.md) · **Release checklist** · [Publishing the Python package](pypi.md) · [Publishing the decision head](publishing-the-head.md) · [Development evidence](development-evidence.md) · [Comparative evaluation protocol](evaluation-plan.md) · [License audit notes](license-audit.md)

</details>

**Maintainers / Release checklist**
<!-- docs-navigation:end -->

<!-- docs-contents:start -->
<details>
<summary>On this page</summary>

- [Prepared](#prepared)
- [Before changing visibility](#before-changing-visibility)
- [At public launch](#at-public-launch)
- [Follow-up improvements](#follow-up-improvements)

</details>
<!-- docs-contents:end -->

The repository remains private; the published model head is separate.
Keep deployment and package publishing disabled until launch.

## Prepared

- [x] Owner confirmed the published `venn-1` head can be redistributed; the model
  card and license page record that permission.
- [x] Refresh the published Hugging Face bundle and pin its immutable revision in
  the runtime, model card and reproduction guide.
- [x] Attach the latest landscape and vertical comparison videos to the private
  `media-v1` release. Posters, captions and manifests accompany the source.
- [x] Embed the comparison video on the project page and comparison guide.
- [x] Set the planned repository homepage to
  `https://markrussinovich.github.io/venn-vdm/`.
- [x] Keep `DEPLOY_PAGES=false` and `PUBLISH_PYPI=false`; workflows also check
  repository visibility before publishing.
- [x] Prepare `publish.yml` for PyPI Trusted Publishing and document its account
  setup in the [PyPI guide](pypi.md).
- [x] Exclude local MP4s, model files, caches and downloaded ticket data from the
  source distribution.

## Before changing visibility

- [ ] Finish the reviewed source snapshot and pass CI on that exact revision.
- [ ] Build fresh distributions and pass the base-dependency wheel smoke test
  outside the source checkout. Do not reuse older files from `dist/`.
- [ ] Squash/reset to the intended clean public history, as planned by the owner.
  Inspect branches, tags and retained PR refs after that operation. A force push
  alone does not purge objects retained by GitHub; resolve any remaining private
  history before making the repository public.
- [ ] Review which maintainer notes, artifacts and release assets should ship.
  Keep private launch planning in the separate research workspace.
- [ ] Confirm the distribution/import naming decision: `venn-vdm` / `venn`.
- [ ] Complete the remaining checks in the private `publish-todo.md`.

## At public launch

- [ ] Make the prepared head public and verify anonymous access to its pinned
  revision if not already public.
- [ ] Make the cleared GitHub repository public.
- [ ] Enable Pages using **GitHub Actions**, set `DEPLOY_PAGES=true` and dispatch
  the documentation workflow. The current account plan rejects Pages for this
  repository while it is private; `has_pages` reflects actual configuration.
- [ ] Verify the live homepage, guide, videos, captions, downloads and first
  decision without owner credentials. See [website setup](website.md).
- [ ] Tag the verified version and optionally enable PyPI publishing after
  configuring its Trusted Publisher. See [PyPI setup](pypi.md).
- [ ] Configure branch protections, private vulnerability reporting and
  protected PyPI environment deployments before enabling publication.

## Follow-up improvements

- [ ] Headless HTTP mode for outcome-only serving.
- [ ] Better no-match behavior and task-specific thresholds.
- [ ] A quantized reader backend and broader generic-head reader selection.
- [ ] A matched comparison with selected OSS decision models.

<!-- docs-pager:start -->
---

[← Previous: Documentation site](website.md) · [Next: Publishing the Python package →](pypi.md) · [Documentation home](../README.md)
<!-- docs-pager:end -->
