# Publishing the Python package

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

**Maintainers:** [Maintainer guide](maintaining.md) · [Documentation site](website.md) · [Release checklist](release-checklist.md) · **Publishing the Python package** · [Publishing the decision head](publishing-the-head.md) · [Development evidence](development-evidence.md) · [Comparative evaluation protocol](evaluation-plan.md) · [License audit notes](license-audit.md)

</details>

**Maintainers / Publishing the Python package**
<!-- docs-navigation:end -->

<!-- docs-contents:start -->
<details>
<summary>On this page</summary>

- [One-time PyPI setup](#one-time-pypi-setup)
- [Prepare privately](#prepare-privately)
- [Publish at launch](#publish-at-launch)

</details>
<!-- docs-contents:end -->

The distribution is `venn-vdm`; Python code imports `venn`. The package version
is in `pyproject.toml`. Creating a GitHub release or pushing a tag does not
upload anything to PyPI.

## One-time PyPI setup

1. Sign into [PyPI](https://pypi.org/account/login/), verify your email and enable
   two-factor authentication.
2. Open [Publishing](https://pypi.org/manage/account/publishing/) and add a
   **pending GitHub Trusted Publisher** for the new project:

   | Field | Value |
   | --- | --- |
   | PyPI project | `venn-vdm` |
   | GitHub owner | `markrussinovich` |
   | Repository | `venn` |
   | Workflow filename | `publish.yml` |
   | Environment | `pypi` |

The [pending publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
creates the PyPI project on its first successful upload. It does not require an
initial manual upload or a long-lived PyPI API token.

## Prepare privately

Keep repository variable `PUBLISH_PYPI=false`. The manually dispatched
`Publish Python package` workflow builds a selected version tag and checks its
distributions with Twine. Its upload job runs only when the repository is
public and `PUBLISH_PYPI=true`. Private preparation can therefore build and
download the distributions without publishing them.
The build also requires the latest push CI run for the exact release-tag
revision to have completed successfully. A running, failed or missing run
blocks publishing; wait for tag CI before dispatching this workflow.

## Publish at launch

After the public launch decision:

1. Verify CI on the release commit, then tag it `vX.Y.Z`, matching the package
   version. Push the tag.
2. Confirm the PyPI Trusted Publisher fields match the table above. Configure
   the GitHub `pypi` environment's release protection as appropriate.
3. Set `PUBLISH_PYPI=true` and manually dispatch `publish.yml` with the version
   tag. Review the build and publishing jobs.
4. Install `venn-vdm` from PyPI in a clean Python 3.12 environment and run
   `venn quickstart`. Turn `PUBLISH_PYPI` off again between releases if desired.

PyPI uploads are public. Use a new version for a replacement release; an
already-used distribution filename cannot be uploaded again. See the
[packaging guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
for manual upload instructions if Trusted Publishing is unavailable.

<!-- docs-pager:start -->
---

[← Previous: Release checklist](release-checklist.md) · [Next: Publishing the decision head →](publishing-the-head.md) · [Documentation home](../README.md)
<!-- docs-pager:end -->
