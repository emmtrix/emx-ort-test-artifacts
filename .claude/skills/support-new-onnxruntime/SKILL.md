---
name: support-new-onnxruntime
description: Advance this repository to a new pinned ONNX Runtime (ORT) release - move the pin, let CI re-extract the artifact dataset, repair extractor breakage, and publish the ort-v<version> tag and release. Use whenever the user asks to support, update, bump, or upgrade an ONNX Runtime version, in English or German (for example "support new ORT", "neue ORT unterstützen", "ORT 1.28 unterstützen", "bump onnxruntime pin", "update ONNX Runtime to <version>").
---

# Supporting a New ONNX Runtime Version

Follow [`UPGRADE_ONNXRUNTIME.md`](../../../UPGRADE_ONNXRUNTIME.md) in the
repository root. It is the authoritative runbook; read it before making any
change.

## Quick Orientation

The whole upgrade is driven by the exact `onnxruntime==<version>` pin in
`requirements.txt`. Everything else either derives from it or is regenerated
from it.

Hand-edited files for a plain bump:

1. `requirements.txt` — the pin
2. `artifacts/MANIFEST.json` — `pinned_onnxruntime_version`
3. `tests/test_onnxruntime_source.py` — both version assertions

Then push a branch, open a pull request, and let the `Refresh Artifact
Dataset` workflow regenerate the dataset and push it back onto the branch.
Fix whatever the new upstream release broke in the extractor
(`tools/scripts/extract_test_artifacts.py`,
`tools/cpp/runtime_extractor/`), with regression tests. After merging, tag
`ort-v<version>` and publish the matching GitHub release.

## Before Starting

If the user did not name a target version, determine the newest exact
`onnxruntime` release that has both a PyPI release and a matching
`v<version>` tag in microsoft/onnxruntime, and confirm it with the user.

## Hard Rules

- Never run the full artifact extraction locally to produce the committed
  dataset; it takes hours. CI owns the refresh.
- Never hand-edit `tests/artifact_validation_expected.json` or anything under
  `artifacts/onnxruntime/`, `artifacts/onnxruntime-negative/`,
  `artifacts/OPERATORS.md`, `artifacts/VALIDATION_ERRORS.md`.
- Never silence a new validation failure by relaxing expectations; fix the
  pipeline, or record a genuinely unreplayable case in
  `artifact_generation_ignored_cases.json` with a concrete reason.
- Always `git pull --rebase` before pushing again — CI pushes refresh commits
  onto the same branch.
- Respect the layering and documentation rules in
  [`AGENTS.md`](../../../AGENTS.md); all code and documentation is English.
