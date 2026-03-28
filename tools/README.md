# Tools

This directory contains maintainer-only infrastructure for refreshing and
validating the checked-in artifact dataset.

- `scripts/`: CLI entry points.
- `python/`: shared Python support code for the CLIs.
- `cpp/`: runtime extractor sources used during artifact refresh.

The repository is intentionally not tool-first. If you are looking for the
actual repository payload, start in [`../artifacts/`](../artifacts/README.md).

Generated dataset reports include:

- `python tools/scripts/generate_artifact_validation_overview.py`
- `python tools/scripts/generate_operator_markdown.py`
