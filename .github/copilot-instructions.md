# GitHub Copilot Instructions

Refer to [AGENTS.md](../AGENTS.md) for project-specific guidance.

## Key Rules For This Repository

- This repository is artifact-first. Prefer changes that improve the checked-in
  dataset and its documentation over tool-centric framing.
- Treat `artifacts/` as the product and `tools/` as maintainer-only
  infrastructure.
- Do not add `pyproject.toml` or packaging configuration.
- Keep modules and scripts explicit, minimal, and single-purpose.
- Scope covers ONNX Runtime artifacts derived from both Python and C++ tests.
- Do not run the full ORT test suite blindly. Build or execute only the
  targeted maintainer flow required for the task.
- Never add `artifacts/` to `.gitignore`.
- All code, comments, and documentation must be in English.
