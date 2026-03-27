# GitHub Copilot Instructions

Refer to [AGENTS.md](../AGENTS.md) for all project-specific guidance.

## Key Rules for This Repository

- **Do not implement extraction logic yet.**  The repository is in the
  scaffolding phase.  All Python files should contain docstrings and
  placeholder structure only.
- **Prioritize clarity and incremental structure.**  Add one well-named,
  single-responsibility module or function at a time.
- **Scope covers all ORT tests** – both Python-based and C++-based.  Do not
  restrict designs or documentation to Python tests only.
- **Do not run the full ORT test suite blindly.**  Build or execute only the
  targeted extractor flow that is required for the current task.
- **Do not add `pyproject.toml`** or any packaging configuration.  This is a
  non-packaged internal utility.
- **Do not add CI workflows** or test runners.
- **Never add `artifacts/` to `.gitignore`.**  Artifact files must be
  committed to the repository.
- All code, comments, and documentation must be in **English**.
- When in doubt, keep it minimal and explicit.
