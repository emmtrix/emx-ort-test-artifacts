# AGENTS.md - Guidance for AI Coding Agents

This file provides project-specific rules and guidance for coding agents
working in this repository.

## Repository Purpose

This is an artifact-first repository.

The tracked files under `artifacts/` are the primary output and the main reason
the repository exists. The code under `tools/` is maintainer-only
infrastructure used to generate, refresh, validate, and document that dataset.

## Current State

The repository already contains a checked-in artifact dataset plus maintenance
tooling for runtime extraction and validation. Changes should preserve that
artifact-first framing instead of turning the repository into a generic Python
tool or package project.

## Constraints

- Building C++ code and executing C++ test code is allowed when needed for
  extraction or validation work.
- Generating final artifact files (`.onnx`, `.pb`) is allowed when explicitly
  requested or when validating the runtime C++ extraction pipeline.
- Generating intermediate JSON extraction metadata is allowed.
- Runtime extraction JSON should default to `build/` unless the user explicitly
  requests another location.
- The ONNX Runtime source checkout is ephemeral and must be cloned or updated
  on demand from the pinned `onnxruntime` version in `requirements.txt`.
- Python files may contain minimal orchestration logic needed to build or run
  the extractor pipeline.
- Do not add `pyproject.toml` or packaging configuration. This is not a
  packaged library.
- Keep automation and workflow changes aligned with the artifact-first nature of
  the repository.
- All code, comments, and documentation must be in English.

## Coding Conventions

- Python 3.10+.
- Use type hints where they aid clarity.
- Keep modules small and focused on a single responsibility.
- Prefer explicit over implicit; avoid premature abstraction.
- No third-party dependencies beyond those listed in `requirements.txt`.
- Do not modify the checked-out ONNX Runtime sources in place.
- Derive the ONNX Runtime checkout ref from `requirements.txt`, not from a
  hard-coded commit or branch.

## Rules

1. Preserve the artifact-first narrative in documentation and structure.
2. Keep `artifacts/` tracked by git; never add it to `.gitignore`.
3. Place maintainer-oriented code and scripts under `tools/`, not in the main
   repository narrative.
4. New modules must include a module-level docstring before implementation.
5. Preserve layering strictly:
   - Generation policy belongs in generation/extraction code.
   - Validation logic must only validate what exists and must not know
     generation-time ignore policy.
   - Reporting/documentation code may format and present configured policy, but
     must not silently change generation or validation behavior.
   - Test-runner concerns such as skip classification or local-environment
     handling must stay outside core domain logic unless the module is
     explicitly responsible for runner integration.
   - Shared helper modules must not hard-code repository policy or workspace
     layout when that responsibility belongs to a script or adapter layer.

## Artifact Layout

```text
artifacts/
  onnxruntime/
    test/
      python/
      contrib_ops/
      testdata/
      providers/
```

The layout mirrors ONNX Runtime source paths so each checked-in artifact can be
traced back to its originating test.
