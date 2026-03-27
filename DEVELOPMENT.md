# Development

This repository is artifact-first. The code under `tools/` exists only to
refresh and verify the checked-in dataset under `artifacts/`.

## Maintainer Workflow

1. Install the maintainer dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Refresh artifacts from the pinned ONNX Runtime version:

```bash
python tools/scripts/extract_test_artifacts.py --artifacts-output artifacts --rebuild
```

3. Recompute stored validation expectations:

```bash
python tools/scripts/update_artifact_validation_expectations.py
```

4. Regenerate the validation overview document:

```bash
python tools/scripts/generate_artifact_validation_overview.py
```

5. Run the repository checks:

```bash
pytest -q
```

## Notes

- The ONNX Runtime checkout is prepared on demand under `build/`.
- Generated runtime JSON defaults to `build/` unless a command overrides it.
- `artifacts/` must stay tracked by git.
- `tools/` is maintainer-only infrastructure and should not be presented as the
  main output of the repository.
