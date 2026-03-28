# emx-ort-test-artifacts

[`ONNX Runtime`](https://github.com/microsoft/onnxruntime)'s tests generate models and test vectors on the fly (Python/C++ test code).
That is great for ORT's own CI, but makes the test cases inconvenient to reuse outside
the ONNX Runtime test harness.

This repository materializes those test cases into a versioned dataset of [`ONNX`](https://github.com/onnx/onnx) artifacts.
The core artifact layout follows the [ONNX backend test-data layout](https://github.com/onnx/onnx/tree/main/onnx/backend/test/data)
(`model.onnx` plus `test_data_set_<n>/` with `input_*.pb` / `output_*.pb`). The main difference is the additional
`validation.json` metadata files shipped alongside each test case (plus aggregated validation reports).
The tracked `artifacts/` tree is the product. The code under `tools/` exists only to
refresh, validate, and document that dataset during development.

Downstream consumers should treat this repository as a data source for checked-in
`model.onnx`, `input_*.pb`, `output_*.pb`, and related metadata files.

## What Is Tracked Here

- ONNX models and TensorProto payloads materialized from ONNX Runtime tests.
- Validation metadata and validation reports for the checked-in artifacts.
- Dataset-level metadata describing the pinned ONNX Runtime source version.

The repository is intentionally artifact-first:

- `artifacts/` is the main payload.
- `tools/` is maintainer-only infrastructure.
- No published Python package is produced from this repository.

## Consumer View

If you only need reusable test inputs and expected outputs, start in
[`artifacts/`](artifacts/README.md).

Useful files:

- [`artifacts/README.md`](artifacts/README.md): artifact layout and format.
- [`artifacts/MANIFEST.json`](artifacts/MANIFEST.json): dataset metadata.
- [`artifacts/OPERATORS.md`](artifacts/OPERATORS.md): generated operator summary with counts and grouped test-case lists.
- [`artifacts/VALIDATION_ERRORS.md`](artifacts/VALIDATION_ERRORS.md): generated validation overview.
- `artifacts/.../validation.json`: per-test-case replay metadata captured from the originating ORT test.

`validation.json` captures (most relevant fields):

- Negative/expected-failure cases (`expects_failure` plus optional `expected_failure_substring`)
- Backend constraints via ORT Execution Providers (`included_providers` / `excluded_providers`)
- Per-output comparison rules (`absolute_error`, `relative_error`, `sort_output`)

The binary artifact payload is tracked with Git LFS via
[`/.gitattributes`](.gitattributes).

## Artifact Layout

The artifact tree mirrors the relevant ONNX Runtime test source layout:

```text
artifacts/
  onnxruntime/
    test/
      python/
      contrib_ops/
      testdata/
      providers/
```

Each currently checked-in test-case directory contains:

- `model.onnx`
- `validation.json`
- one or more `test_data_set_<n>/` directories with serialized protobuf inputs
  and outputs

The validation metadata records replay expectations and output-comparison rules
for that case.

## Repository Layout

```text
.
├── artifacts/              # primary repository payload
├── tests/                  # artifact integrity and maintainer-tool tests
├── tools/
│   ├── scripts/            # maintainer CLIs for refresh/validation
│   ├── python/             # shared Python helpers for maintainer tooling
│   └── cpp/                # runtime C++ extractor sources
├── DEVELOPMENT.md          # maintainer workflow
├── AGENTS.md               # repository-specific coding guidance
└── requirements.txt        # maintainer environment dependencies
```

## Maintainer Note

Maintainers who need to regenerate or validate the dataset should use the
commands documented in [`DEVELOPMENT.md`](DEVELOPMENT.md). The maintenance
tooling is intentionally kept out of the main repository narrative so the
purpose of the repository stays clear: checked-in ORT artifacts are the goal,
and the tooling is only a means to refresh them.

## License

MIT. See [`LICENSE`](LICENSE).
  
## Maintained by

This project is maintained by [emmtrix Technologies GmbH](https://www.emmtrix.com).
