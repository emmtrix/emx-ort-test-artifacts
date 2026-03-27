# Artifacts

This directory is the primary payload of the repository.

It stores checked-in ONNX models, TensorProto `.pb` files, and validation
metadata derived from ONNX Runtime tests. Consumers should start here, not in
the maintenance tooling under `tools/`.

## Contents

- `onnxruntime/`: tracked artifact dataset mirroring ONNX Runtime test sources.
- `MANIFEST.json`: dataset-level metadata, including the pinned source version.
- `VALIDATION_ERRORS.md`: generated summary of non-OK validation outcomes and ignored cases.

All files in this directory are committed so downstream tooling can consume
them without rebuilding ONNX Runtime locally.

## Layout

```text
artifacts/
  onnxruntime/
    test/
      python/
        contrib_ops/
          <test_file>/
            <test_case>/
              model.onnx
              validation.json
              test_data_set_0/
                input_0.pb
                input_1.pb
                output_0.pb
      contrib_ops/
        <test_file>/
          <test_name>_run<n>/
            model.onnx
            validation.json
            test_data_set_0/
              input_0.pb
              output_0.pb
      testdata/
        <name>/
          ...
      providers/
        <name>/
          ...
```

## Source-to-Artifact Mapping

| ORT source path | Artifact sub-path | Extraction method |
| --- | --- | --- |
| `onnxruntime/test/python/contrib_ops/<file>.py` | `onnxruntime/test/python/contrib_ops/<file>/<test_case>/` | Python runtime instrumentation |
| `onnxruntime/test/python/<file>.py` | `onnxruntime/test/python/<file>/<test_case>/` | Python runtime instrumentation |
| `onnxruntime/test/contrib_ops/<file>.cc` | `onnxruntime/test/contrib_ops/<file>/<test_name>_run<n>/` | Runtime `OpTester` wrapper |
| `onnxruntime/test/testdata/<name>/` | `onnxruntime/test/testdata/<name>/` | Static copy from ORT checkout |
| `onnxruntime/test/providers/<name>/` | `onnxruntime/test/providers/<name>/` | Static copy from ORT checkout |

## File Formats

- `model.onnx`: binary ONNX `ModelProto`.
- `input_*.pb` and `output_*.pb`: binary `TensorProto` for dense tensors, or
  `SparseTensorProto` for sparse inputs.
- `validation.json`: replay metadata captured from the originating ORT test.
  In the current checked-in dataset, every test-case directory has this file.
  The Python validation code still tolerates a missing file as a compatibility
  fallback and then uses empty/default metadata.

## `validation.json`

`validation.json` describes how a checked-in test case should be interpreted
during validation. The current schema is:

```json
{
  "expects_failure": false,
  "expected_failure_substring": "",
  "included_providers": ["CPU"],
  "excluded_providers": ["CUDA"],
  "outputs": [
    {
      "name": "Y",
      "relative_error": null,
      "absolute_error": null,
      "sort_output": false
    }
  ]
}
```

Field meanings:

- `expects_failure`: marks a case captured from an ORT test that expected
  failure. The current validator treats such a case as valid without replaying
  output comparisons once the artifact structure is otherwise loadable.
- `expected_failure_substring`: optional substring that must appear in the
  runtime error when `expects_failure` is true and an exception is actually
  observed during model load or execution.
- `included_providers`: optional normalized provider names explicitly used by
  the originating ORT test. These are currently informational only; the Python
  validator always replays with `CPUExecutionProvider`.
- `excluded_providers`: optional normalized provider names explicitly excluded
  by the originating ORT test. These are currently informational only.
- `outputs`: list of per-output comparison rules. Each entry is identified by
  its `name` field and matched against the model output name during validation.
- `outputs[].relative_error`: optional relative tolerance override.
- `outputs[].absolute_error`: optional absolute tolerance override.
- `outputs[].sort_output`: whether validation should compare this output after
  flattening and sorting it.

If both provider lists are absent, the original ORT test used its default
provider selection. The runtime extractor currently writes `expects_failure`,
`expected_failure_substring`, and `outputs` for every generated case, and only
emits the provider lists when they are non-empty.

## Maintainer Context

The artifact refresh and validation commands live in `tools/scripts/` and are
documented in [`../DEVELOPMENT.md`](../DEVELOPMENT.md). Those scripts are
repository maintenance infrastructure, not the main product.
