# emx-ort-test-materializer

A lightweight internal utility to materialize ONNX and TensorProto `.pb` artifacts
from the official [ONNX Runtime](https://github.com/microsoft/onnxruntime) test suite —
covering **both Python and C++ tests**.

---

## Motivation

ONNX Runtime ships a large test suite spanning Python scripts and C++ binaries.
Both kinds of tests exercise many operators and contrib-ops by constructing models
and feeding inputs either programmatically (Python) or through pre-existing test-data
files (C++).  There is currently no easy way to obtain the corresponding `.onnx` model
files and serialized `input_*.pb` / `output_*.pb` tensors as standalone artifacts
outside of a full ORT build.

This tool extracts those artifacts by:

- **Python tests** – instrumenting the Python test modules to intercept
  `InferenceSession` calls and capture model bytes and numpy tensors at runtime.
- **C++ tests** – executing ORT `OpTester`-based unit tests through an
  out-of-tree wrapper, capturing JSON metadata, and serializing `model.onnx`
  plus `input_*.pb` / `output_*.pb` artifacts per `Run()`.

The resulting artifacts are committed to this repository so that downstream consumers
(e.g. emmtrix compiler test-suites) can reference them without requiring a running ORT
installation or a full build.

---

## Scope

- Read test sources and data from a local ONNX Runtime checkout prepared on
  demand from the pinned `onnxruntime` version in `requirements.txt`.
- **Python tests**: instrument `InferenceSession` / helper calls to capture model and
  input/output tensors.
- **C++ tests**: discover and collect pre-existing test-data files (`.onnx`, `.pb`)
  referenced by C++ test sources, and execute `OpTester`-based Microsoft/contrib-op
  tests to materialize reusable JSON metadata and runtime artifacts.
- Write all artifacts to `artifacts/` in a layout that mirrors the ORT test directory tree.
- Commit the resulting artifacts to this repository for reproducible consumption.

---

## Non-Goals

- This is **not** a distributable Python package.
- It does **not** run the full ORT test-suite.
- It does **not** modify ONNX Runtime source code in the prepared checkout.
- It does **not** provide release automation or package publishing.

---

## Artifact Layout

Extracted artifacts are stored under `artifacts/` and mirror the ONNX Runtime
source tree:

```
artifacts/
  onnxruntime/
    test/
      python/
        contrib_ops/
          <test_file>/
            <test_case>/
              model.onnx
              test_data_set_0/
                input_0.pb
                input_1.pb   # if multiple inputs
                output_0.pb  # expected output (optional)
      testdata/
        <op_or_suite_name>/
          model.onnx
          test_data_set_0/
            input_0.pb
            output_0.pb
      providers/
        <provider_test_name>/
          model.onnx
          test_data_set_0/
            input_0.pb
```

See [`artifacts/README.md`](artifacts/README.md) for a detailed description of
the layout and naming conventions.

---

## Repository Structure

```
.
├── LICENSE
├── README.md
├── .gitignore
├── requirements.txt
├── AGENTS.md
├── .github/
│   └── copilot-instructions.md
├── scripts/
│   └── extract_test_artifacts.py     # builds and runs the runtime C++ extractor
├── cpp/
│   └── runtime_extractor/            # runtime C++ wrapper around ORT tests
├── src/
│   └── emx_ort_test_materializer/
│       ├── __init__.py
│       ├── extractor.py              # future extraction logic (Python + C++)
│       └── writer.py                 # future serialization logic
├── artifacts/
│   └── README.md                     # layout documentation
```

---

## Planned Next Steps

1. **Python test instrumentation** – wrap `onnxruntime.InferenceSession` and
   related helpers in the contrib-ops test files to intercept model bytes and
   numpy input arrays before execution.
2. **C++ runtime extraction** – execute ORT `OpTester`-based C++ tests and emit JSON
   metadata plus `model.onnx` / `input_*.pb` / `output_*.pb` artifacts per run.
3. **Serialize Python-test models** – write captured ONNX `ModelProto` objects to
   `model.onnx` using `model.SerializeToString()`.
4. **Serialize Python-test tensors** – convert numpy arrays to `onnx.TensorProto`
   with `numpy_helper.from_array()` and write them as `.pb` files.
5. **Map to artifact paths** – derive the output path from the test file name and
   test-case name, following the artifact layout described above.
6. **Validate** – spot-check a handful of produced artifacts with
   `onnxruntime.InferenceSession` to confirm round-trip correctness.

---

## Current C++ Extractor

The runtime extractor under `cpp/runtime_extractor/` builds a small out-of-tree
ORT test runner, executes the selected C++ unit tests, and captures the real
`OpTester` models and tensors from the running process.

Current behavior:

- Clones or updates a local ONNX Runtime checkout under `build/` based on the
  pinned `onnxruntime` package version.
- Builds ONNX Runtime test utilities without modifying that checkout in place.
- Generates one wrapper target per selected ORT test source.
- Builds those wrapper targets in parallel after the shared ORT libraries are ready.
- Scans only `OpTester`-based C++ sources in directory mode.
- Redirects `OpTester` usage through a capturing subclass.
- Writes `model.onnx`, `test_data_set_0/input_*.pb`, and `test_data_set_0/output_*.pb`
  under `artifacts/` by default. Dense inputs/outputs use `TensorProto`; sparse
  inputs use `SparseTensorProto`.
- Serializes the same captured tensors as base64-encoded raw bytes in JSON.

Artifact directories use the ORT-relative source path and the per-run suffix
`<test_name>_run<n>` as the lowest directory level. For example:

```text
artifacts/onnxruntime/test/contrib_ops/qlinear_lookup_table_test/QLinearLeakyRelu_Int8_run0/
```

Example:

```bash
python scripts/extract_test_artifacts.py \
  --cpp-source onnxruntime/test/contrib_ops/qlinear_lookup_table_test.cc \
  --artifacts-output artifacts \
  --gtest-filter QLinearLookupTableBasedOperatorTests.* \
  --jobs 4
```

The runtime command always emits JSON alongside the artifacts. By default, it
writes to `build/ort_runtime_contrib_ops.json`, artifacts are written under
`artifacts/`, and temporary build products stay under the ignored `build/`
directory. In directory mode, the extractor now parallelizes both the wrapper
build step and the per-source test execution via `--jobs`.

Tracked exceptions for generated artifact cases live in
`artifact_generation_ignored_cases.json`. Each entry names the artifact case
path relative to `artifacts/` together with the reason it is currently ignored.
The extraction and validation-overview scripts both use this file, so ignored
cases are skipped consistently and listed in the generated Markdown overview.

Each artifact directory may also contain a `validation.json` file with replay
metadata captured from the original ORT test. The file currently uses this
shape:

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

Notes:

- `included_providers` is present only when the original test explicitly ran on
  a specific provider list.
- `excluded_providers` is present only when the original test explicitly
  excluded providers.
- Provider names are normalized to short ORT-independent labels such as `CPU`,
  `CUDA`, `DML`, `TensorRT`, `ROCM`, `WebGPU`, and `XNNPACK`.
- If neither field is present, the original test used ORT's implicit default
  provider selection.

---

## License

MIT – see [LICENSE](LICENSE).
