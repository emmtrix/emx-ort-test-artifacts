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
- **C++ tests** – locating and copying (or converting) the test-data files already
  present in the ORT source tree that are referenced by C++ test binaries.

The resulting artifacts are committed to this repository so that downstream consumers
(e.g. emmtrix compiler test-suites) can reference them without requiring a running ORT
installation or a full build.

---

## Scope

- Read test sources and data from the bundled ONNX Runtime submodule.
- **Python tests**: instrument `InferenceSession` / helper calls to capture model and
  input/output tensors.
- **C++ tests**: discover and collect test-data files (`.onnx`, `.pb`) referenced by
  C++ test sources.
- Write all artifacts to `artifacts/` in a layout that mirrors the ORT test directory tree.
- Commit the resulting artifacts to this repository for reproducible consumption.

---

## Non-Goals

- This is **not** a distributable Python package.
- It does **not** run the full ORT test-suite or validate numerical results.
- It does **not** modify ONNX Runtime source code.
- It does **not** provide CI or automated publishing.
- It does **not** compile or execute C++ test binaries.

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
│   └── extract_test_artifacts.py     # entry point (placeholder)
├── src/
│   └── emx_ort_test_materializer/
│       ├── __init__.py
│       ├── extractor.py              # future extraction logic (Python + C++)
│       └── writer.py                 # future serialization logic
├── artifacts/
│   └── README.md                     # layout documentation
└── onnxruntime-org/
    └── onnxruntime/                  # git submodule
```

---

## Planned Next Steps

1. **Python test instrumentation** – wrap `onnxruntime.InferenceSession` and
   related helpers in the contrib-ops test files to intercept model bytes and
   numpy input arrays before execution.
2. **C++ test-data discovery** – scan the ORT source tree for `.onnx` and `.pb`
   files referenced by C++ test sources and copy them into `artifacts/` under the
   mirrored path.
3. **Serialize Python-test models** – write captured ONNX `ModelProto` objects to
   `model.onnx` using `model.SerializeToString()`.
4. **Serialize Python-test tensors** – convert numpy arrays to `onnx.TensorProto`
   with `numpy_helper.from_array()` and write them as `.pb` files.
5. **Map to artifact paths** – derive the output path from the test file name and
   test-case name, following the artifact layout described above.
6. **Validate** – spot-check a handful of produced artifacts with
   `onnxruntime.InferenceSession` to confirm round-trip correctness.

---

## License

MIT – see [LICENSE](LICENSE).
