# artifacts/

This directory stores the ONNX and TensorProto `.pb` artifacts that are
materialized from the ONNX Runtime test suite — covering **both Python and
C++ tests** — by the extraction scripts in this repository.

All files in this directory are **committed to version control** so that
downstream consumers can reference them without requiring a full ONNX Runtime
build or Python environment.

---

## Layout

Artifacts mirror the path structure of the ONNX Runtime source tree:

```
artifacts/
  onnxruntime/
    test/
      python/
        contrib_ops/
          <test_file>/         # name of the .py test file (without extension)
            <test_case>/       # name of the individual test case / sub-test
              model.onnx       # serialized ONNX ModelProto
              test_data_set_0/ # first set of inputs/outputs
                input_0.pb     # first input tensor (TensorProto binary)
                input_1.pb     # second input tensor (if present)
                output_0.pb    # expected output tensor (if captured)
      testdata/
        <op_or_suite_name>/    # mirrors onnxruntime/test/testdata/<name>/
          model.onnx
          test_data_set_0/
            input_0.pb
            output_0.pb
      providers/
        <provider_test_name>/  # mirrors onnxruntime/test/providers/<name>/
          model.onnx
          test_data_set_0/
            input_0.pb
            output_0.pb
```

### Source-to-Artifact Mapping

| ORT source path                                   | Artifact sub-path                              | Extraction method          |
|---------------------------------------------------|------------------------------------------------|----------------------------|
| `onnxruntime/test/python/contrib_ops/<file>.py`   | `test/python/contrib_ops/<file>/<test_case>/`  | Python instrumentation     |
| `onnxruntime/test/python/<file>.py`               | `test/python/<file>/<test_case>/`              | Python instrumentation     |
| `onnxruntime/test/contrib_ops/<file>.cc`          | `test/contrib_ops/<file>/<test_name>_run<n>/`  | Runtime `OpTester` wrapper |
| `onnxruntime/test/testdata/<name>/`               | `test/testdata/<name>/`                        | Static copy from submodule |
| `onnxruntime/test/providers/<name>/`              | `test/providers/<name>/`                       | Static copy from submodule |

### Naming Rules

| Segment              | Derivation                                                       |
|----------------------|------------------------------------------------------------------|
| `<test_file>`        | Python test module name, e.g. `test_conv`                       |
| `<test_case>`        | Python test method or parametrize ID, e.g. `test_conv_basic`    |
| `<test_name>_run<n>` | C++ gtest name plus zero-based `Run()` index, e.g. `Foo_run0`   |
| `input_<i>.pb`       | Inputs in the order passed to `InferenceSession.run()`          |
| `output_<i>.pb`      | Outputs in capture order; for C++ runtime mode these are the expected `OpTester` outputs |
| `test_data_set_<n>/` | Indexed from `0`; currently `test_data_set_0` per extracted run |

---

## File Formats

- **`model.onnx`** – binary serialization of an ONNX `ModelProto`.
  Can be loaded with `onnx.load("model.onnx")` or
  `onnxruntime.InferenceSession("model.onnx")`.
- **`*.pb`** – binary serialization of an ONNX `TensorProto` for dense tensors,
  or `SparseTensorProto` for sparse inputs.
  Dense tensors can be loaded with:
  ```python
  import onnx
  tensor = onnx.TensorProto()
  with open("input_0.pb", "rb") as f:
      tensor.ParseFromString(f.read())
  array = onnx.numpy_helper.to_array(tensor)
  ```
  Sparse tensors can be loaded with:
  ```python
  import onnx
  sparse = onnx.SparseTensorProto()
  with open("input_0.pb", "rb") as f:
      sparse.ParseFromString(f.read())
  ```

---

## Status

The runtime C++ extractor can already generate concrete artifacts for
`OpTester`-based contrib-op tests. The Python-side extraction pipeline and the
static mirroring of existing ORT `testdata/` and `providers/` assets remain
future work.
