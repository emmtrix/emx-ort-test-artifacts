"""
emx_ort_test_materializer
--------------------------
Lightweight internal utility for materializing ONNX and TensorProto ``.pb``
artifacts from ONNX Runtime tests — covering **both Python and C++ test suites**.

This package is **not** intended for distribution.  It is a collection of
helper modules used by the scripts under ``scripts/``.

Modules
~~~~~~~
- :mod:`extractor` – future logic for discovering ORT test sources (Python and
  C++) and intercepting or locating the model bytes and numpy tensors they use.
- :mod:`writer` – future logic for serializing ``ModelProto`` objects and
  numpy arrays to disk in the artifact layout.
"""
