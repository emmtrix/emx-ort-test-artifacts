"""
emx_ort_test_materializer
--------------------------
Maintainer-side helper modules for refreshing and validating the checked-in
ONNX Runtime artifact dataset stored under ``artifacts/``.

This package is **not** intended for distribution. It exists only to support
the maintenance scripts under ``tools/scripts/``.

Modules
~~~~~~~
- :mod:`extractor` – future logic for discovering ORT test sources (Python and
  C++) and intercepting or locating the model bytes and numpy tensors they use.
- :mod:`writer` – future logic for serializing ``ModelProto`` objects and
  numpy arrays to disk in the artifact layout.
"""
