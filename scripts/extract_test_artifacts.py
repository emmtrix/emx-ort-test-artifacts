#!/usr/bin/env python3
"""
scripts/extract_test_artifacts.py
-----------------------------------
Entry point for materializing ONNX and TensorProto artifacts from ONNX
Runtime tests — both Python-based and C++-based test suites.

Intended future functionality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Discover test sources across the ONNX Runtime tree:

   * **Python tests** – modules under
     ``onnxruntime-org/onnxruntime/onnxruntime/test/python/``.
     Instrumented at runtime by monkey-patching
     ``onnxruntime.InferenceSession``.

   * **C++ tests** – test data and model files referenced by C++ test
     binaries, located under paths such as
     ``onnxruntime-org/onnxruntime/onnxruntime/test/testdata/``,
     ``onnxruntime-org/onnxruntime/onnxruntime/test/providers/``, and
     operator-specific subdirectories.

2. For Python tests: execute each module under instrumentation to intercept
   model bytes and numpy input/output tensors.

3. For C++ tests: parse or copy pre-existing test-data files (``*.onnx``,
   ``*.pb``) referenced by the C++ sources.

4. Write all artifacts to ``artifacts/`` in a layout that mirrors the ORT
   source tree.  See ``artifacts/README.md`` for the canonical layout.

5. Report a summary of produced artifacts.

This script is a **placeholder only**.  Do not add extraction logic here until
the instrumentation approach is agreed upon.  See ``AGENTS.md`` for the full
implementation roadmap.

Usage (future)
~~~~~~~~~~~~~~
    python scripts/extract_test_artifacts.py
"""


def main() -> None:
    """Placeholder entry point.  No logic is executed yet."""
    print("emx-ort-test-materializer: extraction not implemented yet.")
    print("See AGENTS.md for the planned implementation steps.")


if __name__ == "__main__":
    main()
