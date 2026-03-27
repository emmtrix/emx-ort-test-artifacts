"""
extractor.py
------------
Future home of the test-discovery and artifact-extraction logic.

This module handles **both** Python-based and C++-based ONNX Runtime tests:

Python tests
~~~~~~~~~~~~
- Discover test modules under a prepared ONNX Runtime source checkout at paths
  such as ``onnxruntime/onnxruntime/test/python/`` and
  ``onnxruntime/onnxruntime/test/python/contrib_ops/``.
- Monkey-patch ``onnxruntime.InferenceSession`` (and related helpers) so that
  model bytes and numpy input/output arrays are intercepted before and after
  each test invocation.
- Yield ``(source_path, test_case, model_bytes, inputs, outputs)`` tuples for
  the writer module to serialize.

C++ tests
~~~~~~~~~
- Scan the ORT source tree for ``.onnx`` and ``.pb`` files that are referenced
  by C++ test sources, located under paths such as:

  * ``onnxruntime/onnxruntime/test/testdata/``
  * ``onnxruntime/onnxruntime/test/providers/``

- No C++ compilation or binary execution is required; files are discovered and
  copied statically.
- Yield ``(source_path, relative_artifact_path)`` pairs for the writer module.

This module is a **placeholder only**.  No extraction logic is implemented yet.
See ``AGENTS.md`` for the full implementation roadmap.
"""

from __future__ import annotations


def discover_python_test_files(base_path: str) -> list[str]:
    """Placeholder – will return paths to ORT Python test files.

    Parameters
    ----------
    base_path:
        Root of the ONNX Runtime source tree, e.g.
        ``"onnxruntime"``.

    Returns
    -------
    list[str]
        Absolute paths to Python test modules.  Always empty until implemented.
    """
    # TODO: implement discovery logic for Python test modules
    return []


def discover_cpp_test_data_files(base_path: str) -> list[tuple[str, str]]:
    """Placeholder – will return ORT C++ test-data files with their artifact paths.

    Scans directories such as ``onnxruntime/test/testdata/`` and
    ``onnxruntime/test/providers/`` for ``.onnx`` and ``.pb`` files without
    compiling or executing any C++ code.

    Parameters
    ----------
    base_path:
        Root of the ONNX Runtime source tree, e.g.
        ``"onnxruntime"``.

    Returns
    -------
    list[tuple[str, str]]
        List of ``(absolute_source_path, relative_artifact_path)`` pairs.
        Always empty until implemented.
    """
    # TODO: implement static discovery of C++ test-data files
    return []


def extract_python_test_artifacts(test_file: str) -> list[dict]:
    """Placeholder – will instrument a single Python test file and capture artifacts.

    Parameters
    ----------
    test_file:
        Absolute path to the Python test module to instrument.

    Returns
    -------
    list[dict]
        List of artifact records, each containing ``test_case``,
        ``model_bytes``, ``inputs``, and ``outputs`` keys.
        Always empty until implemented.
    """
    # TODO: implement instrumentation and extraction logic for Python tests
    return []
