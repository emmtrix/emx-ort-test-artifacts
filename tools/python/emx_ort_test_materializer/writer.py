"""
writer.py
---------
Future home of the artifact-serialization logic.

Planned responsibilities
~~~~~~~~~~~~~~~~~~~~~~~~
- Accept a ``ModelProto`` (or raw model bytes) and write it to
  ``<artifact_root>/model.onnx``.
- Accept a list of numpy arrays (inputs and/or outputs) and serialize each one
  as a ``TensorProto`` ``.pb`` file using ``onnx.numpy_helper.from_array()``.
  Files are written to
  ``<artifact_root>/test_data_set_<n>/input_<i>.pb`` and
  ``<artifact_root>/test_data_set_<n>/output_<i>.pb``.
- Create any missing intermediate directories automatically.

This module is a **placeholder only**.  No serialization logic is implemented
yet.  See ``AGENTS.md`` for the full implementation roadmap.
"""

from __future__ import annotations

import os


def write_model(artifact_dir: str, model_bytes: bytes) -> str:
    """Placeholder – will write a serialized ONNX model to disk.

    Parameters
    ----------
    artifact_dir:
        Directory in which ``model.onnx`` will be created.
    model_bytes:
        Raw serialized ``ModelProto`` bytes.

    Returns
    -------
    str
        Absolute path to the written ``model.onnx`` file.
    """
    # TODO: implement model serialization
    raise NotImplementedError("write_model is not implemented yet")


def write_tensors(
    artifact_dir: str,
    inputs: list,
    outputs: list,
    data_set_index: int = 0,
) -> list[str]:
    """Placeholder – will write input/output tensors as ``.pb`` files.

    Parameters
    ----------
    artifact_dir:
        Root artifact directory for the test case.
    inputs:
        List of numpy arrays representing model inputs.
    outputs:
        List of numpy arrays representing expected model outputs.
    data_set_index:
        Index of the test data set (default ``0``).

    Returns
    -------
    list[str]
        Absolute paths to the written ``.pb`` files.
    """
    # TODO: implement tensor serialization
    raise NotImplementedError("write_tensors is not implemented yet")


def artifact_path(base: str, test_file: str, test_case: str) -> str:
    """Return the canonical artifact directory for a test case.

    Parameters
    ----------
    base:
        Root of the ``artifacts/`` directory.
    test_file:
        Name of the test file (without ``.py`` extension).
    test_case:
        Name of the individual test case.

    Returns
    -------
    str
        Directory path where ``model.onnx`` and ``test_data_set_*/`` live.
    """
    return os.path.join(
        base,
        "onnxruntime",
        "test",
        "python",
        "contrib_ops",
        test_file,
        test_case,
    )
