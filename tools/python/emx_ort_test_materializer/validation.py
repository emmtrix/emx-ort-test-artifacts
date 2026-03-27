"""Validation helpers for replaying extracted ONNX test case artifacts."""

from __future__ import annotations

import ctypes
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


def dataset_sort_key(path: Path) -> tuple[int, str]:
    """Sort test_data_set directories by numeric suffix."""
    match = re.fullmatch(r"test_data_set_(\d+)", path.name)
    if match is None:
        return (sys.maxsize, path.name)
    return (int(match.group(1)), path.name)


def tensor_file_sort_key(path: Path) -> tuple[int, str]:
    """Sort input_*.pb and output_*.pb files by numeric suffix."""
    match = re.fullmatch(r"(?:input|output)_(\d+)\.pb", path.name)
    if match is None:
        return (sys.maxsize, path.name)
    return (int(match.group(1)), path.name)


@dataclass(frozen=True)
class OutputValidation:
    """Per-output replay tolerance captured from ORT's BaseTester."""

    name: str
    absolute_error: float | None = None
    relative_error: float | None = None
    sort_output: bool = False


@dataclass(frozen=True)
class TestCaseValidation:
    """Validation metadata captured alongside one extracted test case."""

    outputs: dict[str, OutputValidation]
    expects_failure: bool = False
    expected_failure_substring: str = ""
    included_providers: tuple[str, ...] = ()
    excluded_providers: tuple[str, ...] = ()


def default_tolerances(dtype: np.dtype) -> tuple[float, float]:
    """Return ORT-like default tolerances for floating-point outputs."""
    if dtype == np.dtype(np.float64):
        return 1e-5, 1e-5
    if dtype == np.dtype(np.float32):
        return 1e-5, 1e-4
    if dtype == np.dtype(np.float16):
        return 2.5e-3, 1e-3
    return 1e-6, 1e-5


def load_tensor_proto(path: Path) -> onnx.TensorProto:
    """Load one TensorProto from disk."""
    tensor = onnx.TensorProto()
    tensor.ParseFromString(path.read_bytes())
    return tensor


def load_sparse_tensor_proto(path: Path) -> onnx.SparseTensorProto:
    """Load one SparseTensorProto from disk."""
    tensor = onnx.SparseTensorProto()
    tensor.ParseFromString(path.read_bytes())
    return tensor


def load_tensor_proto_array(path: Path) -> np.ndarray:
    """Load a TensorProto from disk and convert it to a numpy array."""
    tensor = load_tensor_proto(path)
    return numpy_helper.to_array(tensor)


def sparse_tensor_from_proto(path: Path) -> ort.SparseTensor:
    """Load a SparseTensorProto from disk and create an ORT SparseTensor on CPU."""
    tensor = load_sparse_tensor_proto(path)
    dense_shape = np.asarray(tensor.dims, dtype=np.int64)
    values = numpy_helper.to_array(tensor.values)
    indices = numpy_helper.to_array(tensor.indices)
    cpu_device = ort.OrtDevice.make("cpu", 0)
    if indices.ndim == 1:
        return ort.SparseTensor.sparse_coo_from_numpy(dense_shape, values, indices, cpu_device)
    if indices.ndim == 2:
        return ort.SparseTensor.sparse_coo_from_numpy(
            dense_shape,
            values,
            np.ascontiguousarray(indices.reshape(-1)),
            cpu_device,
        )
    raise ValueError(f"Unsupported sparse indices rank {indices.ndim} in {path}")


def load_validation_metadata(test_case_dir: Path) -> TestCaseValidation:
    """Load per-output validation settings if present."""
    validation_path = test_case_dir / "validation.json"
    if not validation_path.is_file():
        return TestCaseValidation(outputs={})

    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    outputs = payload.get("outputs", [])
    metadata: dict[str, OutputValidation] = {}
    for output in outputs:
        metadata[output["name"]] = OutputValidation(
            name=output["name"],
            absolute_error=output.get("absolute_error"),
            relative_error=output.get("relative_error"),
            sort_output=bool(output.get("sort_output", False)),
        )
    return TestCaseValidation(
        outputs=metadata,
        expects_failure=bool(payload.get("expects_failure", False)),
        expected_failure_substring=str(payload.get("expected_failure_substring", "")),
        included_providers=tuple(
            str(provider) for provider in payload.get("included_providers", [])
        ),
        excluded_providers=tuple(
            str(provider) for provider in payload.get("excluded_providers", [])
        ),
    )


def maybe_sort_array(array: np.ndarray, *, enabled: bool) -> np.ndarray:
    """Return a flattened sorted view when order-insensitive output validation is requested."""
    if not enabled:
        return array
    flattened = np.asarray(array).reshape(-1)
    return np.sort(flattened, axis=None)


def compare_arrays(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float | None,
    rtol: float | None,
    sort_output: bool = False,
) -> tuple[bool, str]:
    """Compare two numpy arrays and return a status plus a concise message."""
    if actual.shape != expected.shape:
        return False, f"shape mismatch: actual={actual.shape}, expected={expected.shape}"

    if actual.dtype != expected.dtype:
        return False, f"dtype mismatch: actual={actual.dtype}, expected={expected.dtype}"

    actual_for_compare = maybe_sort_array(actual, enabled=sort_output)
    expected_for_compare = maybe_sort_array(expected, enabled=sort_output)

    if np.issubdtype(actual.dtype, np.inexact):
        if atol is None or rtol is None:
            default_atol, default_rtol = default_tolerances(expected.dtype)
            atol = default_atol if atol is None else atol
            rtol = default_rtol if rtol is None else rtol
        if np.allclose(
            actual_for_compare, expected_for_compare, rtol=rtol, atol=atol, equal_nan=True
        ):
            return True, "allclose"

        return False, "values differ"

    if np.array_equal(actual_for_compare, expected_for_compare):
        return True, "exact"

    if actual.dtype in {np.dtype(np.int8), np.dtype(np.uint8)}:
        max_abs_diff = np.max(
            np.abs(
                actual_for_compare.astype(np.int16) - expected_for_compare.astype(np.int16)
            )
        )
        if max_abs_diff <= 1:
            return True, "off-by-one"

    return False, "values differ"


def create_input_ort_value(input_proto: onnx.TensorProto) -> ort.OrtValue:
    """Create an OrtValue from one serialized input TensorProto."""
    array = numpy_helper.to_array(input_proto)
    fallback_onnx_type = input_proto.data_type
    fallback_dtype_names = {"int4", "uint4"}

    try:
        return ort.OrtValue.ortvalue_from_numpy(array)
    except RuntimeError:
        if array.dtype.name not in fallback_dtype_names:
            raise
        # For int4/uint4, numpy stores one element per byte (unpacked) but ORT
        # expects the data in the packed TensorProto wire format (two nibbles per
        # byte).  Create an OrtValue with the correct shape and element type,
        # then copy the raw packed bytes from the TensorProto directly into its
        # data buffer to avoid the unpacked-vs-packed mismatch.
        ort_value = ort.OrtValue.ortvalue_from_shape_and_type(
            list(input_proto.dims), fallback_onnx_type
        )
        raw_bytes = bytes(input_proto.raw_data)
        ctypes.memmove(ort_value.data_ptr(), raw_bytes, len(raw_bytes))
        return ort_value


def requires_ort_value_feed(input_proto: onnx.TensorProto) -> bool:
    """Return whether an input requires run_with_ort_values for replay."""
    array = numpy_helper.to_array(input_proto)
    return array.dtype.name in {"int4", "uint4"}


def is_sparse_input_type(input_type: str) -> bool:
    """Return whether a session input is declared as a sparse tensor."""
    return input_type.startswith("sparse_tensor(")


def matches_expected_failure(error: Exception, validation: TestCaseValidation) -> bool:
    """Return whether a runtime exception matches the captured expected failure."""
    if not validation.expects_failure:
        return False
    message = str(error)
    if not validation.expected_failure_substring:
        return True
    return validation.expected_failure_substring in message


def resolve_test_case_path(path: Path) -> tuple[Path, list[Path]]:
    """Resolve a user-provided test case path to the test case directory and datasets."""
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")

    if resolved.is_dir() and re.fullmatch(r"test_data_set_\d+", resolved.name):
        test_case_dir = resolved.parent
        datasets = [resolved]
    else:
        test_case_dir = resolved
        datasets = sorted(
            [
                entry
                for entry in test_case_dir.iterdir()
                if entry.is_dir() and re.fullmatch(r"test_data_set_\d+", entry.name)
            ],
            key=dataset_sort_key,
        )

    if not test_case_dir.is_dir():
        raise ValueError(f"Expected a test case directory or test_data_set directory: {resolved}")

    if not datasets:
        raise ValueError(f"No test_data_set_* directories found under: {test_case_dir}")

    return test_case_dir, datasets


def validate_dataset_result(
    session: ort.InferenceSession,
    dataset_dir: Path,
    test_case_validation: TestCaseValidation,
    *,
    atol: float | None,
    rtol: float | None,
) -> str:
    """Return OK or the first failure message for one dataset directory."""
    input_paths = sorted(dataset_dir.glob("input_*.pb"), key=tensor_file_sort_key)
    output_paths = sorted(dataset_dir.glob("output_*.pb"), key=tensor_file_sort_key)

    session_inputs = session.get_inputs()
    session_outputs = session.get_outputs()

    if len(input_paths) != len(session_inputs):
        return f"{dataset_dir.name}: input count mismatch: files={len(input_paths)}, model_inputs={len(session_inputs)}"

    if len(output_paths) != len(session_outputs):
        return f"{dataset_dir.name}: output count mismatch: files={len(output_paths)}, model_outputs={len(session_outputs)}"

    if test_case_validation.expects_failure:
        return "OK"

    use_ort_values = False
    input_tensors: list[onnx.TensorProto] = []
    sparse_inputs: list[ort.SparseTensor | None] = []
    for input_info, input_path in zip(session_inputs, input_paths, strict=True):
        if is_sparse_input_type(input_info.type):
            use_ort_values = True
            input_tensors.append(onnx.TensorProto())
            sparse_inputs.append(sparse_tensor_from_proto(input_path))
            continue

        input_proto = load_tensor_proto(input_path)
        input_tensors.append(input_proto)
        sparse_inputs.append(None)
        if requires_ort_value_feed(input_proto):
            use_ort_values = True

    try:
        if use_ort_values:
            feeds: dict[str, ort.OrtValue] = {}
            for input_info, input_proto, sparse_input in zip(
                session_inputs, input_tensors, sparse_inputs, strict=True
            ):
                if sparse_input is not None:
                    feeds[input_info.name] = ort.OrtValue.ort_value_from_sparse_tensor(
                        sparse_input
                    )
                else:
                    feeds[input_info.name] = create_input_ort_value(input_proto)
            computed_outputs = session.run_with_ort_values(
                [output.name for output in session_outputs],
                feeds,
            )
            actual_outputs = [np.asarray(output.numpy()) for output in computed_outputs]
        else:
            feeds = {}
            for input_info, input_proto in zip(session_inputs, input_tensors, strict=True):
                feeds[input_info.name] = numpy_helper.to_array(input_proto)
            computed_outputs = session.run(None, feeds)
            actual_outputs = [np.asarray(output) for output in computed_outputs]
    except Exception as error:
        if matches_expected_failure(error, test_case_validation):
            return "OK"
        return str(error)

    for index, (output_info, output_path, actual_output) in enumerate(
        zip(session_outputs, output_paths, actual_outputs, strict=True)
    ):
        expected_output = load_tensor_proto_array(output_path)
        metadata = test_case_validation.outputs.get(output_info.name)
        matches, detail = compare_arrays(
            actual_output,
            np.asarray(expected_output),
            atol=metadata.absolute_error if metadata and metadata.absolute_error is not None else atol,
            rtol=metadata.relative_error if metadata and metadata.relative_error is not None else rtol,
            sort_output=metadata.sort_output if metadata is not None else False,
        )
        if not matches:
            return f"{dataset_dir.name} output_{index} ({output_info.name}): FAIL - {detail}"

    return "OK"


def validate_test_case_result(
    test_case_path: Path,
    *,
    atol: float | None = None,
    rtol: float | None = None,
) -> str:
    """Return OK or the first failure message for one extracted test case."""
    try:
        test_case_dir, datasets = resolve_test_case_path(test_case_path)
        test_case_validation = load_validation_metadata(test_case_dir)
        model_path = test_case_dir / "model.onnx"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing model file: {model_path}")

        session_options = ort.SessionOptions()
        session_options.log_severity_level = 4
        try:
            session = ort.InferenceSession(
                model_path.as_posix(),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as error:
            if matches_expected_failure(error, test_case_validation):
                return "OK"
            raise
        for dataset_dir in datasets:
            result = validate_dataset_result(
                session,
                dataset_dir,
                test_case_validation,
                atol=atol,
                rtol=rtol,
            )
            if result != "OK":
                return result
        return "OK"
    except (FileNotFoundError, ValueError, Exception) as error:
        return str(error)
