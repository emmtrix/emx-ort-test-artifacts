"""Tests for sparse input artifact replay helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "tools" / "python"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from emx_ort_test_materializer.validation import sparse_tensor_from_proto


def test_sparse_tensor_from_proto_loads_coo_sparse_tensor(tmp_path: Path) -> None:
    """Reconstruct an ORT COO sparse tensor from a serialized SparseTensorProto."""
    sparse_proto = onnx.SparseTensorProto()
    sparse_proto.dims.extend([3, 3])

    values = sparse_proto.values
    values.data_type = onnx.TensorProto.FLOAT
    values.dims.extend([2])
    values.float_data.extend([10.0, 20.0])

    indices = sparse_proto.indices
    indices.data_type = onnx.TensorProto.INT64
    indices.dims.extend([2])
    indices.int64_data.extend([0, 4])

    path = tmp_path / "input_0.pb"
    path.write_bytes(sparse_proto.SerializeToString())

    sparse_tensor = sparse_tensor_from_proto(path)

    assert list(sparse_tensor.dense_shape()) == [3, 3]
    assert np.asarray(sparse_tensor.values()).tolist() == [10.0, 20.0]
    assert sparse_tensor.as_coo_view().indices().tolist() == [0, 4]
