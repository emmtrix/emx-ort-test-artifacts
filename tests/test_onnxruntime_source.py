"""Unit tests for ONNX Runtime source checkout helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from emx_ort_test_materializer.onnxruntime_source import (
    default_onnxruntime_checkout_dir,
    onnxruntime_version_tag,
    read_pinned_onnxruntime_version,
)


def test_read_pinned_onnxruntime_version_matches_requirements() -> None:
    """Keep the source checkout version aligned with the pinned runtime package."""
    version = read_pinned_onnxruntime_version(REPO_ROOT / "requirements.txt")
    assert version == "1.24.4"


def test_onnxruntime_version_tag_derives_release_tag() -> None:
    """Map the pinned package version to the matching ORT release tag."""
    assert onnxruntime_version_tag("1.24.4") == "v1.24.4"


def test_default_onnxruntime_checkout_dir_uses_build_tree() -> None:
    """Keep the cloned ORT checkout out of the tracked repository tree."""
    assert default_onnxruntime_checkout_dir(REPO_ROOT) == REPO_ROOT / "build" / "onnxruntime-org"


def test_read_pinned_onnxruntime_version_requires_exact_pin(tmp_path: Path) -> None:
    """Reject requirements files that do not pin one exact ORT package version."""
    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text("onnxruntime>=1.24\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No exact 'onnxruntime=="):
        read_pinned_onnxruntime_version(requirements_path)
