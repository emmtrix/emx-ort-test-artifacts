"""Unit tests for parsing runtime artifact validation metadata."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "tools" / "python"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from emx_ort_test_materializer.validation import load_validation_metadata


def test_load_validation_metadata_includes_execution_provider_fields(tmp_path: Path) -> None:
    """Load the extended validation.json format including execution provider metadata."""
    test_case_dir = tmp_path / "Sample_run0"
    test_case_dir.mkdir()
    validation_path = test_case_dir / "validation.json"
    validation_path.write_text(
        json.dumps(
            {
                "expects_failure": False,
                "expected_failure_substring": "",
                "included_providers": [
                    "CUDA",
                    "CPU",
                ],
                "excluded_providers": [
                    "TensorRT",
                ],
                "outputs": [
                    {
                        "name": "output",
                        "relative_error": 0.1,
                        "absolute_error": 0.2,
                        "sort_output": True,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metadata = load_validation_metadata(test_case_dir)

    assert metadata.included_providers == (
        "CUDA",
        "CPU",
    )
    assert metadata.excluded_providers == ("TensorRT",)
    assert metadata.outputs["output"].relative_error == 0.1
    assert metadata.outputs["output"].absolute_error == 0.2
    assert metadata.outputs["output"].sort_output is True
