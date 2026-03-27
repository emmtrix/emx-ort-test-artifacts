"""Pytest coverage for extracted ORT artifact replay expectations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "tools" / "python"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from emx_ort_test_materializer.validation import validate_test_case_result


EXPECTATIONS_PATH = REPO_ROOT / "tests" / "artifact_validation_expected.json"


def environment_error_message(result: str) -> str | None:
    """Return a normalized skip reason when validation failed due to local environment limits."""
    normalized = result.lower()
    environment_patterns = (
        "not supported on this hardware platform",
        "not yet supported on this hardware platform",
        "is not supported by execution provider",
        "failed to find kernel for",
        "kernel not found",
        "no op registered for",
        "not implemented",
        "no available kernel",
    )
    if any(pattern in normalized for pattern in environment_patterns):
        return result
    return None


def load_expectations() -> list[dict[str, str]]:
    """Load expected validation outcomes for all checked-in artifact test cases."""
    payload = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    cases = payload["test_cases"]
    return sorted(cases, key=lambda case: case["path"])


EXPECTED_CASES = load_expectations()


@pytest.mark.parametrize(
    ("relative_path", "expected_result"),
    [(case["path"], case["expected_result"]) for case in EXPECTED_CASES],
    ids=[case["path"] for case in EXPECTED_CASES],
)
def test_artifact_validation_result(relative_path: str, expected_result: str) -> None:
    """Replay one artifact directory and compare its result string against the stored expectation."""
    actual_result = validate_test_case_result(REPO_ROOT / relative_path)
    skip_reason = environment_error_message(actual_result)
    if skip_reason is not None:
        pytest.skip(skip_reason)
    assert actual_result == expected_result


def test_environment_error_message_classifies_local_runtime_limits() -> None:
    """Recognize environment-dependent ORT failures so pytest can skip them."""
    message = (
        "[ONNXRuntimeError] : 1 : FAIL : Load model failed: "
        "4b quantization not yet supported on this hardware platform!"
    )
    assert environment_error_message(message) == message


def test_environment_error_message_ignores_artifact_failures() -> None:
    """Keep artifact-specific validation errors as hard failures."""
    assert environment_error_message(
        "test_data_set_0: input count mismatch: files=1, model_inputs=2"
    ) is None
