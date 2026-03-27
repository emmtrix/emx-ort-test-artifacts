from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "tools" / "python"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from emx_ort_test_materializer.artifact_validation_overview import (
    DOCS_REGEN_COMMAND,
    load_cases,
    render_overview_markdown,
)
from emx_ort_test_materializer.ignored_artifact_cases import (
    IgnoredArtifactCase,
    load_ignored_artifact_cases,
)


EXPECTATIONS_PATH = REPO_ROOT / "tests" / "artifact_validation_expected.json"
OVERVIEW_PATH = REPO_ROOT / "artifacts" / "VALIDATION_ERRORS.md"
IGNORED_CASES_PATH = REPO_ROOT / "artifact_generation_ignored_cases.json"


def test_artifact_validation_error_doc() -> None:
    cases = load_cases(EXPECTATIONS_PATH)
    ignored_cases = load_ignored_artifact_cases(IGNORED_CASES_PATH)
    expected_markdown = render_overview_markdown(
        cases,
        repo_root=REPO_ROOT,
        expectations_path=EXPECTATIONS_PATH,
        ignored_cases=ignored_cases,
    )

    if os.environ.get("UPDATE_REFS") == "1":
        OVERVIEW_PATH.write_text(expected_markdown, encoding="utf-8")

    if not OVERVIEW_PATH.exists():
        pytest.fail(
            "artifacts/VALIDATION_ERRORS.md is missing. "
            f"Regenerate with: {DOCS_REGEN_COMMAND}"
        )

    actual_markdown = OVERVIEW_PATH.read_text(encoding="utf-8")
    assert actual_markdown == expected_markdown, (
        "artifacts/VALIDATION_ERRORS.md is stale. "
        f"Regenerate with: {DOCS_REGEN_COMMAND}"
    )


def test_render_overview_markdown_lists_ignored_cases() -> None:
    markdown = render_overview_markdown(
        [{"path": "artifacts/example/test_case", "expected_result": "OK"}],
        repo_root=REPO_ROOT,
        expectations_path=EXPECTATIONS_PATH,
        ignored_cases=(
            IgnoredArtifactCase(
                path="onnxruntime/test/contrib_ops/example/TestCase_run0",
                reason="Ignored for a tracked reason.",
            ),
        ),
    )

    assert "## Ignored Artifact Generation Cases" in markdown
    assert "artifacts/onnxruntime/test/contrib_ops/example/TestCase_run0" in markdown
    assert "Ignored for a tracked reason." in markdown
