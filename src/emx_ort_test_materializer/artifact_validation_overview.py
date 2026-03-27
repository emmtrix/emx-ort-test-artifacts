"""Render Markdown summaries for artifact validation expectations."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from emx_ort_test_materializer.ignored_artifact_cases import (
    IgnoredArtifactCase,
)

DOCS_REGEN_COMMAND = (
    "UPDATE_REFS=1 pytest -q tests/test_artifact_validation_docs.py::test_artifact_validation_error_doc"
)


def generated_header() -> list[str]:
    """Return the standard auto-generated file header."""
    return [
        "<!-- AUTO-GENERATED FILE. DO NOT EDIT. -->",
        f"<!-- Regenerate with: {DOCS_REGEN_COMMAND} -->",
        "",
    ]


def escape_markdown_cell(value: str) -> str:
    """Escape Markdown table cell content."""
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").strip()


def is_success_result(result: str) -> bool:
    """Return whether a stored result string represents a successful validation."""
    return result == "OK"


def source_label(case_path: str) -> str:
    """Return the source-style grouping label for one artifact case path."""
    path = Path(case_path)
    return path.parent.as_posix()


def artifact_case_display_path(path: str) -> str:
    """Return the display path used in repository documentation."""
    normalized = path.strip().strip("/")
    return f"artifacts/{normalized}"


def ignored_case_source_label(case: IgnoredArtifactCase) -> str:
    """Return the source-style grouping label for one ignored artifact case."""
    return source_label(artifact_case_display_path(case.path))


def summarize_error(result: str) -> str:
    """Reduce verbose result strings to stable histogram buckets."""
    if result == "OK":
        return "OK"
    if "input count mismatch" in result:
        return "Input count mismatch"
    if "output count mismatch" in result:
        return "Output count mismatch"
    if ": FAIL - values differ" in result:
        return "Values differ"
    if ": FAIL - shape mismatch:" in result:
        return "Shape mismatch"
    if ": FAIL - dtype mismatch:" in result:
        return "Dtype mismatch"
    if result.startswith("[ONNXRuntimeError]"):
        return "ONNX Runtime error"
    return re.sub(r"'[^']*'", "'*'", result)


def load_cases(expectations_path: Path) -> list[dict[str, str]]:
    """Load test case expectations from JSON."""
    payload = json.loads(expectations_path.read_text(encoding="utf-8"))
    cases = payload["test_cases"]
    return sorted(cases, key=lambda case: case["path"])


def render_overview_markdown(
    cases: list[dict[str, str]],
    *,
    repo_root: Path,
    expectations_path: Path,
    ignored_cases: tuple[IgnoredArtifactCase, ...] = (),
) -> str:
    """Render the Markdown overview from expectation entries."""
    failing_cases = [case for case in cases if not is_success_result(case["expected_result"])]
    success_count = len(cases) - len(failing_cases)

    error_counts = Counter(summarize_error(case["expected_result"]) for case in failing_cases)
    error_sources: dict[str, set[str]] = {}
    error_source_pairs: list[tuple[str, str]] = []
    for case in failing_cases:
        summary = summarize_error(case["expected_result"])
        source = source_label(case["path"])
        error_sources.setdefault(summary, set()).add(source)
        error_source_pairs.append((summary, source))

    lines = [
        *generated_header(),
        "# ORT artifact validation errors",
        "",
        "Aggregates non-OK artifact validation outcomes.",
        "",
        f"Expectation source: `{expectations_path.relative_to(repo_root).as_posix()}`",
        "",
        f"Validated cases: {success_count} / {len(cases)} OK, {len(failing_cases)} non-OK.",
    ]
    if ignored_cases:
        lines.extend(
            [
                "",
                f"Ignored artifact generation cases: {len(ignored_cases)}.",
            ]
        )

    lines.extend(
        [
            "",
            "| Error message | Count | Sources |",
            "| --- | --- | --- |",
        ]
    )

    for error, count in sorted(error_counts.items(), key=lambda item: (-item[1], item[0])):
        sources = ", ".join(sorted(error_sources.get(error, set())))
        lines.append(
            f"| {escape_markdown_cell(error)} | {count} | {escape_markdown_cell(sources)} |"
        )

    pair_counts = Counter(error_source_pairs)
    lines.extend(
        [
            "",
            "## Error frequency by source",
            "",
            "| Error message | Source | Count |",
            "| --- | --- | --- |",
        ]
    )

    for (error, source), count in sorted(pair_counts.items(), key=lambda item: (item[0][1], -item[1], item[0][0])):
        lines.append(
            f"| {escape_markdown_cell(error)} | {escape_markdown_cell(source)} | {count} |"
        )

    lines.extend(
        [
            "",
            "## Failing artifact cases",
            "",
            "Lists every artifact case with a non-OK expected validation result.",
            "",
            "| Case | Source | Error |",
            "| --- | --- | --- |",
        ]
    )

    for case in failing_cases:
        lines.append(
            "| "
            f"{escape_markdown_cell(case['path'])} | "
            f"{escape_markdown_cell(source_label(case['path']))} | "
            f"{escape_markdown_cell(case['expected_result'])} |"
        )

    if ignored_cases:
        lines.extend(
            [
                "",
                "## Ignored Artifact Generation Cases",
                "",
                "Lists configured artifact cases that generation skips, together with the tracked reason.",
                "",
                "| Case | Source | Reason |",
                "| --- | --- | --- |",
            ]
        )
        for case in ignored_cases:
            display_path = artifact_case_display_path(case.path)
            lines.append(
                "| "
                f"{escape_markdown_cell(display_path)} | "
                f"{escape_markdown_cell(ignored_case_source_label(case))} | "
                f"{escape_markdown_cell(case.reason)} |"
            )

    lines.append("")
    return "\n".join(lines)
