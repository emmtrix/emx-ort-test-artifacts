#!/usr/bin/env python3
"""Generate a Markdown overview from artifact validation expectations."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPECTATIONS_PATH = REPO_ROOT / "tests" / "artifact_validation_expected.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "ARTIFACT_VALIDATION_ERRORS.md"
DOCS_REGEN_COMMAND = "python scripts/generate_artifact_validation_overview.py"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a Markdown overview from artifact validation expectations."
    )
    parser.add_argument(
        "--expectations",
        type=Path,
        default=DEFAULT_EXPECTATIONS_PATH,
        help="Path to the JSON file with expected artifact validation results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the generated Markdown overview.",
    )
    return parser.parse_args()


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


def render_overview_markdown(cases: list[dict[str, str]], expectations_path: Path) -> str:
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
        f"Expectation source: `{expectations_path.relative_to(REPO_ROOT).as_posix()}`",
        "",
        f"Validated cases: {success_count} / {len(cases)} OK, {len(failing_cases)} non-OK.",
        "",
        "| Error message | Count | Sources |",
        "| --- | --- | --- |",
    ]

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

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    """Generate the Markdown overview file."""
    args = parse_args()
    cases = load_cases(args.expectations)
    markdown = render_overview_markdown(cases, args.expectations.resolve())
    args.output.write_text(markdown, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
