"""Render Markdown operator summaries for tracked ONNX artifact test cases."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re

from emx_ort_test_materializer.operator_inventory import (
    OperatorCase,
    load_operator_cases,
    operator_sort_key,
    split_operator_label,
)

DOCS_REGEN_COMMAND = "python tools/scripts/generate_operator_markdown.py"
RUN_SUFFIX_PATTERN = re.compile(r"^(?P<prefix>.+)_run(?P<run>\d+)$")
CONTRIB_OPERATORS_DOC_URL = (
    "https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md"
)
ONNX_OPERATORS_DOC_BASE_URL = "https://onnx.ai/onnx/operators"


@dataclass(frozen=True)
class SingleOperatorCase:
    """One artifact test case together with its single operator."""

    path: str
    operator: str
    onnx_bytes: int
    pb_bytes: int


def generated_header() -> list[str]:
    """Return the standard auto-generated file header."""
    return [
        "<!-- AUTO-GENERATED FILE. DO NOT EDIT. -->",
        f"<!-- Regenerate with: {DOCS_REGEN_COMMAND} -->",
        "",
    ]


def escape_markdown_cell(value: str) -> str:
    """Escape Markdown table content."""
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").strip()


def operator_markdown_label(domain: str, op_type: str) -> str:
    """Return the Markdown label for one operator cell."""
    if domain == "com.microsoft":
        url = f"{CONTRIB_OPERATORS_DOC_URL}#com.microsoft.{op_type}"
        return f"[{escape_markdown_cell(op_type)}]({url})"
    if domain == "ai.onnx":
        url = f"{ONNX_OPERATORS_DOC_BASE_URL}/onnx__{op_type}.html"
        return f"[{escape_markdown_cell(op_type)}]({url})"
    return escape_markdown_cell(op_type)


def load_single_operator_cases(artifacts_root: Path) -> list[SingleOperatorCase]:
    """Load cases and require exactly one operator per artifact model."""
    matrix_cases = load_operator_cases(artifacts_root)
    result: list[SingleOperatorCase] = []
    for case in matrix_cases:
        result.append(single_operator_case(case, artifacts_root=artifacts_root))
    return result


def single_operator_case(case: OperatorCase, *, artifacts_root: Path) -> SingleOperatorCase:
    """Convert one matrix case into its single-operator representation."""
    if len(case.operators) != 1:
        raise ValueError(
            f"Expected exactly one operator for {case.path}, found {len(case.operators)}: "
            f"{', '.join(case.operators)}"
        )
    case_dir = artifacts_root / Path(case.path)
    return SingleOperatorCase(
        path=case.path,
        operator=case.operators[0],
        onnx_bytes=sum(path.stat().st_size for path in case_dir.rglob("*.onnx")),
        pb_bytes=sum(path.stat().st_size for path in case_dir.rglob("*.pb")),
    )


def compress_run_numbers(runs: list[int]) -> str:
    """Compress sorted run numbers into bracket range syntax."""
    ranges: list[str] = []
    start = runs[0]
    end = runs[0]
    for run in runs[1:]:
        if run == end + 1:
            end = run
            continue
        ranges.append(format_run_range(start, end))
        start = end = run
    ranges.append(format_run_range(start, end))
    return ",".join(ranges)


def format_run_range(start: int, end: int) -> str:
    """Format one inclusive run-number range."""
    if start == end:
        return str(start)
    return f"{start}-{end}"


def summarize_test_case_paths(paths: list[str]) -> list[str]:
    """Group per-run paths into compact `_run[...]` forms when possible."""
    grouped_runs: dict[str, list[int]] = defaultdict(list)
    passthrough: list[str] = []

    for path in sorted(paths):
        match = RUN_SUFFIX_PATTERN.match(path)
        if match is None:
            passthrough.append(path)
            continue
        grouped_runs[match.group("prefix")].append(int(match.group("run")))

    summarized = list(sorted(passthrough))
    for prefix, runs in sorted(grouped_runs.items()):
        sorted_runs = sorted(runs)
        if len(sorted_runs) == 1:
            summarized.append(f"{prefix}_run{sorted_runs[0]}")
        else:
            summarized.append(f"{prefix}_run[{compress_run_numbers(sorted_runs)}]")
    return summarized


def group_cases_by_operator(cases: list[SingleOperatorCase]) -> dict[str, list[SingleOperatorCase]]:
    """Return all test cases grouped by operator."""
    grouped: dict[str, list[SingleOperatorCase]] = defaultdict(list)
    for case in cases:
        grouped[case.operator].append(case)
    return {
        operator: sorted(grouped_cases, key=lambda case: case.path)
        for operator, grouped_cases in sorted(
            grouped.items(), key=lambda item: operator_sort_key(item[0])
        )
    }


def render_operator_markdown(
    cases: list[SingleOperatorCase],
    *,
    repo_root: Path,
    artifacts_root: Path,
) -> str:
    """Render the operator summary as Markdown."""
    try:
        artifacts_root_label = artifacts_root.relative_to(repo_root).as_posix()
    except ValueError:
        artifacts_root_label = artifacts_root.as_posix()

    grouped = group_cases_by_operator(cases)
    lines = [
        *generated_header(),
        "# ORT operator overview",
        "",
        "Summarizes tracked artifact test cases under a single-operator assumption.",
        "",
        f"Artifact root: `{artifacts_root_label}`",
        "",
        f"Tracked test cases: {len(cases)}. Distinct operators: {len(grouped)}.",
        "",
        "## Operator counts",
        "",
        "| Domain | Operator | Test cases | .onnx bytes | .pb bytes |",
        "| --- | --- | --- | --- | --- |",
    ]

    for operator, grouped_cases in grouped.items():
        domain, op_type = split_operator_label(operator)
        test_case_count = len(grouped_cases)
        onnx_bytes = sum(case.onnx_bytes for case in grouped_cases)
        pb_bytes = sum(case.pb_bytes for case in grouped_cases)
        lines.append(
            f"| {escape_markdown_cell(domain)} | "
            f"{operator_markdown_label(domain, op_type)} | "
            f"{test_case_count} | "
            f"{onnx_bytes} | "
            f"{pb_bytes} |"
        )

    lines.extend(
        [
            "",
            "## Operator test cases",
            "",
            "| Domain | Operator | Test cases |",
            "| --- | --- | --- |",
        ]
    )

    for operator, grouped_cases in grouped.items():
        domain, op_type = split_operator_label(operator)
        summarized_paths = summarize_test_case_paths([case.path for case in grouped_cases])
        lines.append(
            f"| {escape_markdown_cell(domain)} | "
            f"{operator_markdown_label(domain, op_type)} | "
            f"{escape_markdown_cell('<br>'.join(summarized_paths))} |"
        )

    lines.append("")
    return "\n".join(lines)
