#!/usr/bin/env python3
"""Generate a Markdown overview from artifact validation expectations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "tools" / "python"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from emx_ort_test_materializer.artifact_validation_overview import (
    load_cases,
    render_overview_markdown,
)
from emx_ort_test_materializer.ignored_artifact_cases import load_ignored_artifact_cases


DEFAULT_EXPECTATIONS_PATH = REPO_ROOT / "tests" / "artifact_validation_expected.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "artifacts" / "VALIDATION_ERRORS.md"
DEFAULT_IGNORED_CASES_PATH = REPO_ROOT / "artifact_generation_ignored_cases.json"


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


def main() -> int:
    """Generate the Markdown overview file."""
    args = parse_args()
    cases = load_cases(args.expectations)
    ignored_cases = load_ignored_artifact_cases(DEFAULT_IGNORED_CASES_PATH)
    markdown = render_overview_markdown(
        cases,
        repo_root=REPO_ROOT,
        expectations_path=args.expectations.resolve(),
        ignored_cases=ignored_cases,
    )
    args.output.write_text(markdown, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
