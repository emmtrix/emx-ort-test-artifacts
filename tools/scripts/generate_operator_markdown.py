#!/usr/bin/env python3
"""Generate a Markdown operator overview from tracked artifact models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "tools" / "python"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from emx_ort_test_materializer.operator_markdown import (
    load_single_operator_cases,
    render_operator_markdown,
)


DEFAULT_ARTIFACTS_ROOT = REPO_ROOT / "artifacts" / "onnxruntime"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "artifacts" / "OPERATORS.md"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a Markdown operator overview from tracked artifact models."
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_ARTIFACTS_ROOT,
        help="Path to the tracked artifact root that contains model.onnx files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the generated Markdown file.",
    )
    return parser.parse_args()


def main() -> int:
    """Generate the Markdown operator overview file."""
    args = parse_args()
    cases = load_single_operator_cases(args.artifacts_root)
    markdown = render_operator_markdown(
        cases,
        repo_root=REPO_ROOT,
        artifacts_root=args.artifacts_root.resolve(),
    )
    args.output.write_text(markdown, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
