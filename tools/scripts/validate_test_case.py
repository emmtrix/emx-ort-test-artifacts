#!/usr/bin/env python3
"""CLI wrapper for validating one extracted test case artifact directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "tools" / "python"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from emx_ort_test_materializer.validation import validate_test_case_result


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate one extracted test case against its stored output TensorProtos."
    )
    parser.add_argument(
        "test_case_path",
        type=Path,
        help="Path to a test case directory or a test_data_set_<n> directory.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for floating-point comparisons.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for floating-point comparisons.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the validation CLI."""
    args = parse_args()
    result = validate_test_case_result(
        args.test_case_path,
        atol=args.atol,
        rtol=args.rtol,
    )
    if result == "OK":
        print(f"Validation passed for {args.test_case_path.resolve()}")
        return 0

    print(result, file=sys.stderr)
    print(f"Validation failed for {args.test_case_path.resolve()}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
