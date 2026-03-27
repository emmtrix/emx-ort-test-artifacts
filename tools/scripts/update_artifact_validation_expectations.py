#!/usr/bin/env python3
"""Regenerate stored artifact validation expectations from checked-in artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "tools" / "python"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from emx_ort_test_materializer.validation import validate_test_case_result


DEFAULT_ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "tests" / "artifact_validation_expected.json"


def dataset_directory_sort_key(path: Path) -> tuple[int, str]:
    """Sort test_data_set directories by numeric suffix."""
    try:
        suffix = int(path.name.removeprefix("test_data_set_"))
    except ValueError:
        return (sys.maxsize, path.name)
    return (suffix, path.name)


def discover_test_case_directories(artifacts_root: Path) -> list[Path]:
    """Return all extracted test-case directories that can be replayed."""
    case_directories: list[Path] = []
    for model_path in sorted(artifacts_root.rglob("model.onnx")):
        test_case_dir = model_path.parent
        datasets = sorted(
            (
                entry
                for entry in test_case_dir.iterdir()
                if entry.is_dir() and entry.name.startswith("test_data_set_")
            ),
            key=dataset_directory_sort_key,
        )
        if datasets:
            case_directories.append(test_case_dir)
    return case_directories


def build_expectations_payload(
    artifacts_root: Path,
    *,
    repo_root: Path,
    atol: float | None,
    rtol: float | None,
) -> dict[str, list[dict[str, str]]]:
    """Validate each discovered artifact case and build the stored JSON payload."""
    test_cases = []
    for test_case_dir in discover_test_case_directories(artifacts_root):
        test_cases.append(
            {
                "path": test_case_dir.relative_to(repo_root).as_posix(),
                "expected_result": validate_test_case_result(
                    test_case_dir,
                    atol=atol,
                    rtol=rtol,
                ),
            }
        )

    return {
        "test_cases": sorted(test_cases, key=lambda case: case["path"]),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Regenerate tests/artifact_validation_expected.json from artifacts/."
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_ARTIFACTS_ROOT,
        help="Root directory containing extracted test-case artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the JSON file with expected validation results.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="Optional absolute tolerance override forwarded to validation.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="Optional relative tolerance override forwarded to validation.",
    )
    return parser.parse_args()


def main() -> int:
    """Regenerate the expectation JSON file."""
    args = parse_args()
    payload = build_expectations_payload(
        args.artifacts_root.resolve(),
        repo_root=REPO_ROOT,
        atol=args.atol,
        rtol=args.rtol,
    )
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
