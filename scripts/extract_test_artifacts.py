#!/usr/bin/env python3
"""
Build and run the runtime C++ extractor for ONNX Runtime OpTester-based tests.

This script compiles a small out-of-tree wrapper around selected ORT unit test
sources, executes those tests, writes per-run ONNX/TensorProto artifacts, and
optionally merges the per-file runtime JSON outputs into a single report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def repo_root() -> Path:
    """Return the repository root based on this script location."""
    return Path(__file__).resolve().parent.parent


def detect_cmake_binary() -> Path:
    """Locate a usable CMake binary for the current host platform."""
    cmake_on_path = shutil.which("cmake")
    if cmake_on_path:
        return Path(cmake_on_path)

    candidates = [
        Path(
            r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
        ),
        Path(
            r"C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
        ),
        Path(
            r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
        ),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Unable to locate a CMake binary required for the runtime extractor."
    )


def ort_test_root() -> Path:
    """Return the ONNX Runtime test root used as working directory for wrapped gtests."""
    return repo_root() / "onnxruntime-org" / "onnxruntime" / "test"


def relative_to_onnxruntime_org(path: Path) -> Path:
    """Return a path relative to the immutable onnxruntime-org checkout."""
    ort_root = repo_root() / "onnxruntime-org"
    return path.resolve().relative_to(ort_root.resolve())


def is_runtime_candidate(source_file: Path) -> bool:
    """Return whether a source file contains OpTester-based tests worth wrapping."""
    return "OpTester" in source_file.read_text(encoding="utf-8", errors="ignore")


def runtime_source_files(source_path: Path) -> list[Path]:
    """Return the runtime extractor input files in deterministic order."""
    if source_path.is_file():
        if not is_runtime_candidate(source_path):
            raise ValueError(
                f"Runtime extractor currently supports only OpTester-based sources: {source_path}"
            )
        return [source_path]

    return sorted(
        entry
        for entry in source_path.rglob("*")
        if entry.is_file() and entry.suffix.lower() in {".cc", ".cpp"}
        if is_runtime_candidate(entry)
    )


def sanitize_filename(path: Path) -> str:
    """Create a stable filename fragment from a relative source path."""
    return "".join(character if character.isalnum() else "_" for character in path.as_posix())


def format_command(command: list[str]) -> str:
    """Return one shell-style string representation for logging."""
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def run_logged_command(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
    """Print one external command before executing it."""
    print(f"> {format_command(command)}", flush=True)
    return subprocess.run(command, **kwargs)


def configure_runtime_extractor(cmake_binary: Path, build_dir: Path) -> None:
    """Configure the runtime extractor build once for the current workspace."""
    command = [
        str(cmake_binary),
        "-S",
        str(repo_root() / "cpp" / "runtime_extractor"),
        "-B",
        str(build_dir),
    ]
    if os.name == "nt":
        command.extend(
            [
                "-G",
                "Visual Studio 17 2022",
                "-A",
                "x64",
            ]
        )
    else:
        if shutil.which("ninja"):
            command.extend(["-G", "Ninja"])
        command.append("-DCMAKE_BUILD_TYPE=RelWithDebInfo")
    run_logged_command(command, check=True)


def write_runtime_capture_config(
    build_dir: Path,
    source_file: Path,
    source_root_relative: Path,
) -> None:
    """Write the per-source capture configuration header consumed by the runtime wrapper."""
    source_file_relative = relative_to_onnxruntime_org(source_file)
    ort_source_root = repo_root() / "onnxruntime-org" / "onnxruntime"
    generated_dir = build_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    config_path = generated_dir / "emx_runtime_capture_config.h"
    extra_includes_path = generated_dir / "emx_runtime_capture_extra_includes.h"

    include_pattern = re.compile(r'#include\s+"([^"]+)"')
    source_text = source_file.read_text(encoding="utf-8", errors="ignore")
    helper_sources: list[str] = []
    for match in include_pattern.finditer(source_text):
        include_path = match.group(1)
        if not include_path.endswith(".h"):
            continue
        candidate = ort_source_root / Path(include_path).with_suffix(".cc")
        if candidate.exists():
            helper_sources.append(candidate.resolve().as_posix())

    helper_sources = sorted(set(helper_sources))

    config_path.write_text(
        "\n".join(
            [
                "#pragma once",
                f'#define EMX_ORT_CAPTURE_TEST_SOURCE "{source_file.resolve().as_posix()}"',
                f'#define EMX_ORT_CAPTURE_SOURCE_FILE_REL "{source_file_relative.as_posix()}"',
                f'#define EMX_ORT_CAPTURE_SOURCE_ROOT_REL "{source_root_relative.as_posix()}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    extra_includes_path.write_text(
        "\n".join(
            ["#pragma once", *[f'#include "{path}"' for path in helper_sources], ""]
        ),
        encoding="utf-8",
    )


def build_runtime_extractor(cmake_binary: Path, build_dir: Path) -> Path:
    """Build the runtime extractor target and return the executable path."""
    command = [str(cmake_binary), "--build", str(build_dir), "--target", "ort_cpp_test_runtime_extractor"]
    if os.name == "nt":
        command.extend(["--config", "RelWithDebInfo"])
    run_logged_command(command, check=True)

    candidates = []
    if os.name == "nt":
        candidates.append(build_dir / "RelWithDebInfo" / "ort_cpp_test_runtime_extractor.exe")
    else:
        candidates.append(build_dir / "ort_cpp_test_runtime_extractor")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    fallback = next(build_dir.rglob("ort_cpp_test_runtime_extractor*"), None)
    if fallback is not None:
        return fallback

    raise FileNotFoundError(
        f"Unable to locate the built runtime extractor under {build_dir.resolve()}"
    )


def run_runtime_extractor(
    extractor_binary: Path,
    source_file: Path,
    output_path: Path,
    artifacts_output: Path,
    gtest_filter: str | None,
) -> subprocess.CompletedProcess[bytes]:
    """Execute the runtime extractor for one compiled C++ test source file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(extractor_binary),
        "--emx_output_json",
        str(output_path),
        "--emx_artifact_root",
        str(artifacts_output),
    ]

    if gtest_filter:
        command.append(f"--gtest_filter={gtest_filter}")

    return run_logged_command(
        command,
        check=False,
        cwd=ort_test_root(),
        capture_output=True,
    )


def write_runtime_merged_json(
    output_path: Path,
    source_root_relative: Path,
    artifacts_output: Path,
    files_attempted: int,
    files_succeeded: int,
    record_chunks: Iterable[dict],
    failed_files: Iterable[dict],
) -> None:
    """Merge per-file runtime JSON fragments into a single JSON file."""
    records: list[dict] = []
    for chunk in record_chunks:
        records.extend(chunk.get("records", []))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged = {
        "capture_mode": "runtime",
        "source_root": source_root_relative.as_posix(),
        "artifact_root": str(artifacts_output.resolve()),
        "files_attempted": files_attempted,
        "files_scanned": files_succeeded,
        "record_count": len(records),
        "failed_files": list(failed_files),
        "records": records,
    }

    output_path.write_text(json.dumps(merged, indent=2) + os.linesep, encoding="utf-8")


def load_runtime_json(path: Path) -> dict:
    """Load one per-file runtime JSON output with a precise error message."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise ValueError(f"Runtime JSON is not valid UTF-8: {path}") from error


def run_runtime_pipeline(
    source_path: Path,
    output_path: Path,
    artifacts_output: Path,
    rebuild: bool,
    gtest_filter: str | None,
) -> None:
    """Configure, build, execute, and merge runtime extraction output."""
    artifacts_output = artifacts_output.resolve()
    cmake_binary = detect_cmake_binary()
    build_dir = repo_root() / "build" / "ort_runtime_extractor"
    temp_output_dir = build_dir / "json"
    source_files = runtime_source_files(source_path)

    if not source_files:
        raise FileNotFoundError(f"No C++ source files found under {source_path}")

    if rebuild and build_dir.exists():
        shutil.rmtree(build_dir)

    if source_path.is_file():
        source_root_relative = relative_to_onnxruntime_org(source_path.parent)
    else:
        source_root_relative = relative_to_onnxruntime_org(source_path)

    merged_chunks: list[dict] = []
    failed_files: list[dict] = []

    configure_runtime_extractor(cmake_binary, build_dir)

    for index, source_file in enumerate(source_files):
        source_file_relative = relative_to_onnxruntime_org(source_file)
        temp_output = temp_output_dir / f"{index:04d}_{sanitize_filename(source_file_relative)}.json"

        try:
            write_runtime_capture_config(build_dir, source_file, source_root_relative)
            extractor_binary = build_runtime_extractor(cmake_binary, build_dir)
            runtime_result = run_runtime_extractor(
                extractor_binary,
                source_file,
                temp_output,
                artifacts_output,
                gtest_filter,
            )
            runtime_stdout = runtime_result.stdout.decode("utf-8", errors="replace")
            runtime_stderr = runtime_result.stderr.decode("utf-8", errors="replace")

            if runtime_result.returncode != 0 and not temp_output.exists():
                raise subprocess.CalledProcessError(
                    runtime_result.returncode,
                    runtime_result.args,
                    output=runtime_stdout,
                    stderr=runtime_stderr,
                )

            runtime_chunk = load_runtime_json(temp_output)
        except (subprocess.CalledProcessError, ValueError) as error:
            exit_code = error.returncode if isinstance(error, subprocess.CalledProcessError) else None
            failed_files.append(
                {
                    "source_file": source_file_relative.as_posix(),
                    "exit_code": exit_code,
                    "error": str(error),
                }
            )
            continue

        if runtime_result.returncode != 0:
            runtime_chunk.setdefault("warnings", []).append(
                f"gtest exited with code {runtime_result.returncode} after writing runtime JSON."
            )

        merged_chunks.append(runtime_chunk)

    if not merged_chunks:
        raise RuntimeError("Runtime extractor did not produce any successful per-file outputs.")

    write_runtime_merged_json(
        output_path,
        source_root_relative,
        artifacts_output,
        len(source_files),
        len(merged_chunks),
        merged_chunks,
        failed_files,
    )

    if failed_files:
        print(f"Runtime extractor skipped {len(failed_files)} source files due to build or execution failures.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Build and run the runtime C++ ONNX Runtime extractor."
    )
    parser.add_argument(
        "--cpp-source",
        type=Path,
        default=root / "onnxruntime-org" / "onnxruntime" / "test" / "contrib_ops",
        help="C++ test file or directory to scan.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=root / "build" / "ort_runtime_contrib_ops.json",
        help="JSON file to write.",
    )
    parser.add_argument(
        "--artifacts-output",
        type=Path,
        default=root / "artifacts",
        help="Artifact root directory.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a rebuild of the runtime extractor.",
    )
    parser.add_argument(
        "--gtest-filter",
        help="Optional gtest filter forwarded to the runtime extractor.",
    )
    return parser.parse_args()


def main() -> int:
    """Build and run the runtime extractor."""
    args = parse_args()

    try:
        run_runtime_pipeline(
            args.cpp_source,
            args.json_output,
            args.artifacts_output,
            args.rebuild,
            args.gtest_filter,
        )
    except subprocess.CalledProcessError as error:
        print(f"Extractor command failed with exit code {error.returncode}.", file=sys.stderr)
        return error.returncode
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1

    print(f"JSON written to {args.json_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
