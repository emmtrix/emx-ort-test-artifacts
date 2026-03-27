#!/usr/bin/env python3
"""
Build and run the runtime C++ extractor for ONNX Runtime OpTester-based tests.

This script compiles out-of-tree wrapper executables around selected ORT unit
test sources, executes those tests, writes per-run ONNX/TensorProto artifacts,
and optionally merges the per-file runtime JSON outputs into a single report.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from emx_ort_test_materializer.ignored_artifact_cases import (
    IgnoredArtifactCase,
    ignored_case_reasons_by_path,
    load_ignored_artifact_cases,
)

DEFAULT_ORT_TEST_RANDOM_SEED = "1337"
DEFAULT_MAX_PARALLEL_JOBS = 8
MINIMUM_CMAKE_VERSION = (3, 28, 0)
DEFAULT_IGNORED_CASES_PATH = REPO_ROOT / "artifact_generation_ignored_cases.json"


@dataclass(frozen=True)
class RuntimeTargetSpec:
    """Describe one generated extractor target for a single ORT C++ source file."""

    index: int
    target_name: str
    source_file: Path
    source_file_relative: Path
    extra_includes_header: Path
    output_path: Path


def repo_root() -> Path:
    """Return the repository root based on this script location."""
    return REPO_ROOT


def detect_cmake_binary() -> Path:
    """Locate a usable CMake binary for the current host platform."""
    candidates = [
        Path(path)
        for path in [
            shutil.which("cmake"),
            r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
            r"C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
            r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        ]
        if path
    ]

    rejected_candidates: list[str] = []
    for candidate in candidates:
        if candidate.exists() and cmake_version_satisfies_minimum(candidate):
            return candidate
        if candidate.exists():
            rejected_candidates.append(candidate.as_posix())

    raise FileNotFoundError(
        "Unable to locate a CMake binary required for the runtime extractor "
        f"with version {format_version_tuple(MINIMUM_CMAKE_VERSION)} or newer."
        + (
            f" Rejected: {', '.join(rejected_candidates)}."
            if rejected_candidates
            else ""
        )
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


def parse_version_tuple(version_text: str) -> tuple[int, int, int] | None:
    """Parse one dotted semantic version string into a numeric tuple."""
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", version_text)
    if match is None:
        return None
    return tuple(int(component) for component in match.groups())


def format_version_tuple(version: tuple[int, int, int]) -> str:
    """Format one numeric version tuple as a dotted string."""
    return ".".join(str(component) for component in version)


def cmake_version_satisfies_minimum(cmake_binary: Path) -> bool:
    """Return whether the provided CMake binary satisfies the project minimum version."""
    try:
        result = subprocess.run(
            [str(cmake_binary), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False

    version = parse_version_tuple(result.stdout)
    return version is not None and version >= MINIMUM_CMAKE_VERSION


def default_parallel_jobs() -> int:
    """Return a conservative default for parallel build and execution jobs."""
    available = os.cpu_count() or 1
    return max(1, min(DEFAULT_MAX_PARALLEL_JOBS, available))


def runtime_target_name(index: int) -> str:
    """Return one deterministic CMake target name for a source-file wrapper."""
    return f"ort_cpp_test_runtime_extractor_{index:04d}"


def cmake_quote(value: str) -> str:
    """Return one CMake string literal."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


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


def helper_source_files(source_file: Path) -> list[Path]:
    """Return helper implementation files that should be included for one test source."""
    ort_source_root = repo_root() / "onnxruntime-org" / "onnxruntime"
    include_pattern = re.compile(r'#include\s+"([^"]+)"')
    source_text = source_file.read_text(encoding="utf-8", errors="ignore")
    helper_sources: list[Path] = []

    for match in include_pattern.finditer(source_text):
        include_path = match.group(1)
        if not include_path.endswith(".h"):
            continue
        candidate = ort_source_root / Path(include_path).with_suffix(".cc")
        if candidate.exists():
            helper_sources.append(candidate.resolve())

    return sorted(set(helper_sources))


def write_runtime_extra_includes_header(
    output_path: Path,
    helper_sources: Iterable[Path],
) -> None:
    """Write the helper-include header consumed by one extractor target."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                "#pragma once",
                *[f'#include "{helper_source.as_posix()}"' for helper_source in helper_sources],
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_runtime_target_manifest(
    build_dir: Path,
    source_files: list[Path],
    source_root_relative: Path,
) -> list[RuntimeTargetSpec]:
    """Write the generated CMake target manifest for all selected runtime sources."""
    generated_dir = build_dir / "generated"
    include_dir = generated_dir / "includes"
    json_dir = build_dir / "json"
    include_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    target_specs: list[RuntimeTargetSpec] = []
    manifest_lines = [
        "# Generated by scripts/extract_test_artifacts.py",
        "set(EMX_ORT_RUNTIME_EXTRACTOR_TARGETS",
    ]

    for index, source_file in enumerate(source_files):
        source_file_relative = relative_to_onnxruntime_org(source_file)
        sanitized_name = sanitize_filename(source_file_relative)
        extra_includes_header = include_dir / f"{index:04d}_{sanitized_name}_extra_includes.h"
        write_runtime_extra_includes_header(
            extra_includes_header,
            helper_source_files(source_file),
        )

        spec = RuntimeTargetSpec(
            index=index,
            target_name=runtime_target_name(index),
            source_file=source_file,
            source_file_relative=source_file_relative,
            extra_includes_header=extra_includes_header,
            output_path=json_dir / f"{index:04d}_{sanitized_name}.json",
        )
        target_specs.append(spec)
        manifest_lines.append(f"  {cmake_quote(spec.target_name)}")

    manifest_lines.append(")")
    manifest_lines.append("")

    for spec in target_specs:
        manifest_lines.append(
            "emx_add_runtime_extractor_target("
            f"{cmake_quote(spec.target_name)} "
            f"{cmake_quote(spec.source_file.resolve().as_posix())} "
            f"{cmake_quote(spec.source_file_relative.as_posix())} "
            f"{cmake_quote(source_root_relative.as_posix())} "
            f"{cmake_quote(spec.extra_includes_header.resolve().as_posix())}"
            ")"
        )

    manifest_lines.append("")
    manifest_lines.append(
        "add_custom_target(ort_cpp_test_runtime_extractors DEPENDS ${EMX_ORT_RUNTIME_EXTRACTOR_TARGETS})"
    )

    manifest_path = generated_dir / "emx_runtime_targets.cmake"
    manifest_path.write_text("\n".join(manifest_lines) + os.linesep, encoding="utf-8")
    return target_specs


def resolve_runtime_extractor_binary(build_dir: Path, target_name: str) -> Path:
    """Locate the built extractor executable for one generated target."""
    candidates: list[Path] = []
    if os.name == "nt":
        candidates.extend(
            [
                build_dir / "RelWithDebInfo" / f"{target_name}.exe",
                build_dir / f"{target_name}.exe",
            ]
        )
    else:
        candidates.extend(
            [
                build_dir / target_name,
                build_dir / "bin" / target_name,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    fallback = next(build_dir.rglob(f"{target_name}*"), None)
    if fallback is not None:
        return fallback

    raise FileNotFoundError(
        f"Unable to locate the built runtime extractor target {target_name} under {build_dir.resolve()}"
    )


def build_runtime_extractors(
    cmake_binary: Path,
    build_dir: Path,
    target_specs: list[RuntimeTargetSpec],
    jobs: int,
) -> dict[str, Path]:
    """Build all generated runtime extractor targets and return their executable paths."""
    command = [
        str(cmake_binary),
        "--build",
        str(build_dir),
        "--target",
        "ort_cpp_test_runtime_extractors",
        "--parallel",
        str(jobs),
    ]
    if os.name == "nt":
        command.extend(["--config", "RelWithDebInfo"])
    run_logged_command(command, check=True)

    return {
        spec.target_name: resolve_runtime_extractor_binary(build_dir, spec.target_name)
        for spec in target_specs
    }


def run_runtime_extractor(
    extractor_binary: Path,
    output_path: Path,
    artifacts_output: Path,
    gtest_filter: str | None,
) -> subprocess.CompletedProcess[bytes]:
    """Execute the runtime extractor for one compiled C++ test source file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_env = os.environ.copy()
    # ORT unit tests may generate inputs via RandomValueGenerator with a time-based
    # default seed, which would churn serialized TensorProto artifacts across runs.
    runtime_env.setdefault("ORT_TEST_RANDOM_SEED_VALUE", DEFAULT_ORT_TEST_RANDOM_SEED)

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
        env=runtime_env,
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


def execute_runtime_target(
    spec: RuntimeTargetSpec,
    extractor_binary: Path,
    artifacts_output: Path,
    gtest_filter: str | None,
) -> tuple[int, dict | None, dict | None]:
    """Run one extractor binary and return either a runtime JSON chunk or a failure record."""
    try:
        runtime_result = run_runtime_extractor(
            extractor_binary,
            spec.output_path,
            artifacts_output,
            gtest_filter,
        )
        runtime_stdout = runtime_result.stdout.decode("utf-8", errors="replace")
        runtime_stderr = runtime_result.stderr.decode("utf-8", errors="replace")

        if runtime_result.returncode != 0 and not spec.output_path.exists():
            raise subprocess.CalledProcessError(
                runtime_result.returncode,
                runtime_result.args,
                output=runtime_stdout,
                stderr=runtime_stderr,
            )

        runtime_chunk = load_runtime_json(spec.output_path)
    except (subprocess.CalledProcessError, ValueError) as error:
        exit_code = error.returncode if isinstance(error, subprocess.CalledProcessError) else None
        error_message = str(error)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            error_message = f"{error_message}\n{error.stderr.strip()}"
        return (
            spec.index,
            None,
            {
                "source_file": spec.source_file_relative.as_posix(),
                "exit_code": exit_code,
                "error": error_message,
            },
        )

    if runtime_result.returncode != 0:
        runtime_chunk.setdefault("warnings", []).append(
            f"gtest exited with code {runtime_result.returncode} after writing runtime JSON."
        )

    return spec.index, runtime_chunk, None


def run_runtime_targets(
    target_specs: list[RuntimeTargetSpec],
    extractor_binaries: dict[str, Path],
    artifacts_output: Path,
    gtest_filter: str | None,
    jobs: int,
) -> tuple[list[dict], list[dict]]:
    """Run all compiled extractor targets, optionally in parallel."""
    successful_chunks_by_index: dict[int, dict] = {}
    failed_files: list[dict] = []

    if jobs == 1 or len(target_specs) == 1:
        for current_file, spec in enumerate(target_specs, start=1):
            print(
                f"=== Runtime extractor [{current_file}/{len(target_specs)}] {spec.source_file_relative.as_posix()} ===",
                flush=True,
            )
            index, runtime_chunk, failed_file = execute_runtime_target(
                spec,
                extractor_binaries[spec.target_name],
                artifacts_output,
                gtest_filter,
            )
            if runtime_chunk is not None:
                successful_chunks_by_index[index] = runtime_chunk
            if failed_file is not None:
                failed_files.append(failed_file)
    else:
        max_workers = min(jobs, len(target_specs))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_by_spec = {}
            for current_file, spec in enumerate(target_specs, start=1):
                print(
                    f"=== Runtime extractor [{current_file}/{len(target_specs)}] {spec.source_file_relative.as_posix()} ===",
                    flush=True,
                )
                future = executor.submit(
                    execute_runtime_target,
                    spec,
                    extractor_binaries[spec.target_name],
                    artifacts_output,
                    gtest_filter,
                )
                future_by_spec[future] = spec

            for future in as_completed(future_by_spec):
                index, runtime_chunk, failed_file = future.result()
                if runtime_chunk is not None:
                    successful_chunks_by_index[index] = runtime_chunk
                if failed_file is not None:
                    failed_files.append(failed_file)

    merged_chunks = [
        successful_chunks_by_index[index]
        for index in range(len(target_specs))
        if index in successful_chunks_by_index
    ]
    return merged_chunks, failed_files


def filter_ignored_runtime_artifact_cases(
    runtime_chunks: list[dict],
    artifacts_output: Path,
    ignored_cases: tuple[IgnoredArtifactCase, ...],
) -> tuple[list[dict], int]:
    """Remove configured ignored artifact cases from runtime output and disk."""
    if not ignored_cases:
        return runtime_chunks, 0

    ignored_reasons = ignored_case_reasons_by_path(ignored_cases)
    filtered_chunks: list[dict] = []
    ignored_count = 0

    for runtime_chunk in runtime_chunks:
        filtered_chunk = dict(runtime_chunk)
        filtered_records = []
        warnings = list(filtered_chunk.get("warnings", []))

        for record in runtime_chunk.get("records", []):
            artifact_directory = str(record.get("artifact_directory", "")).strip().strip("/")
            if artifact_directory not in ignored_reasons:
                filtered_records.append(record)
                continue

            ignored_count += 1
            artifact_path = artifacts_output / artifact_directory
            if artifact_path.exists():
                shutil.rmtree(artifact_path)
            warnings.append(
                "Ignored generated artifact case "
                f"{artifact_directory}: {ignored_reasons[artifact_directory]}"
            )

        filtered_chunk["records"] = filtered_records
        if warnings:
            filtered_chunk["warnings"] = warnings
        filtered_chunks.append(filtered_chunk)

    return filtered_chunks, ignored_count


def run_runtime_pipeline(
    source_path: Path,
    output_path: Path,
    artifacts_output: Path,
    rebuild: bool,
    gtest_filter: str | None,
    jobs: int,
) -> None:
    """Configure, build, execute, and merge runtime extraction output."""
    artifacts_output = artifacts_output.resolve()
    cmake_binary = detect_cmake_binary()
    build_dir = repo_root() / "build" / "ort_runtime_extractor"
    source_files = runtime_source_files(source_path)

    if not source_files:
        raise FileNotFoundError(f"No C++ source files found under {source_path}")

    if rebuild and build_dir.exists():
        shutil.rmtree(build_dir)

    if source_path.is_file():
        source_root_relative = relative_to_onnxruntime_org(source_path.parent)
    else:
        source_root_relative = relative_to_onnxruntime_org(source_path)

    target_specs = write_runtime_target_manifest(
        build_dir,
        source_files,
        source_root_relative,
    )

    configure_runtime_extractor(cmake_binary, build_dir)
    extractor_binaries = build_runtime_extractors(
        cmake_binary,
        build_dir,
        target_specs,
        jobs,
    )
    merged_chunks, failed_files = run_runtime_targets(
        target_specs,
        extractor_binaries,
        artifacts_output,
        gtest_filter,
        jobs,
    )
    ignored_cases = load_ignored_artifact_cases(DEFAULT_IGNORED_CASES_PATH)
    merged_chunks, ignored_count = filter_ignored_runtime_artifact_cases(
        merged_chunks,
        artifacts_output,
        ignored_cases,
    )

    if not merged_chunks:
        raise RuntimeError("Runtime extractor did not produce any successful per-file outputs.")

    write_runtime_merged_json(
        output_path,
        source_root_relative,
        artifacts_output,
        len(target_specs),
        len(merged_chunks),
        merged_chunks,
        failed_files,
    )

    if failed_files:
        print(f"Runtime extractor skipped {len(failed_files)} source files due to build or execution failures.")
    if ignored_count:
        print(f"Runtime extractor ignored {ignored_count} generated artifact case(s) by configuration.")


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
    parser.add_argument(
        "--jobs",
        type=int,
        default=default_parallel_jobs(),
        help="Maximum parallel extractor builds and executions. Default: %(default)s.",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    return args


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
            args.jobs,
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
