"""Unit tests for the runtime extractor build orchestration script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "extract_test_artifacts.py"


def load_script_module():
    """Load the extraction script as an importable module for unit testing."""
    spec = importlib.util.spec_from_file_location("extract_test_artifacts_script", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_write_runtime_target_manifest_creates_aggregate_parallel_target(
    tmp_path: Path,
) -> None:
    """Generate one target manifest and verify the shared aggregate build target exists."""
    module = load_script_module()
    ort_repo_root = tmp_path / "onnxruntime-org"
    source_file = (
        ort_repo_root
        / "onnxruntime"
        / "test"
        / "contrib_ops"
        / "cdist_op_test.cc"
    )
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("OpTester test_source;\n", encoding="utf-8")

    target_specs = module.write_runtime_target_manifest(
        tmp_path / "ort_runtime_extractor",
        [source_file],
        ort_repo_root,
        Path("onnxruntime/test/contrib_ops"),
    )

    assert len(target_specs) == 1
    target_spec = target_specs[0]
    manifest_path = tmp_path / "ort_runtime_extractor" / "generated" / "emx_runtime_targets.cmake"
    manifest_text = manifest_path.read_text(encoding="utf-8")

    assert target_spec.target_name == module.runtime_target_name(0)
    assert target_spec.source_file == source_file
    assert target_spec.source_file_relative == Path(
        "onnxruntime/test/contrib_ops/cdist_op_test.cc"
    )
    assert target_spec.extra_includes_header.exists()
    assert target_spec.extra_includes_header.read_text(encoding="utf-8") == "#pragma once\n"
    assert "emx_add_runtime_extractor_target(" in manifest_text
    assert (
        "add_custom_target(ort_cpp_test_runtime_extractors DEPENDS "
        "${EMX_ORT_RUNTIME_EXTRACTOR_TARGETS})"
    ) in manifest_text


def test_resolve_cpp_source_path_maps_legacy_submodule_prefix(tmp_path: Path) -> None:
    """Accept legacy onnxruntime-org-prefixed source paths against the cloned checkout."""
    module = load_script_module()
    ort_repo_root = tmp_path / "onnxruntime-org"
    source_file = (
        ort_repo_root
        / "onnxruntime"
        / "test"
        / "contrib_ops"
        / "demo_test.cc"
    )
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("OpTester demo;\n", encoding="utf-8")

    resolved = module.resolve_cpp_source_path(
        Path("onnxruntime-org/onnxruntime/test/contrib_ops/demo_test.cc"),
        ort_repo_root,
    )

    assert resolved == source_file.resolve()


def test_default_parallel_jobs_is_never_less_than_one() -> None:
    """Keep the automatic parallelism default in a valid range."""
    module = load_script_module()
    assert module.default_parallel_jobs() >= 1


def test_parse_version_tuple_reads_cmake_versions() -> None:
    """Parse dotted CMake versions for minimum-version selection."""
    module = load_script_module()
    assert module.parse_version_tuple("cmake version 3.28.3") == (3, 28, 3)
    assert module.parse_version_tuple("invalid") is None


def test_filter_ignored_runtime_artifact_cases_removes_records_and_directories(
    tmp_path: Path,
) -> None:
    """Delete configured ignored artifact directories from generated runtime output."""
    module = load_script_module()

    kept_dir = tmp_path / "onnxruntime" / "test" / "suite" / "Keep_run0"
    ignored_dir = tmp_path / "onnxruntime" / "test" / "suite" / "Ignore_run0"
    kept_dir.mkdir(parents=True)
    ignored_dir.mkdir(parents=True)

    runtime_chunks = [
        {
            "records": [
                {"artifact_directory": "onnxruntime/test/suite/Keep_run0"},
                {"artifact_directory": "onnxruntime/test/suite/Ignore_run0"},
            ]
        }
    ]

    ignored_cases = (
        module.IgnoredArtifactCase(
            path="onnxruntime/test/suite/Ignore_run0",
            reason="Ignored for a tracked reason.",
        ),
    )

    filtered_chunks, ignored_count = module.filter_ignored_runtime_artifact_cases(
        runtime_chunks,
        tmp_path,
        ignored_cases,
    )

    assert ignored_count == 1
    assert filtered_chunks == [
        {
            "records": [{"artifact_directory": "onnxruntime/test/suite/Keep_run0"}],
            "warnings": [
                "Ignored generated artifact case onnxruntime/test/suite/Ignore_run0: "
                "Ignored for a tracked reason."
            ],
        }
    ]
    assert kept_dir.exists()
    assert not ignored_dir.exists()
