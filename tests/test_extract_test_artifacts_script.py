"""Unit tests for the maintainer runtime extractor orchestration script."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "scripts" / "extract_test_artifacts.py"


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


def test_helper_source_files_skips_webgpu_helpers(tmp_path: Path) -> None:
    """Skip webgpu contrib helper sources that require unavailable webgpu headers in CI."""
    module = load_script_module()
    ort_repo_root = tmp_path / "onnxruntime-org"
    ort_source_root = ort_repo_root / "onnxruntime"
    source_file = ort_source_root / "test" / "contrib_ops" / "matmul_2bits_test.cc"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text(
        '\n'.join(
            [
                '#include "contrib_ops/webgpu/quantization/matmul_nbits_common.h"',
                '#include "contrib_ops/cpu/quantization/matmul_nbits_helper.h"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    webgpu_header = ort_source_root / "contrib_ops" / "webgpu" / "quantization" / "matmul_nbits_common.h"
    webgpu_header.parent.mkdir(parents=True, exist_ok=True)
    webgpu_header.write_text("// header\n", encoding="utf-8")
    (webgpu_header.with_suffix(".cc")).write_text("// webgpu helper\n", encoding="utf-8")

    cpu_header = ort_source_root / "contrib_ops" / "cpu" / "quantization" / "matmul_nbits_helper.h"
    cpu_header.parent.mkdir(parents=True, exist_ok=True)
    cpu_header.write_text("// header\n", encoding="utf-8")
    cpu_source = cpu_header.with_suffix(".cc")
    cpu_source.write_text("// cpu helper\n", encoding="utf-8")

    helper_sources = module.helper_source_files(source_file, ort_repo_root)

    assert helper_sources == [cpu_source.resolve()]


def test_run_runtime_extractor_includes_disabled_gtest_cases(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Run disabled ORT gtests so refreshed artifacts keep intentionally disabled cases."""
    module = load_script_module()
    captured: dict[str, object] = {}

    def fake_run_logged_command(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return module.subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(module, "run_logged_command", fake_run_logged_command)

    module.run_runtime_extractor(
        tmp_path / "extractor",
        tmp_path / "runtime.json",
        tmp_path / "artifacts",
        tmp_path / "onnxruntime-org",
        "MatMulBnb4.*",
    )

    command = captured["command"]
    assert "--gtest_also_run_disabled_tests" in command
    assert "--gtest_filter=MatMulBnb4.*" in command


def test_optional_lld_linker_cmake_args_returns_lld_flags_when_available(monkeypatch) -> None:
    """Enable lld linker flags for non-Windows hosts when ld.lld is present."""
    module = load_script_module()

    monkeypatch.setattr(module.os, "name", "posix", raising=False)
    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/ld.lld" if name == "ld.lld" else None)

    assert module.optional_lld_linker_cmake_args() == [
        "-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld",
        "-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld",
        "-DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld",
    ]


def test_optional_lld_linker_cmake_args_returns_empty_without_lld(monkeypatch) -> None:
    """Skip linker override when ld.lld is unavailable."""
    module = load_script_module()

    monkeypatch.setattr(module.os, "name", "posix", raising=False)
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)

    assert module.optional_lld_linker_cmake_args() == []


def test_optional_lld_linker_cmake_args_returns_empty_on_windows(monkeypatch) -> None:
    """Skip linker override on Windows even when ld.lld is present."""
    module = load_script_module()

    monkeypatch.setattr(module.os, "name", "nt", raising=False)
    monkeypatch.setattr(module.shutil, "which", lambda _name: "C:/Program Files/LLVM/bin/ld.lld.exe")

    assert module.optional_lld_linker_cmake_args() == []


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


def test_run_logged_command_with_retries_returns_after_transient_failure(monkeypatch) -> None:
    """Retry a failing command until one attempt succeeds."""
    module = load_script_module()

    return_codes = [1, 0]
    observed_commands: list[list[str]] = []
    sleeps: list[float] = []

    def fake_run_logged_command(command: list[str], **_kwargs: object):
        observed_commands.append(command)
        return subprocess.CompletedProcess(command, return_codes.pop(0))

    monkeypatch.setattr(module, "run_logged_command", fake_run_logged_command)
    monkeypatch.setattr(module.time, "sleep", sleeps.append)

    module.run_logged_command_with_retries(["cmake", "-S", "."], attempts=3, delay_seconds=15)

    assert len(observed_commands) == 2
    assert sleeps == [15]


def test_run_logged_command_with_retries_raises_after_last_attempt(monkeypatch) -> None:
    """Surface the original exit code once every attempt failed."""
    module = load_script_module()

    attempts_seen = 0

    def fake_run_logged_command(command: list[str], **_kwargs: object):
        nonlocal attempts_seen
        attempts_seen += 1
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(module, "run_logged_command", fake_run_logged_command)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    with pytest.raises(subprocess.CalledProcessError) as error:
        module.run_logged_command_with_retries(["cmake", "-S", "."], attempts=3, delay_seconds=0)

    assert attempts_seen == 3
    assert error.value.returncode == 1


def test_configure_runtime_extractor_retries_the_cmake_configure(monkeypatch, tmp_path: Path) -> None:
    """Route the configure step through the bounded retry helper."""
    module = load_script_module()

    recorded: dict[str, object] = {}

    def fake_retry(command: list[str], *, attempts: int, delay_seconds: float) -> None:
        recorded["command"] = command
        recorded["attempts"] = attempts
        recorded["delay_seconds"] = delay_seconds

    monkeypatch.setattr(module, "run_logged_command_with_retries", fake_retry)
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)

    module.configure_runtime_extractor(Path("cmake"), tmp_path / "build", tmp_path / "onnxruntime-org")

    assert recorded["attempts"] == module.CONFIGURE_ATTEMPTS
    assert recorded["delay_seconds"] == module.CONFIGURE_RETRY_DELAY_SECONDS
    assert "-B" in recorded["command"]
