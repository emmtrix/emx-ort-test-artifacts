from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR_CPP = REPO_ROOT / "tools" / "cpp" / "runtime_extractor" / "ort_runtime_extractor.cpp"
CAPTURE_CPP = REPO_ROOT / "tools" / "cpp" / "runtime_extractor" / "ort_runtime_capture.cpp"
RUNTIME_EXTRACTOR_CMAKE = REPO_ROOT / "tools" / "cpp" / "runtime_extractor" / "CMakeLists.txt"
TARGET_SIZE_REPORT_CMAKE = (
    REPO_ROOT / "tools" / "cpp" / "runtime_extractor" / "report_target_size.cmake"
)


def test_runtime_extractor_env_uses_single_thread_global_pools() -> None:
    source = EXTRACTOR_CPP.read_text(encoding="utf-8")

    assert "threading_options.SetGlobalIntraOpNumThreads(1);" in source
    assert "threading_options.SetGlobalInterOpNumThreads(1);" in source
    assert "threading_options.SetGlobalSpinControl(0);" in source
    assert "std::make_unique<Ort::Env>(threading_options, ORT_LOGGING_LEVEL_WARNING, \"emx_ort_runtime_extractor\")" in source


def test_runtime_capture_normalizes_session_options_for_determinism() -> None:
    source = CAPTURE_CPP.read_text(encoding="utf-8")

    assert "void ApplyDeterministicSessionOptions(onnxruntime::SessionOptions& session_options)" in source
    assert "session_options.execution_mode = ExecutionMode::ORT_SEQUENTIAL;" in source
    assert "session_options.use_per_session_threads = false;" in source
    assert "session_options.intra_op_param.thread_pool_size = 1;" in source
    assert "session_options.inter_op_param.thread_pool_size = 1;" in source


def test_runtime_capture_rewrites_negative_cases_to_separate_root() -> None:
    source = CAPTURE_CPP.read_text(encoding="utf-8")

    assert "fs::path RewriteArtifactSourcePathForStorage(const CapturedRecord& record)" in source
    assert 'component->generic_string() != "onnxruntime"' in source
    assert 'fs::path rewritten("onnxruntime-negative");' in source


def test_cmake_enables_target_size_reporting() -> None:
    source = RUNTIME_EXTRACTOR_CMAKE.read_text(encoding="utf-8")

    assert "function(emx_enable_target_size_report target_name)" in source
    assert 'TARGET "${target_name}"' in source
    assert "POST_BUILD" in source
    assert '"-Dtarget_file=$<TARGET_FILE:${target_name}>"' in source
    assert (
        'if(NOT target_type MATCHES "^(EXECUTABLE|STATIC_LIBRARY|SHARED_LIBRARY|MODULE_LIBRARY)$")'
        in source
    )
    assert 'emx_enable_target_size_reports_in_directory("${CMAKE_SOURCE_DIR}")' in source


def test_report_script_validates_and_reports_size() -> None:
    source = TARGET_SIZE_REPORT_CMAKE.read_text(encoding="utf-8")

    assert 'message(FATAL_ERROR "target_name and target_file must be defined.")' in source
    assert 'message(FATAL_ERROR "Target file does not exist: ${target_file}")' in source
    assert 'file(SIZE "${target_file}" target_size_bytes)' in source
    assert 'message(STATUS "Built ${target_name}: ${target_size_bytes} bytes (${target_file})")' in source
