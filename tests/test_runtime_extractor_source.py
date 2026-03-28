from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR_CPP = REPO_ROOT / "tools" / "cpp" / "runtime_extractor" / "ort_runtime_extractor.cpp"
CAPTURE_CPP = REPO_ROOT / "tools" / "cpp" / "runtime_extractor" / "ort_runtime_capture.cpp"


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
