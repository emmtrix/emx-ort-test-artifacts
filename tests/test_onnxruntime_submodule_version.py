"""Checks that the ORT submodule commit matches the pinned onnxruntime version."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"
ORT_SUBMODULE_PATH = REPO_ROOT / "onnxruntime-org"
ORT_REMOTE_URL = "https://github.com/microsoft/onnxruntime.git"


def _read_pinned_onnxruntime_version() -> str:
    """Return the exact onnxruntime version pinned in requirements.txt."""
    version_pattern = re.compile(r"^onnxruntime==(?P<version>\d+\.\d+\.\d+)$")

    for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
        match = version_pattern.match(line.strip())
        if match:
            return match.group("version")

    raise AssertionError(
        f"No exact 'onnxruntime==<major>.<minor>.<patch>' pin found in {REQUIREMENTS_PATH}."
    )


def _run_git(arguments: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run git and return the completed process."""
    return subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _require_git_output(arguments: list[str], cwd: Path) -> str:
    """Run git and return stdout, failing with stderr when the command fails."""
    completed = _run_git(arguments, cwd)
    if completed.returncode != 0:
        raise AssertionError(
            f"Git command failed in {cwd}: git {' '.join(arguments)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    return completed.stdout.strip()


def _resolve_release_tag_commit(version: str) -> str:
    """Resolve the commit for the ONNX Runtime release tag matching the pinned version."""
    tag_ref = f"refs/tags/v{version}"
    peeled_tag_ref = f"{tag_ref}^{{}}"

    local_commit = _run_git(["rev-list", "-n", "1", tag_ref], ORT_SUBMODULE_PATH)
    if local_commit.returncode == 0 and local_commit.stdout.strip():
        return local_commit.stdout.strip()

    remote_commit = _run_git(
        ["ls-remote", "--tags", ORT_REMOTE_URL, tag_ref, peeled_tag_ref],
        REPO_ROOT,
    )
    if remote_commit.returncode != 0:
        pytest.skip(
            "Could not resolve the ONNX Runtime release tag from the submodule or the remote."
        )

    refs: dict[str, str] = {}
    for line in remote_commit.stdout.splitlines():
        commit, ref = line.split("\t", 1)
        refs[ref] = commit

    if peeled_tag_ref in refs:
        return refs[peeled_tag_ref]
    if tag_ref in refs:
        return refs[tag_ref]

    raise AssertionError(
        f"Could not find the ONNX Runtime release tag 'v{version}' on {ORT_REMOTE_URL}."
    )


def test_onnxruntime_submodule_commit_matches_pinned_package_version() -> None:
    """Keep the ORT submodule aligned with the exact onnxruntime package version."""
    version = _read_pinned_onnxruntime_version()
    submodule_commit = _require_git_output(["rev-parse", "HEAD"], ORT_SUBMODULE_PATH)
    release_tag_commit = _resolve_release_tag_commit(version)

    assert submodule_commit == release_tag_commit, (
        f"{ORT_SUBMODULE_PATH} is pinned to commit {submodule_commit}, "
        f"but requirements.txt pins onnxruntime=={version}, which resolves to "
        f"ONNX Runtime tag commit {release_tag_commit}. "
        "Update the submodule commit or the pinned package version so they match."
    )
