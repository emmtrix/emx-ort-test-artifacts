"""Manage the local ONNX Runtime source checkout used for artifact extraction.

This module derives the required ONNX Runtime source revision from the pinned
``onnxruntime`` package version in ``requirements.txt`` and prepares a local
Git checkout for extractor scripts.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ORT_REMOTE_URL = "https://github.com/microsoft/onnxruntime.git"
PINNED_ORT_VERSION_PATTERN = re.compile(r"^onnxruntime==(?P<version>\d+\.\d+\.\d+)$")


def read_pinned_onnxruntime_version(requirements_path: Path) -> str:
    """Return the exact onnxruntime version pinned in requirements.txt."""
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        match = PINNED_ORT_VERSION_PATTERN.match(line.strip())
        if match:
            return match.group("version")

    raise ValueError(
        f"No exact 'onnxruntime==<major>.<minor>.<patch>' pin found in {requirements_path}."
    )


def onnxruntime_version_tag(version: str) -> str:
    """Return the Git tag name corresponding to one pinned package version."""
    return f"v{version}"


def default_onnxruntime_checkout_dir(repo_root: Path) -> Path:
    """Return the default local checkout directory for ONNX Runtime sources."""
    return repo_root / "build" / "onnxruntime-org"


def run_git(arguments: list[str], cwd: Path | None = None) -> None:
    """Run one Git command and raise a precise error when it fails."""
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Git command failed: "
            f"git {' '.join(arguments)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def ensure_onnxruntime_checkout(
    checkout_dir: Path,
    requirements_path: Path,
    *,
    remote_url: str = ORT_REMOTE_URL,
) -> Path:
    """Clone or update the local ONNX Runtime checkout to the pinned release tag."""
    version = read_pinned_onnxruntime_version(requirements_path)
    tag_name = onnxruntime_version_tag(version)
    checkout_dir = checkout_dir.resolve()

    if not checkout_dir.exists():
        checkout_dir.parent.mkdir(parents=True, exist_ok=True)
        run_git(
            [
                "clone",
                "--branch",
                tag_name,
                "--depth",
                "1",
                remote_url,
                str(checkout_dir),
            ]
        )
        return checkout_dir

    if not (checkout_dir / ".git").is_dir():
        raise ValueError(
            f"Expected a Git checkout at {checkout_dir}, but '.git' is missing."
        )

    run_git(["remote", "set-url", "origin", remote_url], checkout_dir)
    run_git(
        [
            "fetch",
            "--force",
            "--tags",
            "--prune",
            "--depth",
            "1",
            "origin",
            f"refs/tags/{tag_name}:refs/tags/{tag_name}",
        ],
        checkout_dir,
    )
    run_git(["checkout", "--force", "--detach", tag_name], checkout_dir)
    run_git(["reset", "--hard", tag_name], checkout_dir)
    return checkout_dir
