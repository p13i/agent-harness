"""No-spend validation for the opt-in live smoke entry point."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess

import pytest


def test_live_smoke_packages_daemon_launcher() -> None:
    runfiles = Path(os.environ["TEST_SRCDIR"])
    workspace = os.environ.get("TEST_WORKSPACE", "_main")
    launcher = runfiles / workspace / "cmd" / "agent-harness"

    assert launcher.is_file()
    assert os.access(launcher, os.X_OK)


def test_live_smoke_uses_bazel_callers_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _smoke_module()
    monkeypatch.setenv("BUILD_WORKING_DIRECTORY", str(tmp_path))

    assert module.default_workspace() == tmp_path

    monkeypatch.delenv("BUILD_WORKING_DIRECTORY")
    monkeypatch.chdir(tmp_path)

    assert module.default_workspace() == tmp_path


def test_live_smoke_refuses_to_invoke_without_confirmation() -> None:
    result = subprocess.run(
        [str(_launcher()), "codex"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--confirm-spend is required" in result.stderr
    assert "no provider was invoked" in result.stderr


def _launcher() -> Path:
    runfiles = Path(os.environ["TEST_SRCDIR"])
    workspace = os.environ.get("TEST_WORKSPACE", "_main")
    candidate = runfiles / workspace / "tools" / "live_smoke"
    if candidate.is_file():
        return candidate
    return runfiles / "_main" / "tools" / "live_smoke"


def _smoke_module():
    runfiles = Path(os.environ["TEST_SRCDIR"])
    workspace = os.environ.get("TEST_WORKSPACE", "_main")
    source = runfiles / workspace / "tools" / "live_smoke.gpt.py"
    spec = importlib.util.spec_from_file_location(
        "live_smoke_module",
        source,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load live smoke module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(
            [
                __file__,
                "--import-mode=importlib",
            ]
        )
    )
