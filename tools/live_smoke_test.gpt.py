"""No-spend validation for the opt-in live smoke entry point."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


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
