"""Resolve the current harness launcher across Bazel and source runs."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def launcher_command() -> list[str]:
    source_root = Path(__file__).resolve().parent.parent
    source_launcher = (
        source_root / "bazel-bin" / "cmd" / "agent-harness"
    )
    if source_launcher.is_file() and os.access(
        source_launcher,
        os.X_OK,
    ):
        return [str(source_launcher)]
    runfiles_text = os.environ.get("RUNFILES_DIR", "")
    if runfiles_text:
        runfiles = Path(runfiles_text)
        workspace_name = os.environ.get("TEST_WORKSPACE", "_main")
        candidates = (
            runfiles / workspace_name / "cmd" / "agent-harness",
            runfiles / "_main" / "cmd" / "agent-harness",
        )
        for candidate in candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return [str(candidate)]
    for entry in sys.path:
        root = Path(entry)
        candidate = root / "cmd" / "agent-harness"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return [str(candidate)]
    invoked = Path(sys.argv[0])
    if invoked.name == "agent-harness":
        resolved = invoked.resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return [str(resolved)]
    return [sys.executable, "-m", "agent_harness.cli"]
