"""No-spend validation for the opt-in live smoke entry point."""

from __future__ import annotations

import asyncio
import importlib.util
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

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
        [str(_launcher())],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "--confirm-spend is required" in result.stderr
    assert "no provider was invoked" in result.stderr


def test_live_smoke_uses_exactly_one_turn_per_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _smoke_module()
    providers: list[str] = []
    stopped: list[str] = []

    class Client:
        raw = object()

        def __init__(self, unused_paths: object) -> None:
            del unused_paths

        async def create_session(
            self,
            workspace: Path,
            **values: object,
        ) -> dict[str, str]:
            assert workspace == tmp_path
            assert values["permission_mode"] == "read-only"
            assert values["execution_profile"] == "live-smoke"
            return {"session_id": "session-1"}

        async def send_message(
            self,
            session_id: str,
            prompt: str,
            *,
            provider: str,
            effort: str,
            workload: str,
        ) -> SimpleNamespace:
            assert session_id == "session-1"
            assert prompt == module.PROVIDER_PROMPTS[provider]
            assert effort == "low"
            assert workload == "operations"
            providers.append(provider)
            return SimpleNamespace(command_id="command-" + provider)

        async def usage(self, session_id: str) -> dict[str, object]:
            assert session_id == "session-1"
            return {
                "envelopes": [
                    {
                        "profile": "live-smoke",
                        "limits": {"max_attempts": 1},
                    },
                    {
                        "profile": "live-smoke",
                        "limits": {"max_attempts": 1},
                    },
                ]
            }

        async def command(self, session_id: str, action: str) -> None:
            assert action == "stop"
            stopped.append(session_id)

    async def ready(unused_paths: object) -> None:
        del unused_paths

    async def complete(
        unused_raw: object,
        command_id: str,
        *,
        timeout: float,
    ) -> dict[str, object]:
        del unused_raw
        assert timeout == 330
        provider = command_id.removeprefix("command-")
        return {
            "status": "complete",
            "result": {"native_session_id": "native-" + provider},
        }

    monkeypatch.setattr(module, "ensure_daemon", ready)
    monkeypatch.setattr(module, "AgentHarnessClient", Client)
    monkeypatch.setattr(module, "wait_command", complete)
    result = asyncio.run(
        module.run(
            SimpleNamespace(
                confirm_spend=True,
                state_dir=tmp_path / "state",
                workspace=tmp_path,
            )
        )
    )

    assert providers == ["claude", "codex"]
    assert result["commands"] == ["command-claude", "command-codex"]
    assert result["native_sessions"] == {
        "claude": "native-claude",
        "codex": "native-codex",
    }
    assert stopped == ["session-1"]


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
